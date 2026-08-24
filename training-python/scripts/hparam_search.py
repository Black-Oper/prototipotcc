"""
Busca de Hiperparâmetros — RTDVSR (Optuna + pruning)
=====================================================
Substitui o grid search exaustivo por uma busca guiada (TPE, via Optuna):
cada trial reporta o PSNR de validação ao final de cada época, e trials
claramente piores que a mediana dos trials anteriores no mesmo ponto são
podados antes de gastar o orçamento completo de épocas.

Cada época de treino é limitada a `max_train_batches` batches (não a
época inteira do Vimeo Septuplet) — é isso que torna cada trial rápido o
bastante para valer a pena rodar dezenas deles.

Uso (a partir de training-python/):
    python scripts/hparam_search.py
    python scripts/hparam_search.py --n-trials 50 --epochs 4

O progresso é salvo incrementalmente em rtdvsr_hparam_search.db (SQLite)
— interromper (Ctrl+C) e rodar de novo retoma o estudo em vez de
recomeçar do zero. Ao final, imprime os 5 melhores trials e exporta
rtdvsr_hparam_search_results.csv.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import get_model, get_interface
from train import (
    VimeoSeptupletDataset,
    CharbonnierLoss,
    _detach_state,
    _resolve_vimeo_paths,
    calc_psnr,
    calc_ssim,
)

# ---------------------------------------------------------------------------
# Configuração fixa (não faz parte da busca)
# ---------------------------------------------------------------------------
FIXED = {
    "model_type":        "RTDVSR",
    "scale_factor":       2,
    "seq_len":            3,
    "batch_size":         4,
    "dataset_path":       "./datasets",
    "max_val_samples":    100,   # amostras de validação por época
    "max_train_batches":  500,   # batches de treino por época — é o que torna o trial rápido
    "seed":               42,    # mesmo seed em todo trial: isola o efeito do hiperparâmetro
}

STUDY_NAME = "rtdvsr_hparam_search"
STORAGE = f"sqlite:///{STUDY_NAME}.db"
OUTPUT_CSV = f"{STUDY_NAME}_results.csv"


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_datasets(data_path, scale, seq_len):
    vimeo_seq, vimeo_train, vimeo_test = _resolve_vimeo_paths(data_path)
    if not os.path.exists(vimeo_seq) or not os.path.exists(vimeo_train):
        raise FileNotFoundError(
            f"Vimeo Septuplet não encontrado em: {vimeo_seq}\n"
            "Execute 'Baixar dataset' no menu do app.py primeiro."
        )
    train_ds = VimeoSeptupletDataset(
        vimeo_seq, vimeo_train, scale_factor=scale, seq_len=seq_len, train=True)
    val_ds = VimeoSeptupletDataset(
        vimeo_seq, vimeo_test if os.path.exists(vimeo_test) else vimeo_train,
        scale_factor=scale, seq_len=seq_len, train=False)
    return train_ds, val_ds


def _validate(model, val_loader, device, max_samples, ssim_budget=50):
    model.eval()
    val_psnr = val_ssim = 0.0
    val_count = ssim_count = 0
    with torch.no_grad():
        for i, (lr_seq, hr_seq) in enumerate(val_loader):
            if i >= max_samples:
                break
            lr_seq, hr_seq = lr_seq.to(device), hr_seq.to(device)
            T = lr_seq.shape[1]
            state = None
            for t in range(T):
                sr_frame, state = model(lr_seq[:, t], state)
                val_psnr += calc_psnr(sr_frame, hr_seq[:, t])
                val_count += 1
                if i < ssim_budget:
                    s = calc_ssim(sr_frame, hr_seq[:, t])
                    if s is not None:
                        val_ssim += s
                        ssim_count += 1
    avg_psnr = val_psnr / max(val_count, 1)
    avg_ssim = val_ssim / max(ssim_count, 1) if ssim_count > 0 else float("nan")
    return avg_psnr, avg_ssim


def _train_config(hidden_dim, num_res_blocks, lr, temporal_weight,
                   train_ds, val_ds, device, epochs, trial=None,
                   on_epoch_end=None):
    """Treina uma configuração de hiperparâmetros por `epochs` épocas
    (cada uma limitada a `FIXED["max_train_batches"]` batches) e retorna
    (val_psnr, val_ssim) da última época.

    `trial`: se fornecido (optuna.Trial), reporta progresso e permite poda
    (usado pela busca de hiperparâmetros). Se None, roda todas as épocas
    sem poda (usado pela validação k-fold).
    `on_epoch_end(epoch, val_psnr, val_ssim)`: se fornecido, é chamado ao
    final de cada época — usado para logar métricas por época.
    """
    _set_seed(FIXED["seed"])

    scale = FIXED["scale_factor"]
    batch_size = FIXED["batch_size"]
    max_train_batches = FIXED["max_train_batches"]
    model_type = FIXED["model_type"]

    interface = get_interface(model_type)
    if interface != "recurrent":
        raise ValueError(
            f"hparam_search.py só suporta interface 'recurrent', "
            f"{model_type} usa '{interface}'.")

    is_cuda = device.type == "cuda"
    # num_workers=0: cada trial cria DataLoaders novos: workers de processo
    # (Windows/spawn) teriam custo de start-up alto demais pra ~500 batches.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=is_cuda)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = get_model(model_type, scale_factor=scale, channels=3,
                      hidden_dim=hidden_dim, num_res_blocks=num_res_blocks).to(device)

    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if is_cuda and torch.cuda.is_bf16_supported():
        amp_dtype, use_scaler = torch.bfloat16, False
    elif is_cuda:
        amp_dtype, use_scaler = torch.float16, True
    else:
        amp_dtype, use_scaler = torch.float32, False
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    val_psnr = 0.0
    val_ssim = float("nan")

    for epoch in range(epochs):
        model.train()
        for batch_i, (lr_seq, hr_seq) in enumerate(train_loader):
            if batch_i >= max_train_batches:
                break
            lr_seq = lr_seq.to(device, non_blocking=True)
            hr_seq = hr_seq.to(device, non_blocking=True)
            T = lr_seq.shape[1]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                recon_loss = torch.zeros(1, device=device, dtype=torch.float32)
                temp_loss = torch.zeros(1, device=device, dtype=torch.float32)
                state = prev_sr = prev_hr = None
                for t in range(T):
                    state = _detach_state(state)
                    sr_frame, state = model(lr_seq[:, t], state)
                    recon_loss = recon_loss + criterion(sr_frame, hr_seq[:, t])
                    if prev_sr is not None:
                        sr_diff = sr_frame - prev_sr
                        hr_diff = hr_seq[:, t] - prev_hr
                        temp_loss = temp_loss + torch.mean((sr_diff - hr_diff) ** 2)
                    prev_sr, prev_hr = sr_frame.detach(), hr_seq[:, t]
                recon_loss = recon_loss / T
                temp_loss = temp_loss / max(T - 1, 1)
                total_loss = recon_loss + temporal_weight * temp_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        scheduler.step()

        val_psnr, val_ssim = _validate(model, val_loader, device, FIXED["max_val_samples"])

        if on_epoch_end is not None:
            on_epoch_end(epoch, val_psnr, val_ssim)

        if trial is not None:
            trial.report(val_psnr, epoch)
            trial.set_user_attr("ssim", None if np.isnan(val_ssim) else round(val_ssim, 4))
            if trial.should_prune():
                raise optuna.TrialPruned()

    return val_psnr, val_ssim


def objective(trial, train_ds, val_ds, device, epochs):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [48, 64, 96])
    num_res_blocks = trial.suggest_int("num_res_blocks", 3, 5)
    temporal_weight = trial.suggest_float("temporal_loss_weight", 0.05, 0.2)

    val_psnr, _ = _train_config(hidden_dim, num_res_blocks, lr, temporal_weight,
                                train_ds, val_ds, device, epochs, trial=trial)
    return val_psnr


def _export_csv(study, path):
    param_keys = sorted({k for t in study.trials for k in t.params.keys()})
    fieldnames = ["trial", "state", "val_psnr", "val_ssim"] + param_keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in study.trials:
            row = {
                "trial": t.number,
                "state": t.state.name,
                "val_psnr": round(t.value, 4) if t.value is not None else "",
                "val_ssim": t.user_attrs.get("ssim", ""),
            }
            row.update(t.params)
            writer.writerow(row)
    print(f"\nResultados completos salvos em: {path}")


def _print_top(study, top_k=5):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    print("\n" + "=" * 70)
    print(f"TOP {min(top_k, len(completed))} TRIALS (por PSNR de validação):")
    print("=" * 70)
    for rank, t in enumerate(completed[:top_k], 1):
        ssim = t.user_attrs.get("ssim")
        ssim_str = f"{ssim:.4f}" if ssim is not None else "N/A"
        print(f"\n#{rank} — Trial {t.number} — PSNR: {t.value:.2f} dB | SSIM: {ssim_str}")
        for k, v in t.params.items():
            print(f"    {k}: {v}")

    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"\n{len(completed)} trials completos, {pruned} podados (pruned).")
    print("Próximo passo: use os hiperparâmetros do #1 no config.json e rode o treino completo.")


def main():
    parser = argparse.ArgumentParser(description="Busca de hiperparâmetros do RTDVSR com Optuna")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Quantidade de trials a rodar nesta chamada (padrão: 30)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Épocas por trial, sujeitas a pruning (padrão: 3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, val_ds = _load_datasets(
        FIXED["dataset_path"], FIXED["scale_factor"], FIXED["seq_len"])

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE,
        direction="maximize", pruner=pruner, load_if_exists=True,
    )

    if study.trials:
        print(f"Retomando estudo existente: {len(study.trials)} trial(s) já em {STORAGE}")

    study.optimize(
        lambda trial: objective(trial, train_ds, val_ds, device, args.epochs),
        n_trials=args.n_trials,
    )

    _export_csv(study, OUTPUT_CSV)
    _print_top(study)


if __name__ == "__main__":
    main()
