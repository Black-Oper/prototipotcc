"""
Grid Search de Hiperparâmetros — RTDVSR
========================================
Roda um smoke test de 3 épocas para cada combinação de hiperparâmetros
e salva os resultados em grid_search_results.csv.

Uso:
    python grid_search.py

Ao final, imprime as 5 melhores combinações por PSNR de validação
e salva o CSV completo para análise.
"""

import os
import csv
import time
import math
import random
import itertools
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import get_model, get_interface
from train import (
    VimeoSeptupletDataset,
    CharbonnierLoss,
    _detach_state,
    _resolve_vimeo_paths,
    calc_psnr,
    calc_ssim,
)
from utils.config import ConfigManager

# ---------------------------------------------------------------------------
# Espaço de busca
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "learning_rate":         [1e-3, 2e-4, 5e-5],
    "hidden_dim":            [48, 64, 96],
    "num_res_blocks":        [3, 4, 5],
    "temporal_loss_weight":  [0.05, 0.1, 0.2],
}

# Hiperparâmetros fixos (não variam na busca)
FIXED = {
    "model_type":   "RTDVSR",
    "scale_factor": 2,
    "seq_len":      3,
    "batch_size":   4,
    "epochs":       3,          # smoke test: 3 épocas
    "dataset_path": "./datasets",
    "max_val_samples": 100,     # limita validação para acelerar
}

OUTPUT_CSV = "grid_search_results.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calc_psnr(img1, img2):
    img1 = torch.clamp(img1, 0., 1.)
    img2 = torch.clamp(img2, 0., 1.)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())


def run_smoke(combo, device):
    """
    Treina o RTDVSR por 3 épocas com a combinação de hiperparâmetros dada.
    Retorna (val_psnr, val_ssim, train_loss, elapsed_seconds).
    """
    lr            = combo["learning_rate"]
    hidden_dim    = combo["hidden_dim"]
    num_res_blocks = combo["num_res_blocks"]
    temp_weight   = combo["temporal_loss_weight"]
    scale         = FIXED["scale_factor"]
    seq_len       = FIXED["seq_len"]
    batch_size    = FIXED["batch_size"]
    epochs        = FIXED["epochs"]
    data_path     = FIXED["dataset_path"]
    max_val       = FIXED["max_val_samples"]
    model_type    = FIXED["model_type"]

    # Dataset
    vimeo_seq, vimeo_train, vimeo_test = _resolve_vimeo_paths(data_path)
    if not os.path.exists(vimeo_seq) or not os.path.exists(vimeo_train):
        raise FileNotFoundError(
            f"Vimeo Septuplet não encontrado em: {vimeo_seq}\n"
            "Execute 'Baixar dataset' no menu do app.py primeiro."
        )

    train_ds = VimeoSeptupletDataset(
        vimeo_seq, vimeo_train,
        scale_factor=scale, seq_len=seq_len, train=True)
    val_ds = VimeoSeptupletDataset(
        vimeo_seq, vimeo_test if os.path.exists(vimeo_test) else vimeo_train,
        scale_factor=scale, seq_len=seq_len, train=False)

    is_cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=is_cuda)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2)

    # Modelo
    model = get_model(
        model_type,
        scale_factor=scale,
        channels=3,
        hidden_dim=hidden_dim,
        num_res_blocks=num_res_blocks,
    ).to(device)
    interface = get_interface(model_type)

    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Precisão automática
    if is_cuda and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif is_cuda:
        amp_dtype = torch.float16
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_scaler = False

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    t0 = time.perf_counter()
    last_train_loss = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for lr_seq, hr_seq in train_loader:
            lr_seq = lr_seq.to(device, non_blocking=True)
            hr_seq = hr_seq.to(device, non_blocking=True)
            T = lr_seq.shape[1]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                recon_loss = torch.zeros(1, device=device, dtype=torch.float32)
                temp_loss  = torch.zeros(1, device=device, dtype=torch.float32)
                state    = None
                prev_sr  = None
                prev_hr  = None

                for t in range(T):
                    state = _detach_state(state)
                    sr_frame, state = model(lr_seq[:, t], state)
                    recon_loss = recon_loss + criterion(sr_frame, hr_seq[:, t])

                    if prev_sr is not None:
                        sr_diff = sr_frame - prev_sr
                        hr_diff = hr_seq[:, t] - prev_hr
                        temp_loss = temp_loss + torch.mean((sr_diff - hr_diff) ** 2)

                    prev_sr = sr_frame.detach()
                    prev_hr = hr_seq[:, t]

                recon_loss = recon_loss / T
                temp_loss  = temp_loss / max(T - 1, 1)
                total_loss = recon_loss + temp_weight * temp_loss

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

            epoch_loss += total_loss.item()
            n_batches  += 1

        scheduler.step()
        last_train_loss = epoch_loss / max(n_batches, 1)

    # Validação
    model.eval()
    val_psnr  = 0.0
    val_ssim  = 0.0
    val_count = 0
    ssim_count = 0

    with torch.no_grad():
        for i, (lr_seq, hr_seq) in enumerate(val_loader):
            if i >= max_val:
                break
            lr_seq = lr_seq.to(device)
            hr_seq = hr_seq.to(device)
            T = lr_seq.shape[1]
            state = None
            for t in range(T):
                sr_frame, state = model(lr_seq[:, t], state)
                val_psnr += _calc_psnr(sr_frame, hr_seq[:, t])
                val_count += 1
                if i < 50:  # SSIM apenas nas primeiras 50 sequências (caro)
                    s = calc_ssim(sr_frame, hr_seq[:, t])
                    if s is not None:
                        val_ssim  += s
                        ssim_count += 1

    elapsed = time.perf_counter() - t0
    avg_psnr = val_psnr / max(val_count, 1)
    avg_ssim = val_ssim / max(ssim_count, 1) if ssim_count > 0 else float("nan")

    return avg_psnr, avg_ssim, last_train_loss, elapsed


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Gera todas as combinações
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total  = len(combos)
    print(f"\nTotal de combinações: {total}")
    print(f"Épocas por combinação: {FIXED['epochs']} (smoke test)\n")

    results = []
    csv_path = OUTPUT_CSV

    # Cabeçalho do CSV
    fieldnames = keys + ["val_psnr", "val_ssim", "train_loss", "elapsed_s", "timestamp"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for i, combo in enumerate(combos, 1):
        desc = " | ".join(f"{k}={v}" for k, v in combo.items())
        print(f"[{i:02d}/{total}] {desc}")

        try:
            psnr, ssim, loss, elapsed = run_smoke(combo, device)
            ssim_str = f"{ssim:.4f}" if not math.isnan(ssim) else "N/A"
            print(f"  → PSNR: {psnr:.2f} dB | SSIM: {ssim_str} | "
                  f"Loss: {loss:.5f} | Tempo: {elapsed:.0f}s\n")

            row = dict(combo)
            row["val_psnr"]   = round(psnr, 4)
            row["val_ssim"]   = round(ssim, 4) if not math.isnan(ssim) else ""
            row["train_loss"] = round(loss, 6)
            row["elapsed_s"]  = round(elapsed, 1)
            row["timestamp"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append(row)

        except Exception as e:
            print(f"  → ERRO: {e}\n")
            row = dict(combo)
            row["val_psnr"]   = ""
            row["val_ssim"]   = ""
            row["train_loss"] = ""
            row["elapsed_s"]  = ""
            row["timestamp"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append(row)

        # Salva resultado parcial a cada iteração (não perde progresso)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    # Ranking final
    valid = [r for r in results if r["val_psnr"] != ""]
    valid.sort(key=lambda r: r["val_psnr"], reverse=True)

    print("\n" + "="*70)
    print("TOP 5 COMBINAÇÕES (por PSNR de validação):")
    print("="*70)
    for rank, r in enumerate(valid[:5], 1):
        print(f"\n#{rank} — PSNR: {r['val_psnr']:.2f} dB | SSIM: {r['val_ssim']}")
        for k in keys:
            print(f"    {k}: {r[k]}")

    print(f"\nResultados completos salvos em: {csv_path}")
    print("\nPróximo passo: use a combinação #1 no config.json e rode o treino completo.")


if __name__ == "__main__":
    main()