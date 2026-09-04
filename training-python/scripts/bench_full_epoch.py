"""
Benchmark de throughput do treino "real" (mesmo loop de train.py, mesmo
DataLoader com num_workers=4) usando o dataset sintético gerado por
bench_gen_synthetic.py — evita baixar o Vimeo Septuplet real (~88 GB) só
para medir tempo de batch nesta máquina.

Mede segundos/batch de treino (forward + backward + optimizer step) e de
validação (forward only), com a config final decidida em presets/config.json
(hidden_dim=64, num_res_blocks=4, lr=0.0014), e extrapola para uma época
completa do Vimeo Septuplet real (64.612 sequências / batch_size).

Uso (a partir de training-python/, depois de bench_gen_synthetic.py):
    python scripts/bench_full_epoch.py [--warmup 10] [--batches 150]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import get_model, get_interface
from train import VimeoSeptupletDataset, CharbonnierLoss, _detach_state, calc_psnr
from utils.config import ConfigManager

REAL_TRAIN_SEQUENCES = 64_612  # sep_trainlist.txt oficial do Vimeo Septuplet
REAL_VAL_SEQUENCES = 500       # max_val_samples default de train.py

BENCH_ROOT = Path("datasets/vimeo_septuplet_bench")


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device):
    """Espera a GPU terminar antes de marcar o tempo — sem isso, o Python
    segue em frente enquanto o trabalho ainda está na fila da GPU (CUDA e
    MPS são assíncronos), e a medição de tempo fica errada."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10,
                        help="Batches de aquecimento (não cronometrados)")
    parser.add_argument("--batches", type=int, default=150,
                        help="Batches de treino cronometrados")
    parser.add_argument("--val-batches", type=int, default=60,
                        help="Batches de validação cronometrados")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Sobrepõe o batch_size do config.json só para este benchmark "
                             "(não altera o arquivo real)")
    args = parser.parse_args()

    if not BENCH_ROOT.exists():
        print(f"Dataset sintético não encontrado em {BENCH_ROOT}. "
              "Rode scripts/bench_gen_synthetic.py primeiro.")
        return

    ConfigManager.load_config("presets/config.json")
    config = ConfigManager.get_instance()

    scale = config.get("scale_factor", 2)
    seq_len = config.get("seq_len", 3)
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 4)
    lr = config.get("learning_rate", 1e-3)
    model_type = config.get("model_type", "RTDVSR")
    model_params = config.get("model_params", {"hidden_dim": 64, "num_res_blocks": 4})
    temporal_weight = config.get("temporal_loss_weight", 0.1)
    mask_weight = config.get("mask_loss_weight", 0.05)

    device = _select_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("GPU: Apple Silicon (Metal / MPS)")
    print(f"Config final: model={model_type} params={model_params} lr={lr} "
          f"batch_size={batch_size} seq_len={seq_len}\n")

    train_ds = VimeoSeptupletDataset(
        str(BENCH_ROOT / "sequences"), str(BENCH_ROOT / "sep_trainlist.txt"),
        scale_factor=scale, seq_len=seq_len, train=True)
    val_ds = VimeoSeptupletDataset(
        str(BENCH_ROOT / "sequences"), str(BENCH_ROOT / "sep_trainlist.txt"),
        scale_factor=scale, seq_len=seq_len, train=False)

    is_cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=is_cuda, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, persistent_workers=True)

    interface = get_interface(model_type)
    model = get_model(model_type, scale_factor=scale, channels=3, **model_params).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros do modelo: {num_params:,} | Interface: {interface}\n")

    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if is_cuda and torch.cuda.is_bf16_supported():
        amp_dtype, use_scaler = torch.bfloat16, False
    elif is_cuda:
        amp_dtype, use_scaler = torch.float16, True
    else:
        amp_dtype, use_scaler = torch.float32, False
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def _train_step(lr_seq, hr_seq):
        lr_seq = lr_seq.to(device, non_blocking=True)
        hr_seq = hr_seq.to(device, non_blocking=True)
        T = lr_seq.shape[1]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            recon_loss = torch.zeros(1, device=device, dtype=torch.float32)
            temp_loss = torch.zeros(1, device=device, dtype=torch.float32)
            mask_loss = torch.zeros(1, device=device, dtype=torch.float32)
            state = prev_sr = prev_hr = None
            for t in range(T):
                state = _detach_state(state)
                sr_frame, state = model(lr_seq[:, t], state)
                recon_loss = recon_loss + criterion(sr_frame, hr_seq[:, t])
                if hasattr(model, "get_mask_loss"):
                    mask_loss = mask_loss + model.get_mask_loss()
                if prev_sr is not None:
                    sr_diff = sr_frame - prev_sr
                    hr_diff = hr_seq[:, t] - prev_hr
                    temp_loss = temp_loss + torch.mean((sr_diff - hr_diff) ** 2)
                prev_sr, prev_hr = sr_frame.detach(), hr_seq[:, t]
            recon_loss = recon_loss / T
            temp_loss = temp_loss / max(T - 1, 1)
            mask_loss = mask_loss / T
            total_loss = recon_loss + temporal_weight * temp_loss + mask_weight * mask_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad(set_to_none=True)
            return
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

    # --- Aquecimento (workers subindo, cuDNN autotune, etc. — não cronometrado) ---
    print(f"Aquecendo ({args.warmup} batches, não cronometrado)...")
    model.train()
    it = iter(train_loader)
    for _ in range(args.warmup):
        lr_seq, hr_seq = next(it)
        _train_step(lr_seq, hr_seq)
    _sync(device)

    # --- Treino cronometrado (iterador novo: warmup pode ter consumido o
    # resto da "época" do dataset sintético, e persistent_workers=True já
    # deixa isso barato, igual ao início de época real do train.py) ---
    print(f"Cronometrando {args.batches} batches de treino...")
    it = iter(train_loader)
    t0 = time.perf_counter()
    for _ in range(args.batches):
        lr_seq, hr_seq = next(it)
        _train_step(lr_seq, hr_seq)
    _sync(device)
    train_elapsed = time.perf_counter() - t0
    sec_per_batch = train_elapsed / args.batches
    print(f"  {train_elapsed:.1f}s para {args.batches} batches -> "
          f"{sec_per_batch:.3f} s/batch (treino)\n")

    # --- Validação cronometrada (forward only) ---
    print(f"Cronometrando {args.val_batches} batches de validação...")
    model.eval()
    val_it = iter(val_loader)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.val_batches):
            lr_seq, hr_seq = next(val_it)
            lr_seq, hr_seq = lr_seq.to(device), hr_seq.to(device)
            T = lr_seq.shape[1]
            state = None
            for t in range(T):
                sr_frame, state = model(lr_seq[:, t], state)
                calc_psnr(sr_frame, hr_seq[:, t])
    _sync(device)
    val_elapsed = time.perf_counter() - t0
    sec_per_val_batch = val_elapsed / args.val_batches
    print(f"  {val_elapsed:.1f}s para {args.val_batches} batches -> "
          f"{sec_per_val_batch:.3f} s/batch (validação)\n")

    # --- Extrapolação para a época completa real ---
    real_train_batches = REAL_TRAIN_SEQUENCES / batch_size
    real_val_batches = REAL_VAL_SEQUENCES  # val_loader usa batch_size=1

    train_time_full = real_train_batches * sec_per_batch
    val_time_full = real_val_batches * sec_per_val_batch
    total_time_full = train_time_full + val_time_full

    print("=" * 70)
    print("EXTRAPOLAÇÃO PARA UMA ÉPOCA COMPLETA DO VIMEO SEPTUPLET REAL")
    print("=" * 70)
    print(f"  Batches de treino/época (real): {real_train_batches:.0f}")
    print(f"  Tempo de treino/época: {train_time_full/3600:.2f} h")
    print(f"  Tempo de validação/época: {val_time_full/60:.1f} min")
    print(f"  TOTAL POR ÉPOCA: {total_time_full/3600:.2f} h")


if __name__ == "__main__":
    main()
