"""
Comparação de modelos VSR.

Avaliação quantitativa (PSNR / SSIM) e visual (matplotlib) de múltiplos
checkpoints sobre amostras do dataset de validação.
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import questionary

from models import get_model, get_interface
from utils.config import ConfigManager


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def _calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1 = torch.clamp(img1, 0., 1.)
    img2 = torch.clamp(img2, 0., 1.)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())


def _calc_ssim(img1_np: np.ndarray, img2_np: np.ndarray):
    """SSIM usando scikit-image. Retorna None se não disponível."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    except ImportError:
        return None


def _calc_temporal_consistency(sr_frames: list, hr_frames: list) -> float:
    """
    Temporal Consistency Error (TCE).
    Mede a diferença entre transições temporais SR e HR.
    Quanto menor, mais temporalmente coerente.
    """
    if len(sr_frames) < 2:
        return 0.0
    tce_sum = 0.0
    for i in range(1, len(sr_frames)):
        sr_diff = sr_frames[i] - sr_frames[i - 1]
        hr_diff = hr_frames[i] - hr_frames[i - 1]
        tce_sum += torch.mean((sr_diff - hr_diff) ** 2).item()
    return tce_sum / (len(sr_frames) - 1)


# ---------------------------------------------------------------------------
# Carregamento de modelo via registry
# ---------------------------------------------------------------------------
def _load_checkpoint(path: str, device: torch.device):
    """
    Carrega modelo do checkpoint usando o registry.
    Suporta formato novo (model_type + model_params) e antigo (hidden_dim direto).
    Retorna: (model, interface, scale, label)
    """
    checkpoint = torch.load(path, map_location=device)

    if not (isinstance(checkpoint, dict) and 'model' in checkpoint):
        raise ValueError(f"Formato de checkpoint inválido: {path}")

    model_type = checkpoint.get('model_type', 'LightweightVSR')

    model_params = checkpoint.get('model_params')
    if not model_params:
        # backward compat: formato antigo armazenava hidden_dim diretamente
        model_params = {}
        if 'hidden_dim' in checkpoint:
            model_params['hidden_dim'] = checkpoint['hidden_dim']
        if 'num_res_blocks' in checkpoint:
            model_params['num_res_blocks'] = checkpoint['num_res_blocks']

    saved_config = checkpoint.get('config', {})
    scale = saved_config.get('scale_factor', 2)
    epoch = checkpoint.get('epoch', '?')
    psnr = checkpoint.get('psnr', 0.0)

    model = get_model(model_type, scale_factor=scale, channels=3, **model_params)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    interface = get_interface(model_type)
    label = f"{model_type} ep{epoch} ({psnr:.2f}dB)"
    return model, interface, scale, label


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------
def _run_inference(model, interface: str, lr_seq: torch.Tensor,
                   device: torch.device,
                   return_all_frames: bool = False) -> torch.Tensor:
    """
    Roda inferência.
    Se return_all_frames=True, retorna lista de todos os frames SR.
    Caso contrário, retorna apenas o último frame.
    """
    lr_seq = lr_seq.to(device)
    T = lr_seq.shape[1]
    with torch.no_grad():
        if interface == "recurrent":
            state = None
            sr_frames = []
            for t in range(T):
                sr, state = model(lr_seq[:, t], state)
                sr_frames.append(sr)
            return sr_frames if return_all_frames else sr_frames[-1]
        else:  # sliding_window
            sr = model(lr_seq)
            return [sr] if return_all_frames else sr


# ---------------------------------------------------------------------------
# Seleção interativa de checkpoints
# ---------------------------------------------------------------------------
def _select_checkpoints(ckpt_dir: str) -> list:
    if not os.path.exists(ckpt_dir):
        print(f"Diretório de checkpoints não encontrado: {ckpt_dir}")
        return []

    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pth'))
    if not files:
        print("Nenhum checkpoint (.pth) encontrado.")
        return []

    selected = questionary.checkbox(
        "Selecione os checkpoints para comparar:",
        choices=files
    ).ask()

    return [os.path.join(ckpt_dir, f) for f in (selected or [])]


# ---------------------------------------------------------------------------
# Avaliação quantitativa
# ---------------------------------------------------------------------------
def _evaluate(models_info: list, val_loader, device: torch.device,
              seq_len: int, max_samples: int = 200) -> dict:
    results = {
        info["label"]: {"psnr": 0.0, "ssim": 0.0, "ssim_count": 0,
                         "tce": 0.0, "tce_count": 0, "count": 0}
        for info in models_info
    }

    for i, (lr_seq, hr_seq) in enumerate(val_loader):
        if i >= max_samples:
            break

        hr_center = hr_seq[:, seq_len // 2].to(device)

        for info in models_info:
            sr_all = _run_inference(info["model"], info["interface"],
                                   lr_seq, device, return_all_frames=True)
            sr = sr_all[-1]  # último frame para PSNR/SSIM
            psnr = _calc_psnr(sr, hr_center)

            sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            hr_np = hr_center.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            ssim_val = _calc_ssim(sr_np, hr_np)

            entry = results[info["label"]]
            entry["psnr"] += psnr
            entry["count"] += 1
            if ssim_val is not None:
                entry["ssim"] += ssim_val
                entry["ssim_count"] += 1

            # Temporal Consistency Error
            if len(sr_all) >= 2:
                hr_frames = [hr_seq[:, t].to(device) for t in range(hr_seq.shape[1])]
                tce = _calc_temporal_consistency(sr_all, hr_frames)
                entry["tce"] += tce
                entry["tce_count"] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{min(max_samples, len(val_loader))} amostras avaliadas...")

    return results


def _print_table(results: dict):
    has_ssim = any(v["ssim_count"] > 0 for v in results.values())
    has_tce = any(v["tce_count"] > 0 for v in results.values())
    width = 55
    if has_ssim:
        width += 15
    if has_tce:
        width += 15
    print("\n" + "=" * width)
    header = f"{'Modelo':<40} {'PSNR (dB)':>12}"
    if has_ssim:
        header += f"  {'SSIM':>8}"
    if has_tce:
        header += f"  {'TCE':>10}"
    print(header)
    print("-" * width)
    for label, res in results.items():
        n = max(res["count"], 1)
        avg_psnr = res["psnr"] / n
        row = f"{label:<40} {avg_psnr:>12.2f}"
        if has_ssim:
            ssim_str = f"{res['ssim'] / res['ssim_count']:.4f}" if res["ssim_count"] > 0 else "  N/A"
            row += f"  {ssim_str:>8}"
        if has_tce:
            tce_str = f"{res['tce'] / res['tce_count']:.6f}" if res["tce_count"] > 0 else "  N/A"
            row += f"  {tce_str:>10}"
        print(row)
    print("=" * width)


# ---------------------------------------------------------------------------
# Comparação visual
# ---------------------------------------------------------------------------
def _visual_comparison(models_info: list, val_loader, device: torch.device,
                       seq_len: int, n_samples: int, output_path: str = "comparison.png"):
    samples = []
    for lr_seq, hr_seq in val_loader:
        if len(samples) >= n_samples:
            break
        samples.append((lr_seq, hr_seq[:, seq_len // 2]))

    n_cols = 2 + len(models_info)  # Bicubic | modelo1 | ... | HR
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)

    col_titles = (["Bicubic (referência)"]
                  + [info["label"] for info in models_info]
                  + ["HR (ground truth)"])
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=9)

    for row, (lr_seq, hr_img) in enumerate(samples):
        hr_np = hr_img.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        h_hr, w_hr = hr_np.shape[:2]

        # Bicubic do frame central
        lr_center = lr_seq[:, seq_len // 2]
        lr_pil = TF.to_pil_image(lr_center.squeeze(0).clamp(0, 1))
        bicubic_np = np.array(lr_pil.resize((w_hr, h_hr), Image.BICUBIC)) / 255.0

        axes[row, 0].imshow(bicubic_np.clip(0, 1))
        axes[row, 0].axis('off')

        for col, info in enumerate(models_info, start=1):
            sr = _run_inference(info["model"], info["interface"], lr_seq, device)
            sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            psnr_val = _calc_psnr(sr, hr_img.to(device))
            axes[row, col].imshow(sr_np)
            axes[row, col].axis('off')
            axes[row, col].set_xlabel(f"PSNR: {psnr_val:.2f} dB", fontsize=8)

        axes[row, -1].imshow(hr_np)
        axes[row, -1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparação visual salva em '{output_path}'")
    plt.show()


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------
def evaluate_and_compare():
    try:
        ConfigManager.load_config('presets/config.json')
    except Exception:
        pass

    config = ConfigManager.get_instance()
    ckpt_dir = config.get('checkpoint_dir', './checkpoints')
    data_path = config.get('dataset_path', './datasets')
    scale = config.get('scale_factor', 2)
    crop_size = config.get('crop_size', 96)
    seq_len = config.get('seq_len', 3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Selecionar checkpoints
    ckpt_paths = _select_checkpoints(ckpt_dir)
    if not ckpt_paths:
        return

    # 2. Configurações da comparação
    max_eval = questionary.text(
        "Máximo de amostras para avaliação quantitativa:",
        default="200"
    ).ask()

    n_visual = questionary.text(
        "Quantidade de amostras para comparação visual:",
        default="4"
    ).ask()

    max_eval = int(max_eval)
    n_visual = int(n_visual)

    # 3. Carregar modelos
    print("\nCarregando modelos...")
    models_info = []
    for path in ckpt_paths:
        try:
            model, interface, ckpt_scale, label = _load_checkpoint(path, device)
            models_info.append({
                "model": model,
                "interface": interface,
                "scale": ckpt_scale,
                "label": label,
                "path": path,
            })
            print(f"  OK: {label}")
        except Exception as e:
            print(f"  ERRO ao carregar {os.path.basename(path)}: {e}")

    if not models_info:
        return

    # 4. Dataset de validação
    from train import VimeoSeptupletDataset, _resolve_vimeo_paths
    from torch.utils.data import DataLoader

    vimeo_seq_path, _, vimeo_test_list = _resolve_vimeo_paths(data_path)

    if not os.path.exists(vimeo_seq_path) or not os.path.exists(vimeo_test_list):
        print("Dataset Vimeo não encontrado. Verifique 'dataset_path' no config.json")
        print(f"  Procurado em: {vimeo_seq_path}")
        return

    val_ds = VimeoSeptupletDataset(
        vimeo_seq_path, vimeo_test_list,
        scale_factor=scale, crop_size=crop_size, seq_len=seq_len, train=False
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 5. Avaliação quantitativa
    print(f"\nAvaliando modelos (até {max_eval} amostras)...")
    results = _evaluate(models_info, val_loader, device, seq_len, max_eval)
    _print_table(results)

    # 6. Comparação visual
    print(f"\nGerando comparação visual ({n_visual} amostras)...")
    _visual_comparison(models_info, val_loader, device, seq_len, n_visual)
