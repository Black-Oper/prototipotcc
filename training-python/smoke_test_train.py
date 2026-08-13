"""
Smoke test: reproduz o caminho 'recurrent' do train.py para LightweightVSR,
PyramidVSR e MaskedRecurrentVSR. Verifica forward, backward, dtype/shape
do estado, mask loss e estabilidade numerica em FP32 e (se disponivel) BF16.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from models import get_model, get_interface
from train import CharbonnierLoss, _detach_state, _benchmark_inference


def test_model(name, params, device, amp_dtype, B=2, T=3, C=3, H=128, W=224, scale=2):
    print(f"\n{'='*70}\n[{name}] dtype={amp_dtype} device={device}\n{'='*70}")
    model = get_model(name, scale_factor=scale, channels=C, **params).to(device)
    interface = get_interface(name)
    print(f"  interface={interface}  params={sum(p.numel() for p in model.parameters()):,}")
    assert interface == "recurrent", f"esperado recurrent, got {interface}"

    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    lr_seq = torch.rand(B, T, C, H, W, device=device)
    hr_seq = torch.rand(B, T, C, H * scale, W * scale, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        recon_loss = torch.zeros(1, device=device, dtype=torch.float32)
        temp_loss = torch.zeros(1, device=device, dtype=torch.float32)
        mask_loss = torch.zeros(1, device=device, dtype=torch.float32)
        state = None
        prev_sr = None
        prev_hr = None
        for t in range(T):
            state = _detach_state(state)
            sr_frame, state = model(lr_seq[:, t], state)
            assert sr_frame.shape == (B, C, H * scale, W * scale), \
                f"shape SR errada: {sr_frame.shape}"
            recon_loss = recon_loss + criterion(sr_frame, hr_seq[:, t])
            if hasattr(model, 'get_mask_loss'):
                mask_loss = mask_loss + model.get_mask_loss()
            if prev_sr is not None:
                sr_diff = sr_frame - prev_sr
                hr_diff = hr_seq[:, t] - prev_hr
                temp_loss = temp_loss + torch.mean((sr_diff - hr_diff) ** 2)
            prev_sr = sr_frame.detach()
            prev_hr = hr_seq[:, t]

        recon_loss = recon_loss / T
        temp_loss = temp_loss / max(T - 1, 1)
        mask_loss = mask_loss / T
        total_loss = recon_loss + 0.1 * temp_loss + 0.05 * mask_loss

    assert torch.isfinite(total_loss).all(), f"loss nao finita: {total_loss}"
    print(f"  recon_loss={recon_loss.item():.4f}  temp_loss={temp_loss.item():.4f}  mask_loss={mask_loss.item():.4f}")
    print(f"  total_loss={total_loss.item():.4f}")

    if isinstance(state, tuple):
        state_info = " | ".join(f"shape={s.shape} dtype={s.dtype}" for s in state)
        print(f"  state (tuple of {len(state)} tensors): {state_info}")
    else:
        print(f"  state: shape={state.shape} dtype={state.dtype}")

    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    print(f"  grad_norm={grad_norm.item():.4f}")
    assert torch.isfinite(grad_norm), "grad_norm nao finito"

    # Verifica que gradientes nao sao todos zero
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_p = sum(1 for p in model.parameters())
    print(f"  parametros com grad>0: {has_grad}/{total_p}")
    assert has_grad > total_p * 0.5, "muitos parametros sem gradiente"

    optimizer.step()

    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    assert not has_nan, "NaN nos pesos apos step"
    print("  optimizer.step() OK, sem NaN")

    # Inference benchmark (mesmo path do train.py)
    ms = _benchmark_inference(model, device, scale, interface)
    print(f"  benchmark inferencia (540p->1080p): {ms:.1f} ms")
    print(f"[{name}] OK")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 suportado: {torch.cuda.is_bf16_supported()}")

    cfgs = [
        ("LightweightVSR", {"hidden_dim": 64, "num_res_blocks": 6}),
        ("PyramidVSR",      {"hidden_dim": 64, "num_res_blocks": 5}),
        ("MaskedRecurrentVSR", {"hidden_dim": 64, "num_res_blocks": 5}),
    ]

    # FP32 sempre
    for name, params in cfgs:
        test_model(name, params, device, torch.float32)

    # BF16 se CUDA + suporte
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        for name, params in cfgs:
            test_model(name, params, device, torch.bfloat16)


if __name__ == '__main__':
    main()
