"""Inspeciona os checkpoints existentes e simula a logica de resume."""
import os
import torch
from models import get_model

CKPT_DIR = "checkpoints"
for fname in sorted(os.listdir(CKPT_DIR)):
    if not fname.endswith(".pth"):
        continue
    path = os.path.join(CKPT_DIR, fname)
    print(f"\n{'='*70}\n{fname}\n{'='*70}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  keys: {sorted(ckpt.keys())}")
    print(f"  epoch: {ckpt.get('epoch')}")
    print(f"  psnr:  {ckpt.get('psnr')}")
    print(f"  tce:   {ckpt.get('tce')}")
    print(f"  inference_ms: {ckpt.get('inference_ms')}")
    print(f"  model_type:   {ckpt.get('model_type')}")
    print(f"  model_params: {ckpt.get('model_params')}")
    cfg = ckpt.get('config', {})
    if cfg:
        print(f"  config.model_type:   {cfg.get('model_type')}")
        print(f"  config.model_params: {cfg.get('model_params')}")
        print(f"  config.scale_factor: {cfg.get('scale_factor')}  seq_len: {cfg.get('seq_len')}")
    scaler = ckpt.get('scaler')
    print(f"  scaler: {None if scaler is None else type(scaler).__name__}")
    sd = ckpt['model']
    print(f"  model.state_dict tem {len(sd)} tensores, total params: {sum(v.numel() for v in sd.values()):,}")

    # Tenta instanciar e carregar
    mtype = ckpt.get('model_type') or fname.split('_best_')[0]
    mparams = ckpt.get('model_params') or (cfg.get('model_params') if cfg else {}) or {}
    scale = (cfg.get('scale_factor') if cfg else 2) or 2
    try:
        model = get_model(mtype, scale_factor=scale, channels=3, **mparams)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(f"  load_state_dict(strict=True) OK")
    except Exception as e:
        print(f"  ERRO load_state_dict: {e}")

    # Verifica optimizer state
    opt_state = ckpt.get('optimizer', {})
    if opt_state:
        nparam_groups = len(opt_state.get('param_groups', []))
        nstate = len(opt_state.get('state', {}))
        print(f"  optimizer: {nparam_groups} param_groups, {nstate} param states")
