"""
Gera um dataset sintético (imagens 448x256, dimensão nativa do Vimeo
Septuplet) para benchmark de throughput de treino, sem precisar baixar o
dataset real (~88 GB). Usado por bench_full_epoch.py.

Uso (a partir de training-python/):
    python scripts/bench_gen_synthetic.py [--n-sequences 600]
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

VIMEO_W, VIMEO_H = 448, 256

OUT_ROOT = Path("datasets/vimeo_septuplet_bench")
SEQ_DIR = OUT_ROOT / "sequences"
TRAINLIST = OUT_ROOT / "sep_trainlist.txt"


def _make_frame(rng, base_color):
    # Gradiente + leve ruído: rápido de codificar em PNG (baixa entropia),
    # mas ainda uma imagem RGB "real" nas dimensões corretas.
    x = np.linspace(0, 1, VIMEO_W, dtype=np.float32)
    y = np.linspace(0, 1, VIMEO_H, dtype=np.float32)
    grad = np.outer(y, x)
    arr = np.zeros((VIMEO_H, VIMEO_W, 3), dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = base_color[c] + grad * 40 - 20
    arr += rng.normal(0, 5, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sequences", type=int, default=600)
    args = parser.parse_args()

    if SEQ_DIR.exists():
        shutil.rmtree(OUT_ROOT)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    names = []
    for i in range(args.n_sequences):
        seq_name = f"seq{i:05d}"
        seq_path = SEQ_DIR / seq_name
        seq_path.mkdir(parents=True, exist_ok=True)
        base_color = rng.integers(40, 200, size=3).astype(np.float32)
        for fi in range(1, 8):
            img = _make_frame(rng, base_color)
            img.save(seq_path / f"im{fi}.png", compress_level=1)
        names.append(seq_name)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_sequences} sequências geradas...")

    with open(TRAINLIST, "w") as f:
        f.write("\n".join(names))

    print(f"\nDataset sintético gerado em: {OUT_ROOT}")
    print(f"{len(names)} sequências, {len(names) * 7} imagens ({VIMEO_W}x{VIMEO_H}).")


if __name__ == "__main__":
    main()
