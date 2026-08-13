"""
Dataset Vid4 (calendar, city, foliage, walk) — formato padrão de benchmark VSR.

Estrutura esperada (compatível com o zip que vem do Kaggle/Drive):

    <root>/
        GT/<seq>/00000000.png ...    # ground truth (HR)
        BIx4/<seq>/00000000.png ...  # bicúbico x4 (LR)
        BDx4/<seq>/00000000.png ...  # blur-down x4 (LR)

Modo train=True  : retorna janelas curtas (seq_len) com crop aleatório.
                   Bom para fine-tuning/augmentação.
Modo train=False : retorna a sequência INTEIRA sem crop, para reproduzir os
                   números oficiais de PSNR/SSIM/TCE no Vid4.
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


VID4_SEQUENCES = ("calendar", "city", "foliage", "walk")


def is_vid4_root(path: str | os.PathLike) -> bool:
    """Detecta se o diretório segue o layout Vid4 (GT + BIx4 ou BDx4)."""
    p = Path(path)
    if not p.is_dir():
        return False
    has_gt = (p / "GT").is_dir()
    has_lr = (p / "BIx4").is_dir() or (p / "BDx4").is_dir()
    return has_gt and has_lr


class Vid4Dataset(Dataset):
    """
    Dataset Vid4 com pares (LR, HR).

    Args:
        root: pasta que contém GT/, BIx4/, BDx4/.
        scale_factor: deve ser 4 (Vid4 só tem x4). Outros valores geram aviso.
        seq_len: nº de frames consecutivos por amostra (modo treino).
        crop_size: tamanho do crop HR no treino. None = sem crop.
        train: True = janelas + augmentação; False = sequência inteira.
        degradation: 'BI' (bicúbico) ou 'BD' (blur-down).
        sequences: lista opcional de sequências a usar (default: todas).
        repeat: nº de janelas amostradas por sequência por época (treino).
    """

    def __init__(
        self,
        root: str,
        scale_factor: int = 4,
        seq_len: int = 3,
        crop_size: int | None = 256,
        train: bool = True,
        degradation: str = "BI",
        sequences: list[str] | None = None,
        repeat: int = 100,
        hr_only: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.scale_factor = scale_factor
        self.seq_len = max(1, seq_len)
        self.crop_size = crop_size
        self.train = train
        self.repeat = max(1, repeat)
        self.hr_only = hr_only

        deg = degradation.upper()
        if deg not in ("BI", "BD"):
            raise ValueError(f"degradation deve ser 'BI' ou 'BD', recebi {degradation}")
        self.lr_subdir = f"{deg}x{scale_factor}"

        gt_dir = self.root / "GT"
        lr_dir = self.root / self.lr_subdir
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"GT não encontrado: {gt_dir}")

        if not hr_only:
            if not lr_dir.is_dir():
                raise FileNotFoundError(
                    f"LR não encontrado: {lr_dir}. "
                    f"Verifique o fator de escala (atual: x{scale_factor}) e a degradation."
                )
            if scale_factor != 4:
                warnings.warn(
                    f"Vid4 oficial é x4 mas scale_factor={scale_factor}. "
                    "Não há LR correspondente no archive; pode haver mismatch."
                )

        # Indexa todas as sequências disponíveis e seus frames
        avail = sequences or VID4_SEQUENCES
        self.sequences: list[dict] = []
        for name in avail:
            sg = gt_dir / name
            sl = lr_dir / name
            if not sg.is_dir():
                warnings.warn(f"Sequência GT ausente, pulando: {name}")
                continue
            if not hr_only and not sl.is_dir():
                warnings.warn(f"Sequência LR ausente, pulando: {name}")
                continue
            gt_frames = sorted(sg.glob("*.png"))
            if hr_only:
                lr_frames = gt_frames  # placeholder, não será usado
                n = len(gt_frames)
            else:
                lr_frames = sorted(sl.glob("*.png"))
                if len(gt_frames) != len(lr_frames):
                    warnings.warn(
                        f"{name}: GT ({len(gt_frames)}) != LR ({len(lr_frames)})"
                    )
                n = min(len(gt_frames), len(lr_frames))
            if n < self.seq_len:
                warnings.warn(f"{name} tem {n} frames < seq_len={self.seq_len}, pulando.")
                continue
            self.sequences.append({
                "name": name,
                "gt": gt_frames[:n],
                "lr": lr_frames[:n],
                "n": n,
            })

        if not self.sequences:
            raise RuntimeError(f"Nenhuma sequência válida encontrada em {self.root}")

    # ------------------------------------------------------------------ size
    def __len__(self) -> int:
        if self.train:
            # `repeat` janelas por sequência por época, para que o DataLoader
            # tenha um epoch razoavelmente grande mesmo com Vid4 minúsculo.
            return len(self.sequences) * self.repeat
        return len(self.sequences)  # 1 amostra = sequência inteira

    # ------------------------------------------------------------------ get
    def __getitem__(self, idx):
        try:
            if self.train:
                seq = self.sequences[idx % len(self.sequences)]
                return self._get_train_window(seq)
            seq = self.sequences[idx]
            return self._get_full_sequence(seq)
        except Exception as e:
            warnings.warn(f"Erro lendo idx={idx}: {e}")
            # tenta outra amostra para não quebrar o loader
            return self.__getitem__((idx + 1) % len(self))

    # --------------------------------------------------------------- helpers
    def _get_train_window(self, seq: dict):
        start = random.randint(0, seq["n"] - self.seq_len)
        frame_ids = range(start, start + self.seq_len)

        hr_imgs = [Image.open(seq["gt"][i]).convert("RGB") for i in frame_ids]
        lr_imgs = [Image.open(seq["lr"][i]).convert("RGB") for i in frame_ids]

        # Garante tamanho LR == HR/scale (pequenos arredondamentos em alguns zips)
        hr_w, hr_h = hr_imgs[0].size
        hr_w = (hr_w // self.scale_factor) * self.scale_factor
        hr_h = (hr_h // self.scale_factor) * self.scale_factor
        lr_w, lr_h = hr_w // self.scale_factor, hr_h // self.scale_factor
        hr_imgs = [im.resize((hr_w, hr_h), Image.BICUBIC) if im.size != (hr_w, hr_h)
                   else im for im in hr_imgs]
        lr_imgs = [im.resize((lr_w, lr_h), Image.BICUBIC) if im.size != (lr_w, lr_h)
                   else im for im in lr_imgs]

        # Crop espacial sincronizado (HR e LR coerentes)
        if self.crop_size is not None and self.crop_size < hr_h and self.crop_size < hr_w:
            chs = self.crop_size
            cls = chs // self.scale_factor
            top_lr = random.randint(0, lr_h - cls)
            left_lr = random.randint(0, lr_w - cls)
            top_hr = top_lr * self.scale_factor
            left_hr = left_lr * self.scale_factor
            hr_imgs = [im.crop((left_hr, top_hr, left_hr + chs, top_hr + chs))
                       for im in hr_imgs]
            lr_imgs = [im.crop((left_lr, top_lr, left_lr + cls, top_lr + cls))
                       for im in lr_imgs]

        # Augmentação geométrica (flip H/V — sem rotação 90 p/ não exigir quadrado)
        if random.random() < 0.5:
            hr_imgs = [TF.hflip(im) for im in hr_imgs]
            lr_imgs = [TF.hflip(im) for im in lr_imgs]
        if random.random() < 0.5:
            hr_imgs = [TF.vflip(im) for im in hr_imgs]
            lr_imgs = [TF.vflip(im) for im in lr_imgs]

        lr_t = torch.stack([TF.to_tensor(im) for im in lr_imgs])
        hr_t = torch.stack([TF.to_tensor(im) for im in hr_imgs])
        return lr_t, hr_t

    def _get_full_sequence(self, seq: dict):
        hr_imgs = [Image.open(p).convert("RGB") for p in seq["gt"]]
        hr_w, hr_h = hr_imgs[0].size

        if self.hr_only:
            # Não exige LR: retorna HR sem garantir divisibilidade por scale.
            hr_t = torch.stack([TF.to_tensor(im) for im in hr_imgs])
            return hr_t, hr_t  # primeiro tensor é placeholder, consumidor ignora

        lr_imgs = [Image.open(p).convert("RGB") for p in seq["lr"]]

        hr_w = (hr_w // self.scale_factor) * self.scale_factor
        hr_h = (hr_h // self.scale_factor) * self.scale_factor
        lr_w, lr_h = hr_w // self.scale_factor, hr_h // self.scale_factor
        hr_imgs = [im.resize((hr_w, hr_h), Image.BICUBIC) if im.size != (hr_w, hr_h)
                   else im for im in hr_imgs]
        lr_imgs = [im.resize((lr_w, lr_h), Image.BICUBIC) if im.size != (lr_w, lr_h)
                   else im for im in lr_imgs]

        lr_t = torch.stack([TF.to_tensor(im) for im in lr_imgs])
        hr_t = torch.stack([TF.to_tensor(im) for im in hr_imgs])
        return lr_t, hr_t

    # ------------------------------------------------------------------ info
    def describe(self) -> str:
        mode = "HR-only (LR gerado on-the-fly)" if self.hr_only else f"deg={self.lr_subdir}"
        lines = [f"Vid4 root={self.root} {mode} train={self.train}"]
        for s in self.sequences:
            lines.append(f"  - {s['name']}: {s['n']} frames")
        return "\n".join(lines)
