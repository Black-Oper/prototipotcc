"""
Plota a evolução do treino (loss, PSNR, SSIM, TCE por época) a partir dos
CSVs gerados por train.py em logs/<model_type>_training_log.csv.

Marca o melhor epoch e detecta a partir de qual época o PSNR estagnou
(sem ganho maior que --min-delta por --patience épocas seguidas) — útil
pra frase tipo "até a época X teve Y dB de melhora, depois estagnou".

Uso (a partir de training-python/):
    python scripts/plot_training_history.py                  # todos os logs em logs/
    python scripts/plot_training_history.py --model-type RTDVSR
    python scripts/plot_training_history.py --min-delta 0.05 --patience 5
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS_DIR = Path("logs")
OUTPUT_DIR = Path("assets/plots/training_history")


def _read_log(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_psnr": float(row["val_psnr"]),
                "val_ssim": float(row["val_ssim"]) if row["val_ssim"] else float("nan"),
                "val_tce": float(row["val_tce"]),
            })
    return rows


def _find_plateau_epoch(epochs, psnr, min_delta, patience):
    """Retorna a época a partir da qual o PSNR parou de melhorar de forma
    significativa (nenhum ganho >= min_delta por `patience` épocas seguidas)."""
    best = -math.inf
    best_epoch = epochs[0]
    stagnant = 0
    for e, p in zip(epochs, psnr):
        if p > best + min_delta:
            best = p
            best_epoch = e
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= patience:
            return best_epoch
    return None


def plot_model(model_type, rows, min_delta, patience, output_dir):
    epochs = [r["epoch"] for r in rows]
    loss = [r["train_loss"] for r in rows]
    psnr = [r["val_psnr"] for r in rows]
    ssim = [r["val_ssim"] for r in rows]
    tce = [r["val_tce"] for r in rows]

    best_epoch = epochs[psnr.index(max(psnr))]
    plateau_epoch = _find_plateau_epoch(epochs, psnr, min_delta, patience)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Histórico de treino — {model_type}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(epochs, loss, color="tab:red")
    ax.set_title("Train Loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")

    ax = axes[0, 1]
    ax.plot(epochs, psnr, color="tab:blue", marker="o", markersize=3)
    ax.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7,
              label=f"Melhor: época {best_epoch} ({max(psnr):.2f} dB)")
    if plateau_epoch is not None and plateau_epoch != epochs[-1]:
        ax.axvline(plateau_epoch, color="tab:orange", linestyle=":", alpha=0.8,
                  label=f"Estagnou a partir da época {plateau_epoch}")
    ax.set_title("Val PSNR")
    ax.set_xlabel("Época")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(epochs, ssim, color="tab:purple")
    ax.set_title("Val SSIM")
    ax.set_xlabel("Época")
    ax.set_ylabel("SSIM")

    ax = axes[1, 1]
    ax.plot(epochs, tce, color="tab:brown")
    ax.set_title("Val TCE (Temporal Consistency Error)")
    ax.set_xlabel("Época")
    ax.set_ylabel("TCE")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_type}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] {out_path} — melhor: época {best_epoch} ({max(psnr):.2f} dB)"
          + (f" | estagnou a partir da época {plateau_epoch}" if plateau_epoch else ""))


def main():
    parser = argparse.ArgumentParser(description="Plota o histórico de treino a partir dos logs de época")
    parser.add_argument("--model-type", default=None,
                        help="Nome do modelo (usa logs/<model_type>_training_log.csv). Sem isso, plota todos os logs encontrados.")
    parser.add_argument("--min-delta", type=float, default=0.05,
                        help="Ganho mínimo de PSNR (dB) pra não considerar estagnado (padrão: 0.05)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Épocas seguidas sem ganho mínimo pra declarar estagnação (padrão: 5)")
    args = parser.parse_args()

    if args.model_type:
        log_paths = [LOGS_DIR / f"{args.model_type}_training_log.csv"]
    else:
        log_paths = sorted(LOGS_DIR.glob("*_training_log.csv"))

    if not log_paths or not any(p.exists() for p in log_paths):
        print(f"Nenhum log encontrado em {LOGS_DIR}/. Treine um modelo primeiro (train.py já salva o log automaticamente).")
        return

    for path in log_paths:
        if not path.exists():
            print(f"  [pulado] {path} não encontrado")
            continue
        model_type = path.stem.replace("_training_log", "")
        rows = _read_log(path)
        if len(rows) < 2:
            print(f"  [pulado] {model_type}: só {len(rows)} época(s) registrada(s)")
            continue
        plot_model(model_type, rows, args.min_delta, args.patience, OUTPUT_DIR)

    print(f"\nGráficos salvos em: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
