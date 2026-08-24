"""
Plota os resultados da validação cruzada (k-fold) gerada por
kfold_validate.py, a partir dos CSVs:
    rtdvsr_kfold_results.csv    — 1 linha por fold (valor final)
    rtdvsr_kfold_summary.csv    — 1 linha por candidato (média ± desvio-padrão)
    rtdvsr_kfold_epoch_log.csv  — 1 linha por época de cada fold

Uso (a partir de training-python/, depois de já ter rodado kfold_validate.py):
    python scripts/plot_kfold_results.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_CSV = Path("rtdvsr_kfold_summary.csv")
EPOCH_LOG_CSV = Path("rtdvsr_kfold_epoch_log.csv")
OUTPUT_DIR = Path("assets/plots/kfold_validation")


def _read_summary(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["candidate_trial"] = int(r["candidate_trial"])
        r["mean_psnr"] = float(r["mean_psnr"])
        r["std_psnr"] = float(r["std_psnr"])
        r["mean_ssim"] = float(r["mean_ssim"]) if r["mean_ssim"] else float("nan")
        r["std_ssim"] = float(r["std_ssim"]) if r["std_ssim"] else 0.0
    return sorted(rows, key=lambda r: r["mean_psnr"], reverse=True)


def _read_epoch_log(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_candidate = {}
    for r in rows:
        trial = int(r["candidate_trial"])
        by_candidate.setdefault(trial, []).append({
            "fold": int(r["fold"]),
            "epoch": int(r["epoch"]),
            "val_psnr": float(r["val_psnr"]),
        })
    return by_candidate


def plot_candidate_comparison(summary_rows, output_dir):
    labels = [f"Trial {r['candidate_trial']}" for r in summary_rows]
    psnr_mean = [r["mean_psnr"] for r in summary_rows]
    psnr_std = [r["std_psnr"] for r in summary_rows]
    ssim_mean = [r["mean_ssim"] for r in summary_rows]
    ssim_std = [r["std_ssim"] for r in summary_rows]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Validação cruzada (k-fold) — comparação entre candidatos", fontsize=14)

    ax = axes[0]
    colors = ["tab:green"] + ["tab:blue"] * (len(labels) - 1)
    ax.bar(x, psnr_mean, yerr=psnr_std, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("PSNR médio ± desvio-padrão")
    ax.set_ylabel("PSNR (dB)")
    if psnr_mean:
        ax.set_ylim(min(psnr_mean) - max(psnr_std + [0.1]) - 0.5, max(psnr_mean) + max(psnr_std + [0.1]) + 0.5)

    ax = axes[1]
    ax.bar(x, ssim_mean, yerr=ssim_std, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("SSIM médio ± desvio-padrão")
    ax.set_ylabel("SSIM")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "candidate_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def plot_convergence(epoch_by_candidate, summary_rows, output_dir):
    order = [r["candidate_trial"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Convergência por candidato — média entre folds (sombra = desvio-padrão)", fontsize=12)

    cmap = plt.get_cmap("tab10")
    for i, trial in enumerate(order):
        rows = epoch_by_candidate.get(trial, [])
        if not rows:
            continue
        epochs = sorted({r["epoch"] for r in rows})
        mean_psnr, std_psnr = [], []
        for e in epochs:
            vals = [r["val_psnr"] for r in rows if r["epoch"] == e]
            mean_psnr.append(np.mean(vals))
            std_psnr.append(np.std(vals))
        mean_psnr = np.array(mean_psnr)
        std_psnr = np.array(std_psnr)
        color = cmap(i % 10)
        ax.plot(epochs, mean_psnr, label=f"Trial {trial}", color=color, marker="o", markersize=3)
        ax.fill_between(epochs, mean_psnr - std_psnr, mean_psnr + std_psnr, color=color, alpha=0.15)

    ax.set_xlabel("Época")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "convergence.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")


def main():
    if not SUMMARY_CSV.exists():
        print(f"{SUMMARY_CSV} não encontrado. Rode kfold_validate.py primeiro.")
        return

    summary_rows = _read_summary(SUMMARY_CSV)
    if not summary_rows:
        print(f"{SUMMARY_CSV} está vazio.")
        return

    print(f"Gerando gráficos de {len(summary_rows)} candidato(s)...\n")
    plot_candidate_comparison(summary_rows, OUTPUT_DIR)

    if EPOCH_LOG_CSV.exists():
        epoch_by_candidate = _read_epoch_log(EPOCH_LOG_CSV)
        plot_convergence(epoch_by_candidate, summary_rows, OUTPUT_DIR)
    else:
        print(f"  [pulado] convergence.png: {EPOCH_LOG_CSV} não encontrado")

    print(f"\nGráficos salvos em: {OUTPUT_DIR}/")
    best = summary_rows[0]
    print(f"Melhor candidato (k-fold): Trial {best['candidate_trial']} — "
          f"{best['mean_psnr']:.2f} ± {best['std_psnr']:.2f} dB")


if __name__ == "__main__":
    main()
