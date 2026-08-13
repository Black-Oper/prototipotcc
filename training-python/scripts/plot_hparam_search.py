"""
Gera gráficos do estudo Optuna salvo por hparam_search.py, em PNG de alta
resolução prontos pra apresentação/TCC.

Uso (a partir de training-python/, depois de já ter rodado hparam_search.py):
    python scripts/plot_hparam_search.py

Para acompanhar a busca ao vivo enquanto ela roda (dashboard web, sem
precisar deste script), instale `optuna-dashboard` e rode em outro
terminal:
    pip install optuna-dashboard
    optuna-dashboard sqlite:///rtdvsr_hparam_search.db
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_intermediate_values,
    plot_parallel_coordinate,
    plot_slice,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hparam_search import STUDY_NAME, STORAGE, _print_top

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

OUTPUT_DIR = Path("assets/plots/hparam_search")

PLOTS = {
    "optimization_history": (plot_optimization_history, "PSNR de cada trial ao longo da busca, com o melhor valor até então"),
    "param_importances":    (plot_param_importances, "Quais hiperparâmetros mais influenciaram o PSNR"),
    "intermediate_values":  (plot_intermediate_values, "Curva de PSNR por época de cada trial — mostra os trials podados"),
    "parallel_coordinate":  (plot_parallel_coordinate, "Relação entre combinações de hiperparâmetros e o PSNR resultante"),
    "slice":                (plot_slice, "PSNR em função de cada hiperparâmetro individualmente"),
}


def _save_plot(ax, path):
    fig = np.atleast_1d(ax).flatten()[0].get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except KeyError:
        print(f"Nenhum estudo encontrado em {STORAGE}. Rode hparam_search.py primeiro.")
        return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        print(f"Só {len(completed)} trial(s) completo(s) — rode mais trials antes de plotar "
              "(alguns gráficos precisam de variação entre trials).")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Gerando gráficos de {len(study.trials)} trials ({len(completed)} completos)...\n")

    for name, (plot_fn, desc) in PLOTS.items():
        out_path = OUTPUT_DIR / f"{name}.png"
        try:
            ax = plot_fn(study)
            _save_plot(ax, out_path)
            print(f"  [OK] {out_path.name} — {desc}")
        except Exception as e:
            print(f"  [pulado] {name}: {e}")

    print(f"\nGráficos salvos em: {OUTPUT_DIR}/")
    _print_top(study)


if __name__ == "__main__":
    main()
