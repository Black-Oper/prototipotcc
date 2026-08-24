"""
Validação Cruzada (k-fold) dos Hiperparâmetros — RTDVSR
=========================================================
Segundo passo depois da busca de hiperparâmetros (hparam_search.py): pega os
top-N candidatos por PSNR de validação (rtdvsr_hparam_search_results.csv) e
treina cada um em k folds diferentes do dataset, para confirmar que a
configuração escolhida generaliza — e não só se ajustou às particularidades
de um único split treino/validação.

k=5 (padrão), como recomendado pela literatura (viés/variância equilibrados):
James, Witten, Hastie & Tibshirani, "An Introduction to Statistical
Learning" (2013), p. 184; Kohavi (1995), "A Study of Cross-Validation and
Bootstrap for Accuracy Estimation and Model Selection", IJCAI'95.

Cada fold usa o mesmo orçamento de treino do hparam_search.py
(max_train_batches por época), sem poda — aqui queremos o número final de
cada candidato em cada fold, não descartar cedo.

Uso (a partir de training-python/):
    python scripts/kfold_validate.py
    python scripts/kfold_validate.py --k 5 --top-n 5 --epochs 5

Métricas são gravadas incrementalmente em:
    rtdvsr_kfold_epoch_log.csv   — 1 linha por época de cada fold (convergência)
    rtdvsr_kfold_results.csv     — 1 linha por fold (valor final)
    rtdvsr_kfold_summary.csv     — 1 linha por candidato (média ± desvio-padrão)

Interromper (Ctrl+C) e rodar de novo retoma pulando os pares
(candidato, fold) já presentes em rtdvsr_kfold_results.csv.

Depois de rodar, gere os gráficos com:
    python scripts/plot_kfold_results.py
"""

import argparse
import csv
import math
import os
import statistics
import sys
from pathlib import Path

import torch
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hparam_search import FIXED, _train_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import VimeoSeptupletDataset, _resolve_vimeo_paths

RESULTS_CSV = "rtdvsr_hparam_search_results.csv"
EPOCH_LOG_CSV = "rtdvsr_kfold_epoch_log.csv"
FOLD_RESULTS_CSV = "rtdvsr_kfold_results.csv"
SUMMARY_CSV = "rtdvsr_kfold_summary.csv"

FOLD_LISTS_DIR = Path(__file__).resolve().parent / ".kfold_lists"

PARAM_KEYS = ["hidden_dim", "learning_rate", "num_res_blocks", "temporal_loss_weight"]


def _load_candidates(results_csv, top_n):
    """Lê os trials completos de hparam_search.py, ordena por val_psnr
    e retorna os top-N como lista de dicts {trial, hidden_dim, learning_rate,
    num_res_blocks, temporal_loss_weight}."""
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    completed = [r for r in rows if r["state"] == "COMPLETE" and r["val_psnr"]]
    completed.sort(key=lambda r: float(r["val_psnr"]), reverse=True)

    candidates = []
    for r in completed[:top_n]:
        candidates.append({
            "trial": int(r["trial"]),
            "hidden_dim": int(r["hidden_dim"]),
            "learning_rate": float(r["learning_rate"]),
            "num_res_blocks": int(r["num_res_blocks"]),
            "temporal_loss_weight": float(r["temporal_loss_weight"]),
        })
    return candidates


def _build_folds(data_path, k, seed):
    """Combina sep_trainlist.txt + sep_testlist.txt (cada linha já é uma
    sequência/clipe inteiro) e devolve os caminhos das sequências do Vimeo
    mais os k splits (train_idx, val_idx) de KFold."""
    vimeo_seq, vimeo_train, vimeo_test = _resolve_vimeo_paths(data_path)
    if not os.path.exists(vimeo_seq) or not os.path.exists(vimeo_train):
        raise FileNotFoundError(
            f"Vimeo Septuplet não encontrado em: {vimeo_seq}\n"
            "Execute 'Baixar dataset' no menu do app.py primeiro.")

    with open(vimeo_train) as f:
        pool = [line.strip() for line in f if line.strip()]
    if os.path.exists(vimeo_test):
        with open(vimeo_test) as f:
            pool += [line.strip() for line in f if line.strip()]

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(kf.split(pool))
    return vimeo_seq, pool, splits


def _write_fold_lists(pool, splits, fold_i):
    """Escreve os arquivos de treino/validação do fold `fold_i` em
    FOLD_LISTS_DIR e retorna seus caminhos."""
    FOLD_LISTS_DIR.mkdir(exist_ok=True)
    train_idx, val_idx = splits[fold_i]
    train_path = FOLD_LISTS_DIR / f"fold{fold_i}_train.txt"
    val_path = FOLD_LISTS_DIR / f"fold{fold_i}_val.txt"
    with open(train_path, "w") as f:
        f.write("\n".join(pool[i] for i in train_idx))
    with open(val_path, "w") as f:
        f.write("\n".join(pool[i] for i in val_idx))
    return str(train_path), str(val_path)


def _load_done_pairs(path):
    """Lê rtdvsr_kfold_results.csv (se existir) e devolve o conjunto de
    pares (candidate_trial, fold) já concluídos, para retomar."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(int(r["candidate_trial"]), int(r["fold"])) for r in rows}


def _append_csv_row(path, fieldnames, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _write_summary(fold_results_path, summary_path):
    with open(fold_results_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_candidate = {}
    for r in rows:
        by_candidate.setdefault(int(r["candidate_trial"]), []).append(r)

    summary_rows = []
    for trial, trial_rows in by_candidate.items():
        psnrs = [float(r["val_psnr"]) for r in trial_rows]
        ssims = [float(r["val_ssim"]) for r in trial_rows if r["val_ssim"]]
        row = {
            "candidate_trial": trial,
            "n_folds": len(trial_rows),
            "mean_psnr": round(statistics.mean(psnrs), 4),
            "std_psnr": round(statistics.pstdev(psnrs), 4) if len(psnrs) > 1 else 0.0,
            "mean_ssim": round(statistics.mean(ssims), 4) if ssims else "",
            "std_ssim": round(statistics.pstdev(ssims), 4) if len(ssims) > 1 else 0.0,
        }
        for key in PARAM_KEYS:
            row[key] = trial_rows[0][key]
        summary_rows.append(row)

    summary_rows.sort(key=lambda r: r["mean_psnr"], reverse=True)

    fieldnames = ["candidate_trial", "n_folds", "mean_psnr", "std_psnr",
                  "mean_ssim", "std_ssim"] + PARAM_KEYS
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def _print_summary(summary_rows):
    print("\n" + "=" * 70)
    print("RESUMO — média ± desvio-padrão de PSNR por candidato (k-fold):")
    print("=" * 70)
    for rank, row in enumerate(summary_rows, 1):
        ssim_str = f"{row['mean_ssim']:.4f}" if row["mean_ssim"] != "" else "N/A"
        print(f"\n#{rank} — Trial {row['candidate_trial']} — "
              f"PSNR: {row['mean_psnr']:.2f} ± {row['std_psnr']:.2f} dB | "
              f"SSIM: {ssim_str}")
        for key in PARAM_KEYS:
            print(f"    {key}: {row[key]}")
    print(f"\nResultados completos em: {FOLD_RESULTS_CSV}")
    print(f"Resumo por candidato em: {SUMMARY_CSV}")
    print(f"Histórico por época em: {EPOCH_LOG_CSV}")
    print("Para gerar os gráficos: python scripts/plot_kfold_results.py")


def main():
    parser = argparse.ArgumentParser(
        description="Validação cruzada (k-fold) dos hiperparâmetros do RTDVSR")
    parser.add_argument("--k", type=int, default=5,
                        help="Número de folds (padrão: 5)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Quantos candidatos (trials) do results.csv validar (padrão: 5)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Épocas por fold (padrão: 5)")
    parser.add_argument("--results-csv", type=str, default=RESULTS_CSV,
                        help=f"CSV de resultados do hparam_search.py (padrão: {RESULTS_CSV})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    candidates = _load_candidates(args.results_csv, args.top_n)
    if not candidates:
        print(f"Nenhum trial completo encontrado em {args.results_csv}.")
        return
    print(f"{len(candidates)} candidato(s) carregado(s) de {args.results_csv}: "
          f"{[c['trial'] for c in candidates]}")

    vimeo_seq, pool, splits = _build_folds(FIXED["dataset_path"], args.k, FIXED["seed"])
    print(f"Pool combinado (trainlist + testlist): {len(pool)} sequências, "
          f"{args.k} folds.")

    done_pairs = _load_done_pairs(FOLD_RESULTS_CSV)
    if done_pairs:
        print(f"Retomando: {len(done_pairs)} par(es) (candidato, fold) já concluído(s).")

    fold_result_fields = ["candidate_trial", "fold", "val_psnr", "val_ssim"] + PARAM_KEYS
    epoch_log_fields = ["candidate_trial", "fold", "epoch", "val_psnr", "val_ssim"]

    total_runs = len(candidates) * args.k
    run_i = 0
    for cand in candidates:
        for fold_i in range(args.k):
            run_i += 1
            if (cand["trial"], fold_i) in done_pairs:
                print(f"[{run_i}/{total_runs}] Trial {cand['trial']} / fold {fold_i} "
                      "— já concluído, pulando.")
                continue

            print(f"\n[{run_i}/{total_runs}] Trial {cand['trial']} / fold {fold_i} "
                  f"— hidden_dim={cand['hidden_dim']}, "
                  f"lr={cand['learning_rate']:.2e}, "
                  f"num_res_blocks={cand['num_res_blocks']}, "
                  f"temporal_weight={cand['temporal_loss_weight']:.3f}")

            train_list, val_list = _write_fold_lists(pool, splits, fold_i)
            train_ds = VimeoSeptupletDataset(
                vimeo_seq, train_list, scale_factor=FIXED["scale_factor"],
                seq_len=FIXED["seq_len"], train=True)
            val_ds = VimeoSeptupletDataset(
                vimeo_seq, val_list, scale_factor=FIXED["scale_factor"],
                seq_len=FIXED["seq_len"], train=False)

            def _on_epoch_end(epoch, val_psnr, val_ssim, cand=cand, fold_i=fold_i):
                _append_csv_row(EPOCH_LOG_CSV, epoch_log_fields, {
                    "candidate_trial": cand["trial"],
                    "fold": fold_i,
                    "epoch": epoch,
                    "val_psnr": round(val_psnr, 4),
                    "val_ssim": round(val_ssim, 4) if not math.isnan(val_ssim) else "",
                })

            val_psnr, val_ssim = _train_config(
                cand["hidden_dim"], cand["num_res_blocks"], cand["learning_rate"],
                cand["temporal_loss_weight"], train_ds, val_ds, device, args.epochs,
                trial=None, on_epoch_end=_on_epoch_end)

            row = {
                "candidate_trial": cand["trial"],
                "fold": fold_i,
                "val_psnr": round(val_psnr, 4),
                "val_ssim": round(val_ssim, 4) if not math.isnan(val_ssim) else "",
            }
            for key in PARAM_KEYS:
                row[key] = cand[key]
            _append_csv_row(FOLD_RESULTS_CSV, fold_result_fields, row)

    summary_rows = _write_summary(FOLD_RESULTS_CSV, SUMMARY_CSV)
    _print_summary(summary_rows)


if __name__ == "__main__":
    main()
