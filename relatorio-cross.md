# Validação Cruzada (k-fold) dos Hiperparâmetros — RTDVSR

Registro do que foi implementado nesta etapa. **A validação ainda não foi
executada** — o código está pronto para o Gustavo rodar na máquina dele.

## Contexto

- Depois da busca de hiperparâmetros (Optuna, 54 trials — ver
  `rtdvsr_hparam_search_results.csv` e o relatório da reunião anterior), o
  orientador pediu validação cruzada (k-fold) antes de comprometer o treino
  final.
- Decisão: k-fold só na etapa de validação dos hiperparâmetros (barata — 5
  épocas por fold), não no treino final. O treino final continua rodando
  uma única vez (até estagnar), com a configuração que sair mais robusta
  daqui — repetir o treino final k vezes seria caro demais, já que ele não
  tem mais número fixo de épocas.

## O que é k-fold (resumo, contexto completo já discutido na reunião)

Em vez de validar cada configuração de hiperparâmetros contra um único
split treino/validação, o dataset é dividido em *k* partes; o modelo treina
*k* vezes, cada vez com uma parte diferente como validação. A média (±
desvio-padrão) das *k* métricas é uma estimativa mais confiável de
generalização do que um único número.

## Parâmetros adotados

| Parâmetro | Valor | Justificativa |
|---|---|---|
| k (nº de folds) | 5 | Ver citação abaixo |
| Candidatos validados | Top 5 (trials 44, 24, 25, 34, 39) | Mesmos do relatório da busca de hiperparâmetros |
| Épocas por fold | 5 | Um pouco acima do orçamento da busca (3), pra sinal mais confiável, mantendo custo controlado |
| Batches por época | 500 | Mesmo orçamento de `hparam_search.py` |
| Custo total | 5 × 5 = 25 treinos completos | Sem poda — aqui queremos o número final de cada candidato, não descartar cedo |

**Citação para a monografia** (k=5 como escolha padrão, trade-off
viés-variância):

> James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An
> Introduction to Statistical Learning: with Applications in R*. Springer,
> p. 184 (seção 5.1.4) — "k-fold cross-validation com k=5 ou k=10 tem se
> mostrado empiricamente livre de viés excessivo e de variância excessiva".

Citação complementar (origem do uso consagrado de 5/10 folds na literatura
de model selection):

> Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for
> Accuracy Estimation and Model Selection*. Proceedings of IJCAI, 14(2),
> 1137–1145.

## O que foi implementado

- **`training-python/scripts/hparam_search.py`** — refatorado: a lógica de
  treino/validação que antes vivia dentro de `objective()` (usado só pelo
  Optuna) virou uma função independente, `_train_config()`, reaproveitada
  também pelo k-fold. Sem mudança de comportamento na busca já rodada.
- **`training-python/scripts/kfold_validate.py`** (novo) — lê os top-N
  candidatos de `rtdvsr_hparam_search_results.csv`, monta os k folds a
  partir de `sep_trainlist.txt` + `sep_testlist.txt` combinados (split por
  sequência/clipe inteiro, sem vazamento de frame), treina cada candidato em
  cada fold e grava:
  - `rtdvsr_kfold_epoch_log.csv` — 1 linha por época de cada fold (para
    curva de convergência).
  - `rtdvsr_kfold_results.csv` — 1 linha por fold (valor final de PSNR/SSIM).
  - `rtdvsr_kfold_summary.csv` — 1 linha por candidato, com média ±
    desvio-padrão entre os folds, ranqueado.

  Grava tudo incrementalmente (não só no final) e retoma sozinho pulando
  pares (candidato, fold) já concluídos se for interrompido.
- **`training-python/scripts/plot_kfold_results.py`** (novo) — lê os CSVs
  acima (sem precisar rodar nada de novo) e gera em
  `assets/plots/kfold_validation/`:
  - `candidate_comparison.png` — barras de PSNR/SSIM médio ± desvio-padrão
    por candidato.
  - `convergence.png` — curva de PSNR por época, média entre os 5 folds
    (sombra = desvio-padrão), uma linha por candidato.
- **`.gitignore`** — ignora `training-python/scripts/.kfold_lists/` (listas
  de sequências por fold, arquivos temporários regeneráveis).

## Passo a passo para o Gustavo rodar

A partir de `training-python/`, com o ambiente já configurado (mesmo
`requirements.txt`, dataset Vimeo Septuplet baixado):

```bash
python scripts/kfold_validate.py
python scripts/plot_kfold_results.py
```

Por padrão roda com os parâmetros da tabela acima (`--k 5 --top-n 5
--epochs 5`). Se precisar interromper (Ctrl+C), é só rodar o mesmo comando
de novo depois — ele retoma sozinho.

## Próximo passo

Depois de rodar: comparar o ranking do k-fold com o ranking da busca
original (trial 44 era o vencedor com PSNR único de 36,82 dB) — se o
vencedor se mantiver, segue pro treino final com essa config; se outro
candidato generalizar melhor (média mais alta ou desvio-padrão menor entre
folds), discutir a troca antes do treino final.
