# Super-Resolução de Vídeo em Tempo Real

O projeto tem dois módulos independentes:

- **`training-python/`** — treino, avaliação e inferência em tempo real via
  PyTorch. É o módulo documentado neste README.
- **`inference-cpp/`** — motor de inferência nativo em C++/TensorRT (WIP),
  consumindo o `.onnx` exportado por `training-python/scripts/export_trt.py`.
  Tem build própria via CMake; veja `inference-cpp/CMakeLists.txt`.

Este documento tem duas partes: **Visão Geral** (instalação e uso básico —
o suficiente pra rodar o projeto) e **Funcionalidades em Detalhe**
(como cada parte funciona por dentro).

---

# Visão Geral

## 1. Requisitos

- Windows 10/11, Linux ou macOS
- Python 3.9+
- (Recomendado) GPU NVIDIA com driver compatível com CUDA 12.x

## 2. Instalação

```bash
git clone <seu-repo> prototipotcc
cd prototipotcc/training-python

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

**PyTorch com CUDA (importante).** O `requirements.txt` lista `torch` sem
índice específico. Isso faz o pip resolver o **wheel CPU-only** por padrão
no Windows, e o menu "Testar CUDA" vai reportar "CUDA disponível: Não".
Para habilitar a GPU, instale o build CUDA **antes** do resto das
dependências:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Ajuste `cu121` para a versão do CUDA Toolkit suportada pelo seu driver
NVIDIA (veja https://pytorch.org/get-started/locally/).

**Demais dependências:**

```bash
pip install -r requirements.txt
```

**Verificar a GPU:**

```bash
python app.py
# no menu, selecione "Testar CUDA"
```

A saída deve mostrar `CUDA disponível: Sim` e o nome da sua GPU.

## 3. Dataset

O treino usa o **Vimeo Septuplet** (sequências de 7 frames). No menu, escolha
"Baixar dataset" — o arquivo é baixado, extraído e, se vier com a pasta
aninhada padrão (`vimeo_septuplet/vimeo_septuplet/...`), ela é achatada
automaticamente.

Layout esperado ao final:

```
datasets/
└── vimeo_septuplet/
    ├── sequences/
    │   ├── 00001/0001/im1.png ... im7.png
    │   └── ...
    ├── sep_trainlist.txt
    └── sep_testlist.txt
```

Se a estrutura estiver incorreta, o `train.py` loga os caminhos que procurou
e cai em um fallback SISR **sem** coerência temporal — fique atento à
mensagem no console.

Opcionalmente, o **Vid4** (benchmark clássico de VSR: calendar/city/foliage/walk)
pode ser usado como dataset de avaliação em "Comparar modelos" — veja
[§2 de Funcionalidades](#2-comparação-de-modelos-comparepy).

## 4. Como rodar

Todas as ações passam pelo menu principal:

```bash
python app.py
```

| Opção | O que faz | Detalhes |
|-------|-----------|----------|
| Baixar dataset | Baixa e extrai o Vimeo Septuplet em `./datasets` | [§3](#3-dataset) |
| Treinar modelo | Treina o modelo definido em `presets/config.json` | [Funcionalidades §1](#1-treino) |
| Testar treinamento (3 épocas) | Roda o treino real por só 3 épocas, com o dataset de verdade | [Funcionalidades §1](#1-treino) |
| Smoke test (forward/backward rápido) | Testa forward/backward de 3 arquiteturas com tensores sintéticos (sem dataset) | [Funcionalidades §6](#6-ferramentas-de-desenvolvimento-e-validação-scripts) |
| Comparar modelos | Avalia PSNR/SSIM/TCE e gera comparação visual | [Funcionalidades §2](#2-comparação-de-modelos-comparepy) |
| Inspecionar checkpoints | Lista os `.pth` salvos e valida se ainda carregam | [Funcionalidades §6](#6-ferramentas-de-desenvolvimento-e-validação-scripts) |
| Super Resolução em Tempo Real | Captura vídeo/tela e exibe SR ao vivo | [Funcionalidades §3](#3-super-resolução-em-tempo-real-inference_realtimepy) |
| Configurações | Troca de preset ou ajusta hiperparâmetros interativamente | [Funcionalidades §1](#1-treino) |
| Testar CUDA | Mostra build do torch e diagnóstico de GPU | [§2](#2-instalação) |

## 5. Estrutura do projeto

```
prototipotcc/
├── training-python/            # Módulo de treino/inferência Python (este README)
│   ├── app.py                  # Menu principal — ponto de entrada
│   ├── train.py                # Loop de treino (Vimeo + fallback SISR)
│   ├── compare.py              # Avaliação PSNR/SSIM + comparação visual
│   ├── inference_realtime.py   # SR em tempo real via captura de tela/vídeo/webcam
│   ├── requirements.txt
│   ├── models/
│   │   ├── registry.py         # Registry de arquiteturas
│   │   ├── blocks.py           # Blocos compartilhados entre arquiteturas (SEBlock, ConvGRU, aligners)
│   │   └── __init__.py         # Importa cada módulo para registrar
│   ├── data/prepare_dataset.py # Download e extração de datasets
│   ├── utils/
│   │   ├── cli.py              # Menus interativos (questionary)
│   │   └── config.py           # ConfigManager singleton
│   ├── presets/config.json     # Config ativo
│   ├── checkpoints/            # Pesos salvos (.pth), um por model_type
│   ├── datasets/               # Datasets baixados (git-ignored)
│   ├── logs/                   # CSV de métricas por época, um por model_type
│   ├── assets/
│   │   ├── videos/             # Vídeos de teste (inference_realtime.py, compare_video_outputs.py)
│   │   └── plots/              # Tudo que é gráfico/imagem gerada
│   │       ├── comparison.png          # compare.py
│   │       ├── hparam_search/          # plot_hparam_search.py
│   │       └── training_history/       # plot_training_history.py
│   └── scripts/                # Ferramentas de dev/validação, fora do menu principal
│       ├── inspect_ckpts.py         # Inspeciona/valida checkpoints salvos
│       ├── smoke_test_train.py      # Teste rápido de forward/backward por arquitetura
│       ├── hparam_search.py         # Busca de hiperparâmetros com Optuna (TPE + pruning)
│       ├── plot_hparam_search.py    # Gráficos da busca de hiperparâmetros
│       ├── plot_training_history.py # Gráficos de loss/PSNR/SSIM/TCE por época + detecção de platô
│       ├── export_trt.py            # Exporta um checkpoint para ONNX (consumido pelo inference-cpp)
│       ├── compare_video_outputs.py # Valida paridade numérica PyTorch vs inference-cpp
│       └── valid_model.py           # Inspeciona o .onnx exportado
└── inference-cpp/              # Motor de inferência nativo C++/TensorRT (WIP)
    ├── CMakeLists.txt          # Paths de OpenCV/TensorRT ainda hardcoded p/ ambiente local
    ├── include/RTDVSRInferencer.hpp
    └── src/
        ├── main.cpp
        └── RTDVSRInferencer.cpp
```

Os scripts em `scripts/` importam `models`/`train`/`utils` do diretório pai
via um pequeno bootstrap de `sys.path` no topo do arquivo — por isso
continuam rodando de dentro de `training-python/`:

```bash
cd training-python
python scripts/inspect_ckpts.py
python scripts/hparam_search.py
```

---

# Funcionalidades em Detalhe

## 1. Treino

`train.py` (menu "Treinar modelo") treina o `model_type` configurado sobre
o Vimeo Septuplet, com janelas temporais de `seq_len` frames. Loss:
Charbonnier (L1 suavizado) + peso de consistência temporal
(`temporal_loss_weight`) + peso de esparsidade de máscara pro
`MaskedRecurrentVSR` (`mask_loss_weight`). Precisão automática (BF16 > FP16
> FP32, conforme suporte da GPU).

### Configuração (`presets/config.json`)

```json
{
    "dataset_path": "./datasets",
    "vid4_path": "C:/Users/usuario/Downloads/archive",
    "scale_factor": 2,
    "seq_len": 3,
    "learning_rate": 0.0002,
    "batch_size": 4,
    "epochs": 50,
    "crop_size": 256,
    "model_type": "RTDVSR",
    "model_params": {
        "hidden_dim": 64,
        "num_res_blocks": 6
    },
    "checkpoint_dir": "./checkpoints",
    "logs_dir": "./logs"
}
```

| Chave | Padrão se ausente | Descrição |
|---|---|---|
| `model_type` | `LightweightVSR` | Nome registrado no registry de modelos (ver [§5](#5-adicionando-uma-nova-arquitetura)) |
| `model_params` | `{hidden_dim: 64, num_res_blocks: 6}` | kwargs repassados ao construtor do modelo — variam por arquitetura |
| `scale_factor` | `2` | Fator de upscaling (2 ou 4) |
| `seq_len` | `3` | Tamanho da janela temporal por amostra (máx. 7) |
| `learning_rate` | `1e-3` | LR inicial (warmup linear de 1 época + cosine annealing com restarts) |
| `batch_size` | `8` | Batch de treino |
| `epochs` | `100` | Épocas máximas |
| `crop_size` | `96` | Crop HR usado no fallback SISR (o Vimeo usa a imagem inteira 448×256) |
| `dataset_path` | `./datasets` | Raiz onde o Vimeo Septuplet foi extraído |
| `vid4_path` | — | Pasta do Vid4 (usado só em "Comparar modelos") |
| `checkpoint_dir` | `./checkpoints` | Onde salvar/carregar o `.pth` |
| `logs_dir` | `./logs` | Onde salvar o CSV de métricas por época |
| `temporal_loss_weight` | `0.1` | Peso da perda de consistência temporal |
| `mask_loss_weight` | `0.05` | Peso da perda de esparsidade (só `MaskedRecurrentVSR`) |
| `max_val_samples` | `500` | Limite de amostras de validação por época |
| `val_ssim_budget` | `200` | Quantos frames calculam SSIM por época (caro; PSNR é calculado em todos) |
| `target_psnr` / `target_inference_ms` | `27.0` / `16.0` | Critério de early stopping por qualidade+velocidade |
| `early_stopping_patience` | `10` | Épocas sem melhora de PSNR antes de parar |

Pode-se manter múltiplos presets em `presets/*.json` e alternar pelo menu
"Configurações → Utilizar presets", ou ajustar interativamente em
"Configurações → Definir parâmetros de treino" (essa opção pergunta só os
parâmetros que a arquitetura escolhida de fato aceita, via introspecção do
construtor — não é uma lista fixa de campos).

### Checkpoints e retomada

O checkpoint é salvo em `checkpoints/<model_type>_best_model.pth` (o nome
do arquivo vem do `model_type`, não de uma chave de config) sempre que o
PSNR de validação melhora. Ele guarda os pesos, o optimizer, o `model_type`
e `model_params` usados — por isso `compare.py` e `inference_realtime.py`
reconstroem a arquitetura automaticamente ao carregar, sem precisar saber
de antemão qual foi treinada.

Se o arquivo já existir quando `train.py` roda de novo, o treino **retoma**
da época seguinte automaticamente (pesos, optimizer, scaler e melhor PSNR
são restaurados). Pra treinar do zero, apague o `.pth` correspondente.

### Histórico de treino (loss/PSNR por época)

Toda execução de `train.py` salva uma linha por época em
`logs/<model_type>_training_log.csv`: `train_loss`, `val_psnr`, `val_ssim`,
`val_tce`, `inference_ms`, `learning_rate`, se foi o melhor checkpoint até
então, e timestamp. Retomar de um checkpoint continua o mesmo log; começar
do zero descarta o log anterior daquele `model_type`.

```bash
python scripts/plot_training_history.py                  # todos os logs em logs/
python scripts/plot_training_history.py --model-type RTDVSR
python scripts/plot_training_history.py --min-delta 0.05 --patience 5
```

Salva `assets/plots/training_history/<model_type>.png` com 4 painéis
(Train Loss, Val PSNR, Val SSIM, Val TCE). No painel de PSNR, marca a
melhor época e detecta a partir de qual época o ganho estagnou (sem
melhora maior que `--min-delta` dB por `--patience` épocas seguidas) — é o
gráfico pra frase tipo "até a época X teve Y dB de melhora, depois
estagnou".

### "Testar treinamento (3 épocas)" vs. "Smoke test"

São coisas diferentes, apesar do nome parecido:

- **"Testar treinamento (3 épocas)"** roda o `train.py` de verdade, com o
  dataset real, só que forçando `epochs=3` temporariamente. Serve pra
  validar que o pipeline completo (dataset → treino → checkpoint → log)
  funciona antes de um treino longo. Exige o Vimeo Septuplet baixado.
- **"Smoke test"** (`scripts/smoke_test_train.py`) roda um único passo de
  forward/backward em 3 arquiteturas recorrentes com **tensores
  sintéticos** (sem dataset), em FP32 e BF16 — checa NaN, gradientes e
  faz um benchmark de inferência. É bem mais rápido e não depende de
  nada estar baixado; serve pra pegar bug de arquitetura antes de
  qualquer treino.

## 2. Comparação de modelos (`compare.py`)

Menu "Comparar modelos". Fluxo:

1. Seleciona um ou mais checkpoints em `checkpoints/` (multi-seleção).
2. Escolhe o dataset de avaliação: **Vimeo Septuplet** (`sep_testlist`) ou
   **Vid4** (benchmark clássico — informe o caminho da pasta `archive` com
   `GT/BIx4/BDx4`, ou salve em `vid4_path` no config). No Vid4, o LR é
   gerado on-the-fly a partir do GT via bicúbico, no `scale_factor` de
   **cada** checkpoint — assim dá pra comparar modelos x2, x3 e x4 no
   mesmo benchmark sem precisar de LRs pré-computados por escala.
3. Avaliação quantitativa: PSNR/SSIM (arquiteturas `recurrent` avaliam
   todos os frames da sequência; `sliding_window` só o frame central) e
   TCE (Temporal Consistency Error) entre frames consecutivos. Imprime uma
   tabela comparativa no terminal.
4. Comparação visual: gera `assets/plots/comparison.png` com uma grade
   Bicúbico | modelo 1 | ... | HR (ground truth), com o PSNR de cada
   modelo anotado.

## 3. Super-resolução em tempo real (`inference_realtime.py`)

Menu "Super Resolução em Tempo Real". Escolhe um checkpoint, depois a
fonte de vídeo:

- **Webcam**
- **Arquivo de vídeo** — busca na raiz de `training-python/`, em
  `data/**` (recursivo) e em `assets/videos/`, ou aceita caminho manual
- **Captura de tela** (com seleção de monitor, se houver mais de um)

O pipeline é sempre: captura → downscale pra 960×540 (LR fixo) →
inferência → upscale `scale_factor`×. Modo de exibição (alternável em
tempo real):

- **Comparação lado a lado** (Bicúbico vs SR)
- **Apenas SR** (tela cheia)
- **Triplo** (Original | Bicúbico | SR)

Controles durante a execução: `q` sai, `m` alterna o modo de exibição,
`p` pausa/retoma. O HUD no canto inferior mostra FPS, latência por frame,
modelo e device em uso.

## 4. Busca de hiperparâmetros (`scripts/hparam_search.py`)

Usa [Optuna](https://optuna.org/) com sampler TPE e `MedianPruner` pra
buscar `learning_rate`, `hidden_dim`, `num_res_blocks` e
`temporal_loss_weight` do RTDVSR — mais eficiente que grid search
exaustivo quando só 1-2 hiperparâmetros dominam o resultado. Cada trial
treina por poucas épocas (`--epochs`) com cada época limitada a
`max_train_batches` batches (não o dataset inteiro), e é interrompido cedo
se o PSNR estiver visivelmente pior que a mediana dos trials anteriores no
mesmo ponto.

```bash
python scripts/hparam_search.py
python scripts/hparam_search.py --n-trials 50 --epochs 4
```

O progresso persiste em `rtdvsr_hparam_search.db` (SQLite) — interromper
(Ctrl+C) e rodar de novo **retoma** o estudo em vez de recomeçar do zero.

**Acompanhando ao vivo**, em outro terminal:

```bash
optuna-dashboard sqlite:///rtdvsr_hparam_search.db
```

Abre um dashboard web local com os trials atualizando em tempo real, sem
precisar mexer em código.

**Gráficos estáticos pro TCC**, depois de rodar (ou interromper) a busca:

```bash
python scripts/plot_hparam_search.py
```

Gera 5 PNGs em `assets/plots/hparam_search/`:

| Arquivo | O que mostra |
|---|---|
| `optimization_history.png` | PSNR de cada trial ao longo da busca, com o melhor valor até então |
| `param_importances.png` | Quais hiperparâmetros mais influenciaram o PSNR |
| `intermediate_values.png` | Curva de PSNR por época de cada trial — mostra visualmente os trials podados |
| `parallel_coordinate.png` | Relação entre combinações de hiperparâmetros e o PSNR resultante |
| `slice.png` | PSNR em função de cada hiperparâmetro individualmente |

> Limitação atual: hardcoded pro RTDVSR (`model_type` fixo em `FIXED`).
> Generalizar pras outras arquiteturas seguiria o mesmo padrão de
> introspecção do construtor usado em `utils/cli.py` (`get_model_class` +
> `inspect.signature`).

## 5. Adicionando uma nova arquitetura

O projeto usa um **registry** que desacopla arquitetura do loop de treino.
Basta decorar sua classe com `@register_model`.

**5.1 Criar o arquivo da arquitetura** — `models/MinhaArquitetura.py`:

```python
import torch
import torch.nn as nn
from .registry import register_model


@register_model("MinhaArquitetura", interface="recurrent")
class MinhaArquitetura(nn.Module):
    """
    Parâmetros obrigatórios no construtor:
        scale_factor: fator de upscaling
        channels: canais da imagem (3 para RGB)
    Parâmetros livres vão em model_params no config.json.
    """

    def __init__(self, scale_factor=2, channels=3, hidden_dim=64, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        # ... defina seus módulos ...

    def forward(self, x, prev_state=None):
        # x: (B, C, H, W) — um único frame LR
        # prev_state: estado oculto do frame anterior (ou None no t=0)
        # Retorna: (sr, new_state)
        ...
        return sr, new_state
```

Se sua arquitetura reaproveita blocos comuns (SE-attention, ConvGRU,
alinhamento deformável), veja `models/blocks.py` antes de reimplementar.

**5.2 Registrar o módulo** — edite `models/__init__.py` e importe o novo
arquivo para que o decorator rode na hora do `import`:

```python
from .registry import register_model, get_model, get_interface, list_models, get_model_class
from . import LightweightVSR
from . import MinhaArquitetura  # <-- adicionar
```

**5.3 Interfaces suportadas** — o parâmetro `interface` do decorator diz
ao `train.py` / `compare.py` / `inference_realtime.py` como chamar seu
`forward`:

- **`recurrent`** (causal, online) — `forward(frame, state) -> (sr, new_state)`.
  Processa frame a frame, carregando um hidden state. É a interface usada
  por todas as arquiteturas atuais e a recomendada para streaming em
  tempo real.
- **`sliding_window`** — `forward(frames) -> sr`. Recebe uma janela
  `(B, T, C, H, W)` e devolve o SR do frame central. Útil para modelos
  não-causais que olham para frames passados e futuros.

**5.4 Apontar o config para o novo modelo:**

```json
{
    "model_type": "MinhaArquitetura",
    "model_params": {
        "hidden_dim": 96,
        "seu_param_especifico": 4
    }
}
```

Tudo que estiver em `model_params` é repassado como `**kwargs` para o
construtor. Não é preciso mexer em `train.py` nem `compare.py`.

**5.5 Rodar:**

```bash
python app.py
# Configurações -> Definir parâmetros de treino (escolha MinhaArquitetura)
# Treinar modelo
```

O checkpoint salva `model_type` e `model_params`, então `compare.py` e
`inference_realtime.py` reconstroem a arquitetura correta automaticamente —
você pode comparar vários modelos diferentes no mesmo gráfico.

## 6. Ferramentas de desenvolvimento e validação (`scripts/`)

Não fazem parte do fluxo principal do menu (exceto onde indicado), mas
todas rodam de dentro de `training-python/`:

- **`inspect_ckpts.py`** (menu "Inspecionar checkpoints") — carrega todos
  os `.pth` de `checkpoints/`, imprime metadados (epoch, PSNR, TCE, tempo
  de inferência, arquitetura, tamanho do optimizer) e testa se cada um
  ainda carrega no código atual (`load_state_dict(strict=True)`). Útil
  depois de mexer nas classes dos modelos, pra garantir que checkpoints
  antigos continuam compatíveis.
- **`smoke_test_train.py`** (menu "Smoke test") — ver [§1](#testar-treinamento-3-épocas-vs-smoke-test).
- **`hparam_search.py`** / **`plot_hparam_search.py`** — ver [§4](#4-busca-de-hiperparâmetros-scriptshparam_searchpy).
- **`plot_training_history.py`** — ver [§1](#histórico-de-treino-losspsnr-por-época).
- **`export_trt.py`** — exporta um checkpoint do RTDVSR para ONNX
  (`../inference-cpp/rtdvsr.onnx`, opset 19), incluindo uma ponte de
  tradução manual do `DeformConv2d` do TorchVision (que não tem exportador
  ONNX oficial) para o operador `DeformConv` padrão do ONNX. É o arquivo
  que o `inference-cpp/` consome.
- **`compare_video_outputs.py`** — roda o mesmo vídeo pelo motor PyTorch e
  pelo executável C++/TensorRT já compilado (`inference-cpp/build/Debug/`)
  e calcula PSNR/MSE entre os dois, pra validar que a portagem pra C++
  não introduziu divergência numérica.
- **`valid_model.py`** — inspeciona rapidamente o `.onnx` exportado
  (conta operadores por tipo). Aviso: tem um bug conhecido — chama
  `model.parameters()` num `onnx.ModelProto`, que não existe nessa classe
  — quebra se executado como está.

---

# Solução de problemas

- **"CUDA disponível: Não"** — você tem o build `torch+cpu`. Veja
  [§2](#2-instalação).
- **"Caindo para dataset genérico de imagens (modo SISR, sem
  temporalidade)"** — o caminho do Vimeo está errado. Confira o layout em
  [§3](#3-dataset). Os caminhos inspecionados aparecem no log.
- **`TypeError: __init__() got an unexpected keyword argument 'verbose'`**
  — PyTorch 2.7+ removeu `verbose` de `ReduceLROnPlateau`. O projeto já
  está corrigido; se aparecer de novo, é porque algum fork reintroduziu.
- **`DeformConv2d` indisponível** — os modelos que usam alinhamento
  deformável (`RTDVSR`, `PyramidVSR`, `LightweightVSR`) caem em fallback
  de convolução padrão, com aviso no console. Qualidade reduzida mas
  funcional.
