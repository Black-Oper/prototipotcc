# Super-Resolução de Vídeo em Tempo Real

O projeto tem dois módulos independentes:

- **`training-python/`** — treino, avaliação e inferência em tempo real via
  PyTorch. É o módulo documentado neste README.
- **`inference-cpp/`** — motor de inferência nativo em C++/TensorRT (WIP),
  consumindo o `.onnx` exportado por `training-python/export_trt.py`. Tem
  build própria via CMake; veja `inference-cpp/CMakeLists.txt`.

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

### 2.1 PyTorch com CUDA (importante)

O `requirements.txt` lista `torch` sem índice específico. Isso faz o pip
resolver o **wheel CPU-only** por padrão no Windows, e o menu "Testar CUDA"
vai reportar "CUDA disponível: Não".

Para habilitar a GPU, instale o build CUDA **antes** do resto das dependências:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Ajuste `cu121` para a versão do CUDA Toolkit suportada pelo seu driver NVIDIA
(veja https://pytorch.org/get-started/locally/).

### 2.2 Demais dependências

```bash
pip install -r requirements.txt
```

### 2.3 Verificar a GPU

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

## 4. Como rodar

Todas as ações passam pelo menu principal:

```bash
python app.py
```

Opções:

| Opção | O que faz |
|-------|-----------|
| Baixar dataset | Faz download e extrai o Vimeo Septuplet em `./datasets` |
| Treinar modelo | Treina o modelo definido em `presets/config.json` |
| Testar treinamento (3 épocas) | Smoke test rápido do pipeline de treino |
| Comparar modelos | Avalia PSNR/SSIM e gera `assets/plots/comparison.png` lado a lado |
| Super Resolução em Tempo Real | Captura a tela e exibe SR ao vivo (`q` sai, `c` alterna comparação) |
| Configurações | Troca de preset ou ajuste interativo de hiperparâmetros |
| Testar CUDA | Mostra build do torch e diagnóstico de GPU |

Checkpoints são salvos em `./checkpoints/best_model.pth` e o treino retoma
automaticamente se o arquivo existir.

## 5. Configuração (`presets/config.json`)

```json
{
    "dataset_path": "./datasets",
    "scale_factor": 2,
    "seq_len": 3,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 50,
    "crop_size": 96,
    "model_type": "LightweightVSR",
    "model_params": {
        "hidden_dim": 64,
        "num_res_blocks": 6
    },
    "checkpoint_dir": "./checkpoints",
    "checkpoint_name": "best_model.pth"
}
```

- `model_type` — nome registrado no registry de modelos (ver §6)
- `model_params` — kwargs repassados ao construtor do modelo
- `seq_len` — tamanho da janela temporal por amostra (máx. 7)
- `scale_factor` — fator de upscaling (2 ou 4)

Pode-se manter múltiplos presets em `presets/*.json` e alternar pelo menu
"Configurações → Utilizar presets".

## 6. Adicionando um novo modelo

O projeto usa um **registry** que desacopla arquitetura do loop de treino.
Basta decorar sua classe com `@register_model`.

### 6.1 Criar o arquivo da arquitetura

Crie `models/MinhaArquitetura.py`:

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

### 6.2 Registrar o módulo

Edite `models/__init__.py` e importe o novo arquivo para que o decorator
rode na hora do `import`:

```python
from .registry import register_model, get_model, get_interface, list_models, get_model_class
from . import LightweightVSR
from . import MinhaArquitetura  # <-- adicionar
```

### 6.3 Interfaces suportadas

O parâmetro `interface` do decorator diz ao `train.py` / `compare.py` /
`inference_realtime.py` como chamar seu `forward`:

- **`recurrent`** (causal, online) — `forward(frame, state) -> (sr, new_state)`
  Processa frame a frame, carregando um hidden state. É a interface usada
  pelo `LightweightVSR` e a recomendada para streaming em tempo real.

- **`sliding_window`** — `forward(frames) -> sr`
  Recebe uma janela `(B, T, C, H, W)` e devolve o SR do frame central.
  Útil para modelos não-causais que olham para frames passados e futuros.

### 6.4 Apontar o config para o novo modelo

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

### 6.5 Rodar

```bash
python app.py
# Configurações -> Definir parâmetros de treino (escolha MinhaArquitetura)
# Treinar modelo
```

O checkpoint salva `model_type` e `model_params`, então `compare.py` e
`inference_realtime.py` reconstroem a arquitetura correta automaticamente —
você pode comparar vários modelos diferentes no mesmo gráfico.

## 7. Estrutura do projeto

```
prototipotcc/
├── training-python/            # Módulo de treino/inferência Python (este README)
│   ├── app.py                  # Menu principal — ponto de entrada
│   ├── train.py                # Loop de treino (Vimeo + fallback SISR)
│   ├── compare.py               # Avaliação PSNR/SSIM + comparação visual
│   ├── inference_realtime.py   # SR em tempo real via captura de tela
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
│   ├── checkpoints/            # Pesos salvos (.pth)
│   ├── datasets/               # Datasets baixados (git-ignored)
│   ├── logs/                   # CSV de métricas por época, um por model_type (gerado por train.py)
│   ├── assets/
│   │   ├── videos/             # Vídeos de teste (inference_realtime.py, compare_video_outputs.py)
│   │   └── plots/              # Tudo que é gráfico/imagem gerada
│   │       ├── comparison.png          # compare.py
│   │       ├── hparam_search/          # plot_hparam_search.py
│   │       └── training_history/       # plot_training_history.py
│   └── scripts/                # Ferramentas de dev/validação, fora do menu principal
│       ├── inspect_ckpts.py    # Inspeciona/valida checkpoints salvos
│       ├── smoke_test_train.py # Teste rápido de forward/backward por arquitetura
│       ├── hparam_search.py    # Busca de hiperparâmetros com Optuna (TPE + pruning)
│       ├── plot_hparam_search.py   # Gráficos da busca de hiperparâmetros
│       ├── plot_training_history.py # Gráficos de loss/PSNR/SSIM/TCE por época + detecção de platô
│       ├── export_trt.py       # Exporta um checkpoint para ONNX (consumido pelo inference-cpp)
│       ├── compare_video_outputs.py # Valida paridade numérica PyTorch vs inference-cpp
│       └── valid_model.py      # Inspeciona o .onnx exportado
└── inference-cpp/              # Motor de inferência nativo C++/TensorRT (WIP)
    ├── CMakeLists.txt          # Paths de OpenCV/TensorRT ainda hardcoded p/ ambiente local
    ├── include/RTDVSRInferencer.hpp
    └── src/
        ├── main.cpp
        └── RTDVSRInferencer.cpp
```

Os scripts em `scripts/` importam `models`/`train`/`utils` do diretório pai
via um pequeno bootstrap de `sys.path` no topo do arquivo — por isso
continuam rodando de dentro de `training-python/`, por exemplo:

```bash
cd training-python
python scripts/hparam_search.py
python scripts/inspect_ckpts.py
```

### 7.1 Acompanhando e apresentando a busca de hiperparâmetros

`hparam_search.py` salva o progresso incrementalmente em
`rtdvsr_hparam_search.db` (SQLite). Duas formas de acompanhar/usar isso:

**Ao vivo, enquanto a busca roda** — em outro terminal:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///rtdvsr_hparam_search.db
```

Abre um dashboard web local com os trials atualizando em tempo real
(gráficos de convergência, importância de hiperparâmetro, etc.), sem
precisar mexer em código.

**Gráficos estáticos para o TCC** — depois de rodar (ou interromper) a
busca:

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

### 7.2 Histórico de treino (loss/PSNR por época)

Todo treino via `train.py` (menu "Treinar modelo") salva uma linha por
época em `logs/<model_type>_training_log.csv` — `train_loss`, `val_psnr`,
`val_ssim`, `val_tce`, `inference_ms`, `learning_rate`, se foi o melhor
checkpoint até então, e timestamp. Interromper e retomar o treino a
partir de um checkpoint continua o mesmo log; começar um treino do zero
descarta o log anterior daquele `model_type`.

Para gerar o gráfico:

```bash
python scripts/plot_training_history.py                  # todos os logs em logs/
python scripts/plot_training_history.py --model-type RTDVSR
```

Salva `assets/plots/training_history/<model_type>.png` com 4 painéis
(Train Loss, Val PSNR, Val SSIM, Val TCE). No painel de PSNR, marca a
melhor época e detecta a partir de qual época o ganho estagnou (sem
melhora maior que `--min-delta` dB por `--patience` épocas seguidas,
padrão 0.05 dB / 5 épocas) — é o gráfico pra frase tipo "até a época X
teve Y dB de melhora, depois estagnou".

## 8. Solução de problemas

- **"CUDA disponível: Não"** — você tem o build `torch+cpu`. Veja §2.1.
- **"Caindo para dataset genérico de imagens (modo SISR, sem temporalidade)"** —
  o caminho do Vimeo está errado. Confira o layout em §3. Os caminhos
  inspecionados aparecem no log.
- **`TypeError: __init__() got an unexpected keyword argument 'verbose'`** —
  PyTorch 2.7+ removeu `verbose` de `ReduceLROnPlateau`. O projeto já está
  corrigido; se aparecer de novo, é porque algum fork reintroduziu.
- **`DeformConv2d` indisponível** — o `LightweightVSR` tem fallback para
  convolução padrão com aviso no console. Qualidade reduzida mas funcional.
