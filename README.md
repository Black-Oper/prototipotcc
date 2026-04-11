# Super-Resolução de Vídeo em Tempo Real

## 1. Requisitos

- Windows 10/11, Linux ou macOS
- Python 3.9+
- (Recomendado) GPU NVIDIA com driver compatível com CUDA 12.x

## 2. Instalação

```bash
git clone <seu-repo> prototipotcc
cd prototipotcc

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
| Comparar modelos | Avalia PSNR/SSIM e gera `comparison.png` lado a lado |
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
from .registry import register_model, get_model, get_interface, list_models
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
├── app.py                     # Menu principal
├── train.py                   # Loop de treino (Vimeo + fallback SISR)
├── compare.py                 # Avaliação PSNR/SSIM + comparação visual
├── inference_realtime.py      # SR em tempo real via captura de tela
├── models/
│   ├── registry.py            # Registry de arquiteturas
│   ├── LightweightVSR.py      # Arquitetura baseline
│   └── __init__.py            # Importa cada módulo para registrar
├── data/prepare_dataset.py    # Download e extração de datasets
├── utils/
│   ├── cli.py                 # Menus interativos (questionary)
│   └── config.py              # ConfigManager singleton
├── presets/config.json        # Config ativo
├── checkpoints/               # Pesos salvos (.pth)
└── datasets/                  # Datasets baixados
```

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
