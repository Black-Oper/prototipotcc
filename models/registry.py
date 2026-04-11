"""
Registry de arquiteturas VSR.

Para registrar um novo modelo, use o decorator @register_model:

    from models.registry import register_model

    @register_model("MinhaArq", interface="sliding_window")
    class MinhaArquitetura(nn.Module):
        def __init__(self, scale_factor=2, channels=3, **kwargs):
            ...
        def forward(self, frames):  # (B, T, C, H, W) -> (B, C, H*s, W*s)
            ...

Interfaces suportadas:
  "recurrent"      — forward(frame, state) -> (sr, new_state)
                     Processa frame a frame com estado oculto (online/causal).
  "sliding_window" — forward(frames) -> sr
                     Recebe janela de T frames, retorna SR do frame central.
"""

MODEL_REGISTRY: dict = {}


def register_model(name: str, interface: str = "recurrent"):
    """Decorator que registra uma classe de modelo no registry global."""
    if interface not in ("recurrent", "sliding_window"):
        raise ValueError(f"interface deve ser 'recurrent' ou 'sliding_window', não '{interface}'")

    def decorator(cls):
        MODEL_REGISTRY[name] = {"class": cls, "interface": interface}
        return cls

    return decorator


def get_model(name: str, **kwargs):
    """Instancia um modelo pelo nome, repassando kwargs ao construtor."""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Modelo '{name}' não encontrado. Disponíveis: {available}")
    return MODEL_REGISTRY[name]["class"](**kwargs)


def get_interface(name: str) -> str:
    """Retorna a interface do modelo ('recurrent' ou 'sliding_window')."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{name}' não encontrado.")
    return MODEL_REGISTRY[name]["interface"]


def list_models() -> list:
    """Retorna lista de nomes de modelos registrados."""
    return list(MODEL_REGISTRY.keys())
