import os
import questionary
from utils.config import ConfigManager


def purificar_terminal():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def menu():
    escolha = questionary.select(
        "Escolha uma opção:",
        choices=[
            "Baixar dataset",
            "Treinar modelo",
            "Testar treinamento (3 épocas)",
            "Smoke test (forward/backward rápido)",
            "Comparar modelos",
            "Inspecionar checkpoints",
            "Super Resolução em Tempo Real",
            "Configurações",
            "Testar CUDA",
            "Sair"
        ]
    ).ask()
    return escolha


def menu_configuracoes():
    config_manager = ConfigManager.get_instance()
    config_manager.show_config()

    escolha = questionary.select(
        "Configurações:",
        choices=[
            "Utilizar presets",
            "Definir parâmetros de treino",
            "Voltar ao menu principal"
        ]
    ).ask()

    if escolha == "Utilizar presets":
        carregar_preset()
    elif escolha == "Definir parâmetros de treino":
        definir_parametros()

    return escolha


def carregar_preset():
    presets_dir = "presets"

    if not os.path.exists(presets_dir):
        print("Pasta de presets não encontrada.")
        return

    presets = [f for f in os.listdir(presets_dir) if f.endswith('.json')]

    if not presets:
        print("Nenhum preset encontrado.")
        return

    preset_escolhido = questionary.select(
        "Escolha um preset:",
        choices=presets
    ).ask()

    if preset_escolhido:
        preset_path = os.path.join(presets_dir, preset_escolhido)
        ConfigManager.load_config(preset_path)
        print(f"Preset '{preset_escolhido}' carregado com sucesso!")


def _ask_model_params(model_type, current_params):
    """Pergunta os parâmetros do construtor de `model_type` (via inspect),
    em vez de um conjunto fixo de campos que não vale para toda arquitetura."""
    import inspect
    from models import get_model_class

    sig = inspect.signature(get_model_class(model_type).__init__)
    skip = {"self", "scale_factor", "channels"}

    model_params = {}
    for name, param in sig.parameters.items():
        if name in skip or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        default = current_params.get(
            name, param.default if param.default is not inspect.Parameter.empty else 0)
        value = questionary.text(f"{name}:", default=str(default)).ask()
        if isinstance(default, bool):
            model_params[name] = value.strip().lower() in ("true", "1", "yes", "sim")
        else:
            model_params[name] = type(default)(value)

    return model_params


def definir_parametros():
    from models import list_models

    config_manager = ConfigManager.get_instance()

    model_type = questionary.select(
        "Arquitetura do modelo:",
        choices=list_models()
    ).ask()

    scale_factor = questionary.select(
        "Fator de escala:",
        choices=["2", "4"]
    ).ask()

    seq_len = questionary.text(
        "Tamanho da janela de frames (seq_len, ex: 3 para n-1,n,n+1):",
        default=str(config_manager.get('seq_len', 3))
    ).ask()

    learning_rate = questionary.text(
        "Learning rate:",
        default=str(config_manager.get('learning_rate', 0.001))
    ).ask()

    batch_size = questionary.text(
        "Batch size:",
        default=str(config_manager.get('batch_size', 8))
    ).ask()

    epochs = questionary.text(
        "Número de epochs:",
        default=str(config_manager.get('epochs', 50))
    ).ask()

    print(f"\nDefina os parâmetros específicos de {model_type} "
          f"(deixe em branco para usar o padrão).")
    # Só reaproveita os valores salvos se forem da mesma arquitetura —
    # parâmetros de um modelo diferente não fazem sentido para este.
    current_params = (config_manager.get('model_params', {})
                       if config_manager.get('model_type') == model_type else {})
    model_params = _ask_model_params(model_type, current_params)

    nova_config = config_manager.get_config().copy()
    nova_config.update({
        "model_type": model_type,
        "scale_factor": int(scale_factor),
        "seq_len": int(seq_len),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "model_params": model_params,
    })

    config_manager.new_config(nova_config)
    print("Parâmetros definidos com sucesso!")
