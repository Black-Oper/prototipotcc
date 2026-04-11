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
            "Comparar modelos",
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

    # Parâmetros específicos do modelo (model_params)
    print("\nDefina os parâmetros específicos da arquitetura (deixe em branco para usar o padrão do preset).")
    current_params = config_manager.get('model_params', {})

    hidden_dim = questionary.text(
        "hidden_dim:",
        default=str(current_params.get('hidden_dim', 64))
    ).ask()

    num_res_blocks = questionary.text(
        "num_res_blocks:",
        default=str(current_params.get('num_res_blocks', 6))
    ).ask()

    nova_config = config_manager.get_config().copy()
    nova_config.update({
        "model_type": model_type,
        "scale_factor": int(scale_factor),
        "seq_len": int(seq_len),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "model_params": {
            "hidden_dim": int(hidden_dim),
            "num_res_blocks": int(num_res_blocks),
        },
    })

    config_manager.new_config(nova_config)
    print("Parâmetros definidos com sucesso!")
