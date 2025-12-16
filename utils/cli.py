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
            "Avaliar modelo",
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
            "Definir parâmetros do modelo",
            "Voltar ao menu principal"
        ]
    ).ask()
    
    if escolha == "Utilizar presets":
        carregar_preset()
    elif escolha == "Definir parâmetros do modelo":
        definir_parametros()
    
    return escolha

def carregar_preset():
    
    import os
    
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
    
    config_manager = ConfigManager.get_instance()
    
    learning_rate = questionary.text(
        "Learning rate:",
        default="0.001"
    ).ask()
    
    batch_size = questionary.text(
        "Batch size:",
        default="16"
    ).ask()
    
    epochs = questionary.text(
        "Número de epochs:",
        default="100"
    ).ask()
    
    model_type = questionary.select(
        "Tipo de modelo:",
        choices=["ESPCN"]
    ).ask()
    
    scale_factor = questionary.select(
        "Fator de escala:",
        choices=["2"]
    ).ask()
    
    nova_config = {
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "model_type": model_type,
        "scale_factor": int(scale_factor)
    }
    
    config_manager.new_config(nova_config)
    print("Parâmetros definidos com sucesso!")


