from utils import cli
from data import prepare_dataset
from train import train
from inference_realtime import run_realtime
from compare import evaluate_and_compare
import torch


def main():

    cli.purificar_terminal()

    while True:

        cli.purificar_terminal()

        escolha = cli.menu()

        if escolha == "Baixar dataset":
            baixar_dataset()
        elif escolha == "Treinar modelo":
            treinar_modelo()
        elif escolha == "Testar treinamento (3 épocas)":
            testar_treinamento()
        elif escolha == "Comparar modelos":
            comparar_modelos()
        elif escolha == "Super Resolução em Tempo Real":
            super_resolucao_realtime()
        elif escolha == "Configurações":
            configurar()
        elif escolha == "Testar CUDA":
            testar_cuda()
        elif escolha == "Sair":
            print("Saindo do programa...")
            break


def testar_cuda():
    print(f"Torch: {torch.__version__}")
    print(f"Build CUDA do torch: {torch.version.cuda or 'nenhum (wheel CPU-only)'}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA disponível: Sim")
        print(f"Número de GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA disponível: Não")
        if torch.version.cuda is None:
            print("Motivo: o PyTorch instalado é a build CPU-only.")
            print("Reinstale com a build CUDA, por exemplo:")
            print("  pip uninstall -y torch torchvision")
            print("  pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu121")
        else:
            print("Motivo: PyTorch foi compilado com CUDA, mas o driver/GPU "
                  "não foi detectado. Verifique o driver NVIDIA.")
        print("O modelo rodará na CPU (mais lento).")
    print("\nPressione Enter para continuar...")
    input()


def baixar_dataset():
    prepare_dataset.download_datasets()
    print("\nPressione Enter para continuar...")
    input()


def treinar_modelo():
    train()
    print("\nPressione Enter para continuar...")
    input()


def testar_treinamento():
    from utils.config import ConfigManager
    try:
        ConfigManager.load_config('presets/config.json')
    except Exception:
        pass
    config = ConfigManager.get_instance()
    epochs_originais = config.get('epochs', 50)
    config.get_config()['epochs'] = 3
    print("Modo de teste: treinando por 3 épocas...\n")
    train()
    config.get_config()['epochs'] = epochs_originais
    print("\nPressione Enter para continuar...")
    input()


def comparar_modelos():
    evaluate_and_compare()
    print("\nPressione Enter para continuar...")
    input()


def super_resolucao_realtime():
    run_realtime()
    print("\nPressione Enter para continuar...")
    input()


def configurar():
    cli.menu_configuracoes()
    print("\nPressione Enter para continuar...")
    input()


if __name__ == "__main__":
    main()
