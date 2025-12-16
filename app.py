from utils import cli
from data import prepare_dataset
from train import train
from inference_realtime import run_realtime
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
        elif escolha == "Avaliar modelo":
            avaliar_modelo()
        elif escolha == "Configurações":
            configurar()
        elif escolha == "Testar CUDA":
            testar_cuda()
        elif escolha == "Sair":
            print("Saindo do programa...")
            break
        
def testar_cuda():
    
    torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    print(f"Número de GPUs disponíveis: {num_gpus}")
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    print("Pressione Enter para continuar...")
    input()
    
        
def baixar_dataset():
    
    prepare_dataset.download_datasets()
    print("Pressione Enter para continuar...")
    input()

def treinar_modelo():
    
    train()
    print("Pressione Enter para continuar...")
    input()

def avaliar_modelo():
    run_realtime()
    print("Pressione Enter para continuar...")
    input()
    
    
def configurar():
    
    cli.menu_configuracoes()
    print("Pressione Enter para continuar...")
    input()

    
if __name__ == "__main__":
    main()