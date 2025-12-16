import torch
import cv2
import numpy as np
import mss
import time
from collections import deque
from models.model import ESPCN

# --- Configurações ---
CHECKPOINT_PATH = './checkpoints/best_model.pth'
SCALE_FACTOR = 2           # O modelo vai aumentar a imagem em 2x
NUM_FRAMES = 3             # Input do modelo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Área da tela para capturar (VSR em tela cheia é muito pesado para tempo real puro)
# Sugestão: Capture uma janela pequena (ex: player de vídeo) para testar primeiro
CAPTURE_BOX = {'top': 100, 'left': 100, 'width': 480, 'height': 270} 

def run_realtime():
    # 1. Carregar Modelo
    print(f"Carregando modelo ESPCN em {DEVICE}...")
    model = ESPCN(scale_factor=SCALE_FACTOR, num_frames=NUM_FRAMES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print("Checkpoint não encontrado! Rodando com pesos aleatórios para teste.")

    # 2. Configurar Captura e Buffer
    sct = mss.mss()
    frame_buffer = deque(maxlen=NUM_FRAMES)
    
    print("Iniciando VSR em Tempo Real. Pressione 'q' na janela de output para sair.")
    
    prev_time = 0
    
    with torch.no_grad(): # Desativa gradientes para velocidade
        while True:
            loop_start = time.time()
            
            # --- A. Captura de Tela ---
            screenshot = sct.grab(CAPTURE_BOX)
            img = np.array(screenshot) # BGRA
            
            # Conversão e Normalização
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img_normalized = img.astype(np.float32) / 255.0
            
            # Adiciona ao buffer
            frame_buffer.append(img_normalized)
            
            # Só roda inferência se o buffer estiver cheio
            if len(frame_buffer) == NUM_FRAMES:
                
                # --- B. Prepara Tensor ---
                # Transforma cada frame em (C, H, W) e empilha na dimensão de canais
                tensor_list = [torch.from_numpy(f).permute(2, 0, 1) for f in frame_buffer]
                
                # Input shape: (1, Channels*NumFrames, H, W)
                input_tensor = torch.cat(tensor_list, dim=0).unsqueeze(0).to(DEVICE)
                
                # --- C. Inferência ---
                output_tensor = model(input_tensor)
                
                # --- D. Pós-processamento ---
                output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_img = np.clip(output_img, 0, 1)
                output_img = (output_img * 255).astype(np.uint8)
                
                # Volta para BGR para exibir no OpenCV
                output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                
                # --- E. Exibição ---
                cv2.imshow('ESPCN Output (Super Resolution)', output_img)
                
                # Opcional: Mostrar input original para comparação
                # cv2.imshow('Input Original (Low Res)', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Cálculo de FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            print(f"FPS: {fps:.2f}", end='\r')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()