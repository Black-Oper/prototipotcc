import torch
import cv2
import numpy as np
import mss
import time

from models import get_model, get_interface
from utils.config import ConfigManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAPTURE_BOX = {'top': 100, 'left': 100, 'width': 480, 'height': 270}


def load_model(checkpoint_path: str):
    """Carrega modelo do checkpoint usando o registry."""
    try:
        ConfigManager.load_config('presets/config.json')
    except Exception:
        pass

    config = ConfigManager.get_instance()
    scale = config.get('scale_factor', 2)
    model_type = config.get('model_type', 'LightweightVSR')
    model_params = config.get('model_params', {'hidden_dim': 64, 'num_res_blocks': 6})

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_type = checkpoint.get('model_type', model_type)
            ckpt_params = checkpoint.get('model_params')
            if not ckpt_params:
                # backward compat: formato antigo
                ckpt_params = {}
                if 'hidden_dim' in checkpoint:
                    ckpt_params['hidden_dim'] = checkpoint['hidden_dim']
                if 'num_res_blocks' in checkpoint:
                    ckpt_params['num_res_blocks'] = checkpoint['num_res_blocks']
            if ckpt_params:
                model_params = ckpt_params

            saved_config = checkpoint.get('config', {})
            scale = saved_config.get('scale_factor', scale)
            state_dict = checkpoint['model']
            print(f"Checkpoint: epoch {checkpoint.get('epoch', '?')} | "
                  f"PSNR {checkpoint.get('psnr', 0):.2f} dB | "
                  f"modelo={model_type} | params={model_params}")
        else:
            state_dict = checkpoint

        model = get_model(model_type, scale_factor=scale, channels=3, **model_params)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"Modelo '{model_type}' pronto em {DEVICE}.")
        return model, get_interface(model_type), scale

    except FileNotFoundError:
        print(f"Checkpoint não encontrado em '{checkpoint_path}'.")
        print("Rodando com pesos aleatórios para teste.")
    except Exception as e:
        print(f"Erro ao carregar checkpoint: {e}")
        print("Rodando com pesos aleatórios para teste.")

    model = get_model(model_type, scale_factor=scale, channels=3, **model_params)
    model.to(DEVICE)
    model.eval()
    return model, get_interface(model_type), scale


def run_realtime():
    checkpoint_path = ConfigManager.get('checkpoint_dir', './checkpoints')
    checkpoint_name = ConfigManager.get('checkpoint_name', 'best_model.pth')
    full_path = f"{checkpoint_path}/{checkpoint_name}"

    model, interface, scale = load_model(full_path)
    sct = mss.mss()

    print("Iniciando Super Resolução em Tempo Real.")
    print(f"  Captura: {CAPTURE_BOX['width']}x{CAPTURE_BOX['height']} -> "
          f"{CAPTURE_BOX['width'] * scale}x{CAPTURE_BOX['height'] * scale}")
    print("  'q' = sair | 'c' = comparação lado a lado")

    show_comparison = False
    prev_time = time.time()
    state = None  # estado recorrente (usado apenas para interface "recurrent")

    with torch.no_grad():
        while True:
            # --- A. Captura ---
            screenshot = sct.grab(CAPTURE_BOX)
            img_bgra = np.array(screenshot)
            img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0

            # --- B. Tensor: (1, C, H, W) ---
            input_tensor = (torch.from_numpy(img_norm)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .to(DEVICE))

            # --- C. Inferência ---
            if interface == "recurrent":
                output_tensor, state = model(input_tensor, state)
            else:
                # sliding_window espera (B, T, C, H, W); envia janela de 1 frame
                output_tensor = model(input_tensor.unsqueeze(1))

            # --- D. Pós-processamento ---
            output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_img = np.clip(output_img, 0, 1)
            output_img = (output_img * 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            # --- E. FPS overlay ---
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            cv2.putText(output_bgr, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_bgr, f"x{scale} SR ({DEVICE})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            # --- F. Exibição ---
            if show_comparison:
                h_sr, w_sr = output_bgr.shape[:2]
                input_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                bicubic = cv2.resize(input_bgr, (w_sr, h_sr),
                                     interpolation=cv2.INTER_CUBIC)
                cv2.putText(bicubic, "Bicubic", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                combined = np.hstack([bicubic, output_bgr])
                cv2.imshow('VSR - Bicubic vs Super Resolucao', combined)
            else:
                cv2.imshow('Super Resolucao em Tempo Real', output_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_comparison = not show_comparison
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
