import torch
import cv2
import numpy as np
import mss
import time
import os
import glob
import questionary

from models import get_model, get_interface
from utils.config import ConfigManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolução LR fixa — captura nativa 540p → 1080p SR (x2)
LR_SIZE = (960, 540)  # (width, height)


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
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_type = checkpoint.get('model_type', model_type)
            ckpt_params = checkpoint.get('model_params')
            if not ckpt_params:
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
        return model, get_interface(model_type), scale, model_type

    except FileNotFoundError:
        print(f"Checkpoint não encontrado em '{checkpoint_path}'.")
        print("Rodando com pesos aleatórios para teste.")
    except Exception as e:
        print(f"Erro ao carregar checkpoint: {e}")
        print("Rodando com pesos aleatórios para teste.")

    model = get_model(model_type, scale_factor=scale, channels=3, **model_params)
    model.to(DEVICE)
    model.eval()
    return model, get_interface(model_type), scale, model_type


def _select_checkpoint():
    """Lista checkpoints disponíveis e permite o usuário escolher."""
    ckpt_dir = ConfigManager.get('checkpoint_dir', './checkpoints')

    if not os.path.exists(ckpt_dir):
        print("Pasta de checkpoints não encontrada.")
        return None

    checkpoints = [f for f in os.listdir(ckpt_dir)
                   if f.endswith('.pth') and '_best_model' in f]

    if not checkpoints:
        print("Nenhum checkpoint encontrado. Treine um modelo primeiro.")
        return None

    choices = []
    for ckpt_file in sorted(checkpoints):
        path = os.path.join(ckpt_dir, ckpt_file)
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            model_name = data.get('model_type', '?')
            epoch = data.get('epoch', '?')
            psnr = data.get('psnr', 0)
            inference = data.get('inference_ms', 0)
            inf_str = f" | {inference:.1f} ms" if inference else ""
            label = f"{ckpt_file}  ({model_name} | epoch {epoch} | {psnr:.2f} dB{inf_str})"
        except Exception:
            label = ckpt_file
        choices.append(questionary.Choice(title=label, value=ckpt_file))

    escolha = questionary.select(
        "Escolha um checkpoint para Super Resolução:",
        choices=choices
    ).ask()

    if escolha:
        return os.path.join(ckpt_dir, escolha)
    return None


def _select_source():
    """Permite o usuário escolher a fonte de vídeo."""
    choices = [
        questionary.Choice("Webcam", value="webcam"),
        questionary.Choice("Arquivo de vídeo", value="video"),
        questionary.Choice("Captura de tela", value="screen"),
    ]
    return questionary.select("Fonte de vídeo:", choices=choices).ask()


def _select_video_file():
    """Busca arquivos de vídeo e permite o usuário escolher."""
    video_exts = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv', '*.webm')
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(ext))
        video_files.extend(glob.glob(os.path.join('data', '**', ext), recursive=True))
        video_files.extend(glob.glob(os.path.join('videos', '**', ext), recursive=True))

    if not video_files:
        path = questionary.text(
            "Nenhum vídeo encontrado. Digite o caminho completo:"
        ).ask()
        return path

    choices = [questionary.Choice(title=os.path.basename(f), value=f) for f in sorted(set(video_files))]
    choices.append(questionary.Choice(title="Digitar caminho manualmente", value="__manual__"))

    escolha = questionary.select("Escolha um vídeo:", choices=choices).ask()
    if escolha == "__manual__":
        return questionary.text("Caminho do vídeo:").ask()
    return escolha


def _select_display_mode():
    """Permite o usuário escolher o modo de exibição."""
    choices = [
        questionary.Choice("Comparação lado a lado (Bicúbico vs SR)", value="side_by_side"),
        questionary.Choice("Apenas SR (tela cheia)", value="sr_only"),
        questionary.Choice("Triplo (Original | Bicúbico | SR)", value="triple"),
    ]
    return questionary.select("Modo de exibição:", choices=choices).ask()


def _build_frame_display(img_lr_bgr, output_bgr, display_mode, display_w, display_h):
    """Monta o frame de exibição conforme o modo selecionado."""

    if display_mode == "sr_only":
        frame = cv2.resize(output_bgr, (display_w, display_h),
                           interpolation=cv2.INTER_LINEAR)
        return frame

    h_sr, w_sr = output_bgr.shape[:2]
    bicubic = cv2.resize(img_lr_bgr, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)

    if display_mode == "side_by_side":
        cv2.putText(bicubic, "Bicubico", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(output_bgr, "Super Resolucao", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        combined = np.hstack([bicubic, output_bgr])
        # Redimensiona para caber na tela
        frame = cv2.resize(combined, (display_w, display_h // 2),
                           interpolation=cv2.INTER_LINEAR)
        # Mantém proporção: display_w para 2 painéis lado a lado
        panel_h = int(display_w * h_sr / (w_sr * 2))
        frame = cv2.resize(combined, (display_w, panel_h),
                           interpolation=cv2.INTER_LINEAR)
        return frame

    if display_mode == "triple":
        original = cv2.resize(img_lr_bgr, (w_sr, h_sr),
                              interpolation=cv2.INTER_NEAREST)
        cv2.putText(original, "Original (LR)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(bicubic, "Bicubico", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(output_bgr, "Super Resolucao", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        combined = np.hstack([original, bicubic, output_bgr])
        panel_h = int(display_w * h_sr / (w_sr * 3))
        frame = cv2.resize(combined, (display_w, panel_h),
                           interpolation=cv2.INTER_LINEAR)
        return frame

    return output_bgr


def run_realtime():
    # 1. Seleção de checkpoint
    full_path = _select_checkpoint()
    if full_path is None:
        return

    model, interface, scale, model_type = load_model(full_path)

    # 2. Seleção de fonte
    source = _select_source()
    cap = None
    sct = None

    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro: Webcam não encontrada.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        source_label = "Webcam"

    elif source == "video":
        video_path = _select_video_file()
        if not video_path:
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir '{video_path}'.")
            return
        source_label = os.path.basename(video_path)

    else:  # screen
        sct = mss.mss()
        monitors = sct.monitors
        if len(monitors) > 2:
            mon_choices = []
            for i, m in enumerate(monitors[1:], 1):
                mon_choices.append(questionary.Choice(
                    title=f"Monitor {i}: {m['width']}x{m['height']}",
                    value=i))
            mon_idx = questionary.select("Monitor:", choices=mon_choices).ask()
        else:
            mon_idx = 1
        monitor = monitors[mon_idx]
        capture_box = {
            'top': monitor['top'],
            'left': monitor['left'],
            'width': min(monitor['width'], 1920),
            'height': min(monitor['height'], 1080),
        }
        source_label = f"Tela {mon_idx} ({capture_box['width']}x{capture_box['height']})"

    # 3. Modo de exibição
    display_mode = _select_display_mode()

    # Resolução da janela de exibição — cabe na tela
    display_w = 1280
    display_h = 720

    out_w, out_h = LR_SIZE[0] * scale, LR_SIZE[1] * scale
    print(f"\n{'='*60}")
    print(f"  Modelo: {model_type} | Device: {DEVICE}")
    print(f"  Fonte: {source_label}")
    print(f"  Pipeline: entrada -> {LR_SIZE[0]}x{LR_SIZE[1]} (LR) -> "
          f"{out_w}x{out_h} (SR x{scale})")
    print(f"  Modo: {display_mode}")
    print(f"  Controles: 'q' = sair | 'm' = trocar modo | 'p' = pausar")
    print(f"{'='*60}\n")

    window_name = f'VSR - {model_type}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h)

    prev_time = time.time()
    state = None
    paused = False
    frame_count = 0
    fps_avg = 0.0

    with torch.no_grad():
        while True:
            if not paused:
                # --- A. Captura ---
                if cap is not None:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        if source == "video":
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                            continue
                        break
                    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                else:
                    screenshot = sct.grab(capture_box)
                    img_bgra = np.array(screenshot)
                    img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)

                # --- B. Downscale para LR ---
                img_lr = cv2.resize(img_rgb, LR_SIZE, interpolation=cv2.INTER_CUBIC)
                img_norm = img_lr.astype(np.float32) / 255.0

                # --- C. Tensor ---
                input_tensor = (torch.from_numpy(img_norm)
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                                .to(DEVICE))

                # --- D. Inferência ---
                if interface == "recurrent":
                    output_tensor, state = model(input_tensor, state)
                else:
                    output_tensor = model(input_tensor.unsqueeze(1))

                # --- E. Pós-processamento ---
                output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_img = np.clip(output_img, 0, 1)
                output_img = (output_img * 255).astype(np.uint8)
                output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                img_lr_bgr = cv2.cvtColor(img_lr, cv2.COLOR_RGB2BGR)

                # --- F. FPS ---
                curr_time = time.time()
                dt = max(curr_time - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = curr_time
                frame_count += 1
                fps_avg = fps_avg * 0.9 + fps * 0.1 if frame_count > 1 else fps

                # --- G. Montar exibição ---
                display = _build_frame_display(
                    img_lr_bgr, output_bgr, display_mode, display_w, display_h)

                # HUD no canto inferior
                h_disp = display.shape[0]
                info = (f"FPS: {fps_avg:.1f} | {dt*1000:.0f}ms | "
                        f"{model_type} x{scale} | {DEVICE}")
                cv2.putText(display, info, (10, h_disp - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # --- H. Exibição ---
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                modes = ["side_by_side", "sr_only", "triple"]
                idx = (modes.index(display_mode) + 1) % len(modes)
                display_mode = modes[idx]
                print(f"Modo: {display_mode}")
            elif key == ord('p'):
                paused = not paused
                print("Pausado" if paused else "Retomado")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
