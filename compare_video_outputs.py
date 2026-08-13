import sys
import os
import glob
import subprocess
import torch
import cv2
import numpy as np
import questionary

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_SIZE = (960, 540)


def imread_unicode(path):
    """Lê arquivos de imagem sem falhar quando o caminho possui acentos no Windows."""
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_video_file():
    """Busca vídeos na raiz do projeto ou aceita argumento via terminal."""
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    video_exts = ('*.mp4', '*.avi', '*.mkv')
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(ext))
        video_files.extend(glob.glob(os.path.join('data', '**', ext), recursive=True))

    video_files = sorted(set(video_files))

    if not video_files:
        print("[Erro] Nenhum arquivo de vídeo encontrado na raiz do projeto.")
        return None

    try:
        choices = [questionary.Choice(title=f, value=f) for f in video_files]
        return questionary.select("Escolha o vídeo para validação:", choices=choices).ask()
    except Exception:
        print(f"[Info] Selecionando automaticamente: {video_files[0]}")
        return video_files[0]


def run_pytorch_stream(video_path, checkpoint_path, max_frames=30):
    """Executa a inferência no PyTorch."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_type = checkpoint.get('model_type', 'RTDVSR')
    model_params = checkpoint.get('model_params', {'hidden_dim': 64, 'num_res_blocks': 6})
    scale = checkpoint.get('config', {}).get('scale_factor', 2)

    from models import get_model
    model = get_model(model_type, scale_factor=scale, channels=3, **model_params)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames_pytorch = []
    state = None

    print(f"\n[PyTorch] Processando {max_frames} frames no dispositivo {DEVICE}...")

    with torch.no_grad():
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_lr = cv2.resize(img_rgb, LR_SIZE, interpolation=cv2.INTER_CUBIC)
            img_norm = img_lr.astype(np.float32) / 255.0

            input_tensor = (torch.from_numpy(img_norm)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .to(DEVICE))

            output_tensor, state = model(input_tensor, state)

            output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_img = np.clip(output_img, 0, 1)
            output_img = (output_img * 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

            frames_pytorch.append(output_bgr)
            count += 1

    cap.release()
    return frames_pytorch


def run_cpp_validation(video_path, max_frames=30):
    """Executa o executável C++ usando pasta relativa para evitar problemas com acentos."""
    cpp_dir = os.path.abspath("inferencecpp/build/Debug")
    cpp_exe = os.path.join(cpp_dir, "RTDVSR_CPP.exe")

    if not os.path.exists(cpp_exe):
        print(f"[Erro] Executável não encontrado em: {cpp_exe}")
        return None

    output_dir = os.path.join(cpp_dir, "output_frames")
    os.makedirs(output_dir, exist_ok=True)

    abs_video_path = os.path.abspath(video_path)

    print("[C++] Executando inferência no TensorRT...")

    cmd = [cpp_exe, abs_video_path, "output_frames", str(max_frames)]
    subprocess.run(cmd, cwd=cpp_dir)

    cpp_frames = []
    for i in range(max_frames):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        if os.path.exists(frame_path):
            img = imread_unicode(frame_path)
            if img is not None:
                cpp_frames.append(img)

    return cpp_frames


def compare_results(py_frames, cpp_frames):
    """Calcula estatísticas de fidelidade matemática."""
    if not cpp_frames or len(py_frames) != len(cpp_frames):
        print("\n[Erro] Não foi possível carregar os frames gerados pelo C++.")
        print(f"Frames PyTorch: {len(py_frames)} | Frames C++: {len(cpp_frames) if cpp_frames else 0}")
        return

    print("\n" + "=" * 60)
    print("      MÉTRICAS DE COMPARABILIDADE: PYTORCH vs C++ (TensorRT)")
    print("=" * 60)

    psnr_list = []
    mse_list = []
    max_diff_list = []

    for f_py, f_cpp in zip(py_frames, cpp_frames):
        diff = np.abs(f_py.astype(np.float32) - f_cpp.astype(np.float32))
        mse = np.mean(diff ** 2)
        max_diff = np.max(diff)

        psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

        psnr_list.append(psnr)
        mse_list.append(mse)
        max_diff_list.append(max_diff)

    avg_psnr = np.mean([p for p in psnr_list if p != float('inf')])
    avg_mse = np.mean(mse_list)
    avg_max_diff = np.mean(max_diff_list)

    print(f" Frames Avaliados:            {len(py_frames)}")
    print(f" Desvio Máximo Médio/Pixel:   {avg_max_diff:.2f} / 255")
    print(f" Erro Quadrático Médio (MSE):  {avg_mse:.6f}")
    print(f" PSNR Médio entre Motores:    {avg_psnr:.2f} dB")
    print("=" * 60)


def main():
    video_path = get_video_file()
    if not video_path:
        return

    ckpt_path = "checkpoints/RTDVSR_best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"[Erro] Checkpoint não encontrado em: {ckpt_path}")
        return

    num_frames_test = 30
    py_frames = run_pytorch_stream(video_path, ckpt_path, max_frames=num_frames_test)
    cpp_frames = run_cpp_validation(video_path, max_frames=num_frames_test)

    if cpp_frames:
        compare_results(py_frames, cpp_frames)


if __name__ == "__main__":
    main()