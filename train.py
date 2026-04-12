import torch
import torch.nn as nn
import warnings
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import os
import io
import math
import time
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from models import get_model, get_interface
from utils.config import ConfigManager

def _resolve_vimeo_paths(data_root):
    """
    Retorna (sequences_dir, train_list, test_list) para o Vimeo Septuplet,
    aceitando tanto o layout direto quanto o aninhado que o zip oficial
    produz (datasets/vimeo_septuplet/vimeo_septuplet/...).
    """
    candidates = [
        os.path.join(data_root, 'vimeo_septuplet'),
        os.path.join(data_root, 'vimeo_septuplet', 'vimeo_septuplet'),
    ]
    for base in candidates:
        seqs = os.path.join(base, 'sequences')
        train_list = os.path.join(base, 'sep_trainlist.txt')
        if os.path.isdir(seqs) and os.path.isfile(train_list):
            test_list = os.path.join(base, 'sep_testlist.txt')
            return seqs, train_list, test_list
    # Nenhum válido: devolve o caminho canônico para log de erro.
    base = candidates[0]
    return (os.path.join(base, 'sequences'),
            os.path.join(base, 'sep_trainlist.txt'),
            os.path.join(base, 'sep_testlist.txt'))



def _apply_realistic_degradation(lr_img, train=True):
    """
    Aplica degradação realista ao frame LR para simular condições reais.
    Pipeline: bicubic downscale (já feito) → compressão JPEG → ruído gaussiano.

    Baseado em Real-ESRGAN (Wang et al., 2021) — degradação de segunda ordem
    simplificada para manter o treinamento viável em hardware limitado.
    """
    # 1. Compressão JPEG — simula artefatos de streaming/codec
    if train:
        quality = random.randint(30, 85)
    else:
        quality = 50  # fixo para validação — típico de streaming real

    buf = io.BytesIO()
    lr_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    lr_img = Image.open(buf).convert('RGB')

    # 2. Converte para tensor e adiciona ruído gaussiano
    lr_tensor = TF.to_tensor(lr_img)

    if train:
        sigma = random.uniform(3.0, 15.0) / 255.0
    else:
        sigma = 8.0 / 255.0  # fixo para validação — ruído de sensor típico

    noise = torch.randn_like(lr_tensor) * sigma
    lr_tensor = torch.clamp(lr_tensor + noise, 0.0, 1.0)

    return lr_tensor


def _benchmark_inference(model, device, scale, interface):
    """
    Mede o tempo médio de inferência em resolução real (960x540 LR → 1920x1080 SR).
    Pipeline: 720p captura → downscale 540p → SR 2x → 1080p.
    Retorna o tempo em milissegundos.
    """
    model.eval()
    lr_h, lr_w = 540, 960  # resolução LR após downscale de 720p
    dummy_frame = torch.randn(1, 3, lr_h, lr_w, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            if interface == "recurrent":
                model(dummy_frame, None)
            else:
                model(dummy_frame.unsqueeze(1))

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Medição
    times = []
    with torch.no_grad():
        for _ in range(20):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if interface == "recurrent":
                model(dummy_frame, None)
            else:
                model(dummy_frame.unsqueeze(1))

            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


class VimeoSeptupletDataset(Dataset):
    def __init__(self, data_root, list_file, scale_factor=2, crop_size=96,
                 seq_len=3, train=True):
        """
        Retorna subsequências de `seq_len` frames como pares (LR_seq, HR_seq).

        data_root: Caminho para a pasta sequences do vimeo_septuplet
        list_file: Caminho para sep_trainlist.txt ou sep_testlist.txt
        seq_len: Quantidade de frames consecutivos por amostra (max 7)
        """
        self.data_root = data_root
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.seq_len = min(seq_len, 7)
        self.train = train

        with open(list_file, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        try:
            seq_path = os.path.join(self.data_root, self.sequences[idx])

            max_start = 8 - self.seq_len
            if self.train:
                start = random.randint(1, max_start)
            else:
                start = max(1, (max_start + 1) // 2)

            frame_indices = list(range(start, start + self.seq_len))

            imgs = []
            for fi in frame_indices:
                img_path = os.path.join(seq_path, f'im{fi}.png')
                img = Image.open(img_path).convert('RGB')
                imgs.append(img)

            if imgs[0].width < self.crop_size or imgs[0].height < self.crop_size:
                imgs = [im.resize((self.crop_size, self.crop_size), Image.BICUBIC)
                        for im in imgs]

            w, h = imgs[0].size
            if self.train:
                left = random.randint(0, w - self.crop_size)
                top = random.randint(0, h - self.crop_size)
            else:
                left = (w - self.crop_size) // 2
                top = (h - self.crop_size) // 2

            imgs = [im.crop((left, top, left + self.crop_size, top + self.crop_size))
                    for im in imgs]

            if self.train:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                rot = random.choice([0, 90, 180, 270])
                if hflip:
                    imgs = [TF.hflip(im) for im in imgs]
                if vflip:
                    imgs = [TF.vflip(im) for im in imgs]
                if rot > 0:
                    imgs = [TF.rotate(im, rot) for im in imgs]

            lr_size = self.crop_size // self.scale_factor
            lr_tensors = []
            hr_tensors = []
            for im in imgs:
                lr_img = im.resize((lr_size, lr_size), Image.BICUBIC)
                lr_tensors.append(_apply_realistic_degradation(lr_img, train=self.train))
                hr_tensors.append(TF.to_tensor(im))

            return torch.stack(lr_tensors), torch.stack(hr_tensors)

        except Exception as e:
            warnings.warn(f"Erro ao carregar {self.sequences[idx]}: {e}. Redirecionando.")
            return self.__getitem__(random.randint(0, len(self) - 1))

class SRDataset(Dataset):
    def __init__(self, image_files, scale_factor=2, crop_size=96, train=True):
        self.image_files = image_files
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_files[idx]).convert('RGB')

            if img.width < self.crop_size or img.height < self.crop_size:
                img = img.resize((self.crop_size, self.crop_size), Image.BICUBIC)

            w, h = img.size
            if self.train:
                left = random.randint(0, w - self.crop_size)
                top = random.randint(0, h - self.crop_size)
                img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
                if random.random() < 0.5:
                    img = TF.hflip(img)
                if random.random() < 0.5:
                    img = TF.vflip(img)
                rot = random.choice([0, 90, 180, 270])
                if rot > 0:
                    img = TF.rotate(img, rot)
            else:
                left = (w - self.crop_size) // 2
                top = (h - self.crop_size) // 2
                img = img.crop((left, top, left + self.crop_size, top + self.crop_size))

            lr_size = self.crop_size // self.scale_factor
            lr_img = img.resize((lr_size, lr_size), Image.BICUBIC)

            lr_tensor = _apply_realistic_degradation(lr_img, train=self.train)
            hr_tensor = TF.to_tensor(img)

            return lr_tensor.unsqueeze(0), hr_tensor.unsqueeze(0)

        except Exception as e:
            print(f"Erro ao carregar imagem {self.image_files[idx]}: {e}")
            lr_size = self.crop_size // self.scale_factor
            return (torch.zeros(1, 3, lr_size, lr_size),
                    torch.zeros(1, 3, self.crop_size, self.crop_size))

def calc_psnr(img1, img2):
    img1 = torch.clamp(img1, 0., 1.)
    img2 = torch.clamp(img2, 0., 1.)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(1. / mse.item())


def calc_temporal_consistency(sr_frames, hr_frames):
    """
    Temporal Consistency Error (TCE).
    Compara as diferenças temporais entre frames SR consecutivos com as
    diferenças entre frames HR consecutivos. Quanto menor, mais consistente.

    sr_frames: lista de tensores SR (B, C, H, W)
    hr_frames: lista de tensores HR (B, C, H, W)
    Retorna: TCE médio (float) — 0.0 = perfeitamente consistente
    """
    if len(sr_frames) < 2:
        return 0.0

    tce_sum = 0.0
    count = 0
    for i in range(1, len(sr_frames)):
        sr_diff = sr_frames[i] - sr_frames[i - 1]
        hr_diff = hr_frames[i] - hr_frames[i - 1]
        tce_sum += torch.mean((sr_diff - hr_diff) ** 2).item()
        count += 1

    return tce_sum / count


def _detach_state(state):
    """Desanexa o estado recorrente do grafo computacional.
    Suporta tensor único, tupla/lista de tensores, ou None."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.float().detach()
    if isinstance(state, (tuple, list)):
        return type(state)(s.float().detach() for s in state)
    return state


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (L1 suavizado) — mais robusto que MSE para SR."""
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps_sq = epsilon ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps_sq))

def train():
    config = ConfigManager.get_instance()
    if not config.get_config():
        try:
            ConfigManager.load_config('presets/config.json')
        except Exception as e:
            print(f"Aviso: {e}. Criando config padrão.")
            ConfigManager.new_config({
                "dataset_path": "./datasets",
                "scale_factor": 2,
                "learning_rate": 0.001,
                "batch_size": 8,
                "epochs": 50,
                "crop_size": 96,
                "seq_len": 3,
                "model_type": "LightweightVSR",
                "model_params": {"hidden_dim": 64, "num_res_blocks": 6},
            })

    config = ConfigManager.get_instance()

    lr = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 100)
    scale = config.get('scale_factor', 2)
    crop_size = config.get('crop_size', 96)
    seq_len = config.get('seq_len', 3)
    data_path = config.get('dataset_path', './datasets')
    ckpt_dir = config.get('checkpoint_dir', './checkpoints')
    model_type = config.get('model_type', 'LightweightVSR')
    model_params = config.get('model_params', {'hidden_dim': 64, 'num_res_blocks': 6})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    print(f"Device: {device} | Modelo: {model_type} | Scale: x{scale} | "
          f"Batch: {batch_size} | seq_len: {seq_len}")
    print(f"Parâmetros do modelo: {model_params}")

    vimeo_sequences_path, vimeo_train_list, vimeo_test_list = _resolve_vimeo_paths(data_path)

    if os.path.exists(vimeo_sequences_path) and os.path.exists(vimeo_train_list):
        print(f"Usando dataset Vimeo Septuplet (VSR temporal) em: {vimeo_sequences_path}")
        train_ds = VimeoSeptupletDataset(
            vimeo_sequences_path, vimeo_train_list,
            scale_factor=scale, crop_size=crop_size, seq_len=seq_len, train=True)

        if os.path.exists(vimeo_test_list):
            val_ds = VimeoSeptupletDataset(
                vimeo_sequences_path, vimeo_test_list,
                scale_factor=scale, crop_size=crop_size, seq_len=seq_len, train=False)
        else:
            print("Aviso: sep_testlist.txt não encontrado. Usando 10% do treino.")
            with open(vimeo_train_list, 'r') as f:
                all_seqs = [line.strip() for line in f.readlines()]
            train_seqs, val_seqs = train_test_split(all_seqs, test_size=0.1,
                                                     random_state=42)
            temp_train = os.path.join(data_path, 'vimeo_septuplet', 'temp_train.txt')
            temp_val = os.path.join(data_path, 'vimeo_septuplet', 'temp_val.txt')
            with open(temp_train, 'w') as f:
                f.write('\n'.join(train_seqs))
            with open(temp_val, 'w') as f:
                f.write('\n'.join(val_seqs))
            train_ds = VimeoSeptupletDataset(
                vimeo_sequences_path, temp_train,
                scale_factor=scale, crop_size=crop_size, seq_len=seq_len, train=True)
            val_ds = VimeoSeptupletDataset(
                vimeo_sequences_path, temp_val,
                scale_factor=scale, crop_size=crop_size, seq_len=seq_len, train=False)
    else:
        print("[AVISO] Vimeo Septuplet não encontrado. Caminhos inspecionados:")
        print(f"  - sequences: {vimeo_sequences_path}")
        print(f"  - trainlist: {vimeo_train_list}")
        print("Caindo para dataset genérico de imagens (modo SISR, sem temporalidade).")
        path_obj = Path(data_path)
        all_images = (list(path_obj.rglob("*.png")) +
                      list(path_obj.rglob("*.jpg")) +
                      list(path_obj.rglob("*.jpeg")))

        if len(all_images) == 0:
            print("Nenhuma imagem encontrada. Verifique o caminho no config.json")
            return

        train_files, val_files = train_test_split(all_images, test_size=0.1,
                                                   random_state=42)
        train_ds = SRDataset(train_files, scale_factor=scale,
                             crop_size=crop_size, train=True)
        val_ds = SRDataset(val_files, scale_factor=scale,
                           crop_size=crop_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=is_cuda,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2,
                            persistent_workers=True)
    max_val_samples = config.get('max_val_samples', 500)

    model = get_model(model_type, scale_factor=scale, channels=3, **model_params).to(device)
    interface = get_interface(model_type)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros do modelo: {num_params:,} | Interface: {interface}")

    criterion = CharbonnierLoss()
    temporal_weight = config.get('temporal_loss_weight', 0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup linear de 1 época + cosine annealing — evita explosão de gradientes
    warmup_epochs = 1
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs])
    
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = f"{model_type}_best_model.pth"
    full_ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    # Early stopping: para quando PSNR >= 27 dB e inferência <= 16 ms,
    # ou quando PSNR não melhora por `patience` épocas consecutivas.
    target_psnr = config.get('target_psnr', 27.0)
    target_inference_ms = config.get('target_inference_ms', 16.0)
    patience = config.get('early_stopping_patience', 10)
    epochs_without_improvement = 0

    start_epoch = 0
    best_psnr = 0.0

    if os.path.exists(full_ckpt_path):
        print(f"Carregando checkpoint: {full_ckpt_path}")
        # weights_only=False adicionado para limpar o aviso de segurança do PyTorch 2.4+
        checkpoint = torch.load(full_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scaler' in checkpoint and checkpoint['scaler'] is not None and is_cuda:
            scaler.load_state_dict(checkpoint['scaler'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('psnr', 0.0)
        print(f"Resumindo da época {start_epoch} com PSNR base: {best_psnr:.2f} dB")
    else:
        print(f"Nenhum checkpoint encontrado para '{model_type}' ({ckpt_name}). Iniciando treino do zero.")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}',
                  unit='batch') as pbar:
            for lr_seq, hr_seq in train_loader:
                lr_seq = lr_seq.to(device, non_blocking=True)
                hr_seq = hr_seq.to(device, non_blocking=True)
                T = lr_seq.shape[1]

                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    if interface == "recurrent":
                        recon_loss = torch.zeros(1, device=device, dtype=torch.float32)
                        temp_loss = torch.zeros(1, device=device, dtype=torch.float32)
                        state = None
                        prev_sr = None
                        prev_hr = None
                        for t in range(T):
                            state = _detach_state(state)

                            sr_frame, state = model(lr_seq[:, t], state)
                            recon_loss = recon_loss + criterion(sr_frame, hr_seq[:, t])

                            # Temporal consistency loss
                            if prev_sr is not None:
                                sr_diff = sr_frame - prev_sr
                                hr_diff = hr_seq[:, t] - prev_hr
                                temp_loss = temp_loss + torch.mean((sr_diff - hr_diff) ** 2)

                            prev_sr = sr_frame.detach()
                            prev_hr = hr_seq[:, t]

                        recon_loss = recon_loss / T
                        temp_loss = temp_loss / max(T - 1, 1)
                        total_loss = recon_loss + temporal_weight * temp_loss
                    else:
                        sr_frame = model(lr_seq)
                        total_loss = criterion(sr_frame, hr_seq[:, T // 2])
                        
                # Proteção contra NaN — pula o batch se loss explodir
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    optimizer.zero_grad(set_to_none=True)
                    pbar.set_postfix({'loss': 'NaN (skip)'})
                    pbar.update(1)
                    continue

                if is_cuda:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                # Se pesos ficaram NaN após o step, restaura do checkpoint
                has_nan = any(torch.isnan(p).any() for p in model.parameters())
                if has_nan:
                    if os.path.exists(full_ckpt_path):
                        print("\n[AVISO] NaN detectado nos pesos! Restaurando último checkpoint...")
                        ckpt = torch.load(full_ckpt_path, map_location=device, weights_only=False)
                        model.load_state_dict(ckpt['model'])
                        optimizer.load_state_dict(ckpt['optimizer'])
                    else:
                        print("\n[AVISO] NaN detectado nos pesos sem checkpoint para restaurar!")
                    pbar.set_postfix({'loss': 'NaN (restored)'})
                    pbar.update(1)
                    continue

                epoch_loss += total_loss.item()
                pbar.set_postfix({'loss': f'{total_loss.item():.5f}'})
                pbar.update(1)


        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        val_psnr = 0.0
        val_tce = 0.0
        val_count = 0
        val_seq_count = 0
        with torch.no_grad():
            for val_i, (lr_seq, hr_seq) in enumerate(val_loader):
                if val_i >= max_val_samples:
                    break
                lr_seq = lr_seq.to(device)
                hr_seq = hr_seq.to(device)
                T = lr_seq.shape[1]

                if interface == "recurrent":
                    state = None
                    sr_frames = []
                    hr_frames = []
                    for t in range(T):
                        sr_frame, state = model(lr_seq[:, t], state)
                        val_psnr += calc_psnr(sr_frame, hr_seq[:, t])
                        val_count += 1
                        sr_frames.append(sr_frame)
                        hr_frames.append(hr_seq[:, t])
                    val_tce += calc_temporal_consistency(sr_frames, hr_frames)
                    val_seq_count += 1
                else:  # sliding_window
                    sr_frame = model(lr_seq)
                    val_psnr += calc_psnr(sr_frame, hr_seq[:, T // 2])
                    val_count += 1

        avg_psnr = val_psnr / max(val_count, 1)
        avg_tce = val_tce / max(val_seq_count, 1)
        scheduler.step()

        # Benchmark de inferência a cada 5 épocas ou na primeira
        if epoch == start_epoch or (epoch + 1) % 5 == 0:
            inference_ms = _benchmark_inference(model, device, scale, interface)
        print(f"Epoch {epoch+1} -> Train Loss: {avg_loss:.6f} | "
              f"Val PSNR: {avg_psnr:.2f} dB | TCE: {avg_tce:.6f} | "
              f"Inferência: {inference_ms:.1f} ms")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if is_cuda else None,
                'psnr': best_psnr,
                'tce': avg_tce,
                'inference_ms': inference_ms,
                'model_type': model_type,
                'model_params': model_params,
                'config': config.get_config()
            }, full_ckpt_path)
            print(f"Salvo novo melhor modelo! ({best_psnr:.2f} dB | TCE: {avg_tce:.6f})\n")
        else:
            epochs_without_improvement += 1

        # Early stopping: modelo atingiu os critérios de qualidade + velocidade
        if avg_psnr >= target_psnr and inference_ms <= target_inference_ms:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING: Modelo atingiu os critérios!")
            print(f"  PSNR: {avg_psnr:.2f} dB >= {target_psnr} dB")
            print(f"  Inferência: {inference_ms:.1f} ms <= {target_inference_ms} ms")
            print(f"{'='*60}\n")
            break

        # Early stopping: PSNR estagnado
        if epochs_without_improvement >= patience:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING: PSNR não melhorou por {patience} épocas.")
            print(f"  Melhor PSNR: {best_psnr:.2f} dB")
            if best_psnr < target_psnr:
                print(f"  (Não atingiu o alvo de {target_psnr} dB)")
            print(f"{'='*60}\n")
            break


if __name__ == '__main__':
    train()
