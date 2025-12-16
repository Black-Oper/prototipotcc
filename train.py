import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import os
import math
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Útil para dividir a lista de arquivos

# Importações locais (assumindo a estrutura de pastas correta)
from models.model import ESPCN
from utils.config import ConfigManager

# --- Dataset Class for Vimeo Septuplet ---
class VimeoSeptupletDataset(Dataset):
    def __init__(self, data_root, list_file, scale_factor=2, crop_size=96, train=True):
        """
        data_root: Caminho para a pasta sequences do vimeo_septuplet
        list_file: Caminho para sep_trainlist.txt ou sep_testlist.txt
        """
        self.data_root = data_root
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.train = train
        
        # Ler a lista de sequências
        with open(list_file, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        try:
            seq_path = os.path.join(self.data_root, self.sequences[idx])
            
            # Escolher um frame aleatório da sequência (im1 a im7) para treino
            # Para validação, usar sempre o frame central (im4)
            if self.train:
                frame_num = random.randint(1, 7)
            else:
                frame_num = 4  # Frame central
            
            img_path = os.path.join(seq_path, f'im{frame_num}.png')
            img = Image.open(img_path).convert('RGB')
            
            # Validação de tamanho mínimo
            if img.width < self.crop_size or img.height < self.crop_size:
                img = img.resize((self.crop_size, self.crop_size), Image.BICUBIC)

            # 1. Crop
            w, h = img.size
            if self.train:
                # Random Crop para Treino
                left = random.randint(0, w - self.crop_size)
                top = random.randint(0, h - self.crop_size)
                img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
                
                # 2. Augmentation (Apenas Treino)
                if random.random() < 0.5: img = TF.hflip(img)
                if random.random() < 0.5: img = TF.vflip(img)
                rot = random.choice([0, 90, 180, 270])
                if rot > 0: img = TF.rotate(img, rot)
            else:
                # Center Crop para Validação
                left = (w - self.crop_size) // 2
                top = (h - self.crop_size) // 2
                img = img.crop((left, top, left + self.crop_size, top + self.crop_size))

            # 3. Preparar LR (Input) e HR (Target)
            lr_size = self.crop_size // self.scale_factor
            
            # Downscale para criar a entrada LR artificialmente
            lr_img = img.resize((lr_size, lr_size), Image.BICUBIC)
            
            # Transformar em Tensor e Normalizar [0, 1]
            lr_tensor = TF.to_tensor(lr_img)
            hr_tensor = TF.to_tensor(img)

            return lr_tensor, hr_tensor
            
        except Exception as e:
            print(f"Erro ao carregar sequência {self.sequences[idx]}: {e}")
            # Retorna um tensor zerado em caso de erro
            return torch.zeros(3, self.crop_size//self.scale_factor, self.crop_size//self.scale_factor), torch.zeros(3, self.crop_size, self.crop_size)

# --- Dataset Class ---
class SRDataset(Dataset):
    def __init__(self, image_files, scale_factor=2, crop_size=96, train=True):
        """
        image_files: Lista de caminhos para as imagens
        """
        self.image_files = image_files
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.train = train

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_files[idx]).convert('RGB')
            
            # Validação de tamanho mínimo
            if img.width < self.crop_size or img.height < self.crop_size:
                # Se a imagem for muito pequena, redimensiona para poder cropar
                img = img.resize((self.crop_size, self.crop_size), Image.BICUBIC)

            # 1. Crop
            w, h = img.size
            if self.train:
                # Random Crop para Treino
                left = random.randint(0, w - self.crop_size)
                top = random.randint(0, h - self.crop_size)
                img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
                
                # 2. Augmentation (Apenas Treino)
                if random.random() < 0.5: img = TF.hflip(img)
                if random.random() < 0.5: img = TF.vflip(img)
                rot = random.choice([0, 90, 180, 270])
                if rot > 0: img = TF.rotate(img, rot)
            else:
                # Center Crop para Validação (Consistência de métrica)
                # Ou usar a imagem full se a GPU aguentar, mas Center Crop é padrão para validar training loops
                left = (w - self.crop_size) // 2
                top = (h - self.crop_size) // 2
                img = img.crop((left, top, left + self.crop_size, top + self.crop_size))

            # 3. Preparar LR (Input) e HR (Target)
            lr_size = self.crop_size // self.scale_factor
            
            # Downscale para criar a entrada LR artificialmente
            lr_img = img.resize((lr_size, lr_size), Image.BICUBIC)
            
            # Transformar em Tensor e Normalizar [0, 1]
            lr_tensor = TF.to_tensor(lr_img)
            hr_tensor = TF.to_tensor(img)

            return lr_tensor, hr_tensor
            
        except Exception as e:
            print(f"Erro ao carregar imagem {self.image_files[idx]}: {e}")
            # Retorna um tensor zerado em caso de erro para não quebrar o batch
            return torch.zeros(3, self.crop_size//self.scale_factor, self.crop_size//self.scale_factor), torch.zeros(3, self.crop_size, self.crop_size)

# --- Utils ---
def calc_psnr(img1, img2):
    # Clampar valores para garantir intervalo válido
    img1 = torch.clamp(img1, 0., 1.)
    img2 = torch.clamp(img2, 0., 1.)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 10 * math.log10(1. / mse.item())

# --- Training Loop ---
def train():
    # 1. Carregar Configuração
    try:
        ConfigManager.load_config('presets/config.json')
    except Exception as e:
        print(f"Aviso: {e}. Criando config padrão.")
        ConfigManager.new_config({
            "dataset_path": "./dataset", 
            "scale_factor": 2,
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 50,
            "crop_size": 96
        })

    config = ConfigManager.get_instance()
    
    # Parâmetros
    lr = config.get('learning_rate', 1e-3)
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 100)
    scale = config.get('scale_factor', 2)
    crop_size = config.get('crop_size', 96)
    data_path = config.get('dataset_path', './dataset')
    ckpt_dir = config.get('checkpoint_dir', './checkpoints')
    ckpt_name = config.get('checkpoint_name', 'best_model.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Scale: x{scale} | Batch: {batch_size}")    # 2. Preparar Dados
    # Verificar se é dataset vimeo_septuplet
    vimeo_sequences_path = os.path.join(data_path, 'vimeo_septuplet', 'sequences')
    vimeo_train_list = os.path.join(data_path, 'vimeo_septuplet', 'sep_trainlist.txt')
    vimeo_test_list = os.path.join(data_path, 'vimeo_septuplet', 'sep_testlist.txt')
    
    if os.path.exists(vimeo_sequences_path) and os.path.exists(vimeo_train_list):
        # Usar dataset Vimeo Septuplet
        print("Usando dataset Vimeo Septuplet")
        train_ds = VimeoSeptupletDataset(vimeo_sequences_path, vimeo_train_list, 
                                         scale_factor=scale, crop_size=crop_size, train=True)
        
        # Usar lista de teste se existir, senão dividir a lista de treino
        if os.path.exists(vimeo_test_list):
            val_ds = VimeoSeptupletDataset(vimeo_sequences_path, vimeo_test_list, 
                                          scale_factor=scale, crop_size=crop_size, train=False)
        else:
            print("Aviso: sep_testlist.txt não encontrado. Usando 10% do treino para validação.")
            # Dividir manualmente a lista de treino
            with open(vimeo_train_list, 'r') as f:
                all_seqs = [line.strip() for line in f.readlines()]
            train_seqs, val_seqs = train_test_split(all_seqs, test_size=0.1, random_state=42)
            
            # Criar arquivos temporários
            temp_train = os.path.join(data_path, 'vimeo_septuplet', 'temp_train.txt')
            temp_val = os.path.join(data_path, 'vimeo_septuplet', 'temp_val.txt')
            
            with open(temp_train, 'w') as f:
                f.write('\n'.join(train_seqs))
            with open(temp_val, 'w') as f:
                f.write('\n'.join(val_seqs))
            
            train_ds = VimeoSeptupletDataset(vimeo_sequences_path, temp_train, 
                                           scale_factor=scale, crop_size=crop_size, train=True)
            val_ds = VimeoSeptupletDataset(vimeo_sequences_path, temp_val, 
                                          scale_factor=scale, crop_size=crop_size, train=False)
    else:
        # Usar dataset genérico (imagens soltas)
        print("Usando dataset genérico de imagens")
        path_obj = Path(data_path)
        all_images = list(path_obj.rglob("*.png")) + list(path_obj.rglob("*.jpg")) + list(path_obj.rglob("*.jpeg"))
        
        if len(all_images) == 0:
            print("Nenhuma imagem encontrada. Verifique o caminho no config.json")
            return

        # Dividimos a LISTA de arquivos, não o Dataset
        train_files, val_files = train_test_split(all_images, test_size=0.1, random_state=42)
          # Criamos Datasets distintos com comportamentos distintos
        train_ds = SRDataset(train_files, scale_factor=scale, crop_size=crop_size, train=True)
        val_ds = SRDataset(val_files, scale_factor=scale, crop_size=crop_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2) # Batch 1 para validação é mais preciso para PSNR médio

    # 3. Modelo e Otimizador
    model = ESPCN(scale_factor=scale, num_frames=1, channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Scheduler: reduz LR quando o Loss para de cair
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    full_ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    # 4. Resume Checkpoint
    start_epoch = 0
    best_psnr = 0.0

    if os.path.exists(full_ckpt_path):
        print(f"Carregando checkpoint: {full_ckpt_path}")
        checkpoint = torch.load(full_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('psnr', 0.0)
        print(f"Resumindo da época {start_epoch} com PSNR base: {best_psnr:.2f} dB")
    else:
        print("Iniciando treino do zero.")

    # 5. Loop Principal
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for lr_imgs, hr_imgs in train_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                
                optimizer.zero_grad()
                sr_imgs = model(lr_imgs)
                
                loss = criterion(sr_imgs, hr_imgs)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.5f}'})
                pbar.update(1)
        
        avg_loss = epoch_loss / len(train_loader)
        
        # --- Validação ---
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_v, hr_v in val_loader:
                lr_v, hr_v = lr_v.to(device), hr_v.to(device)
                sr_v = model(lr_v)
                val_psnr += calc_psnr(sr_v, hr_v)
        
        avg_psnr = val_psnr / len(val_loader)
        
        # O scheduler monitora o Loss de treino (ou você pode mudar para monitorar -avg_psnr)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1} -> Train Loss: {avg_loss:.6f} | Val PSNR: {avg_psnr:.2f} dB")

        # Salvar Melhor Modelo
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': best_psnr,
                'config': config.get_config()
            }, full_ckpt_path)
            print(f"✓ Salvo novo melhor modelo! ({best_psnr:.2f} dB)\n")

if __name__ == '__main__':
    train()