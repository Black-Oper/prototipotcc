import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_datasets():
    
    base_path = Path("./datasets")
    base_path.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        'vimeo_septuplet': {
            'url': 'http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip',
            'type': 'zip',
            'folder': 'vimeo_septuplet'
        }
    }

    print("Iniciando download dos datasets...")

    for dataset_name, dataset_info in datasets.items():
        dataset_folder = base_path / dataset_info['folder']
        
        if dataset_folder.exists() and any(dataset_folder.iterdir()):
            print(f"✓ {dataset_name} já existe")
            continue
        
        dataset_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Baixando {dataset_name}...")
        temp_file = base_path / f"{dataset_name}.zip"
        
        response = requests.get(dataset_info['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_file, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        print(f"Extraindo {dataset_name}...")
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)

        # Se o zip criou uma pasta com o mesmo nome dentro da pasta de destino
        # (ex.: datasets/vimeo_septuplet/vimeo_septuplet/...), achatar para que
        # o restante do código encontre sequences/ diretamente.
        nested = dataset_folder / dataset_folder.name
        if nested.is_dir():
            for item in nested.iterdir():
                item.rename(dataset_folder / item.name)
            nested.rmdir()

        temp_file.unlink()
        print(f"✓ {dataset_name} baixado com sucesso!\n")
    
    print("Download concluído!")

if __name__ == "__main__":
    download_datasets()
    
    