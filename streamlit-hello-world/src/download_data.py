"""
Download cleaned data files from Google Drive and build models.
This runs automatically on first startup in Streamlit Cloud.
"""
import gdown
from pathlib import Path
import sys

# Add parent to path to import similarity
sys.path.insert(0, str(Path(__file__).parent))

GOOGLE_DRIVE_IDS = {
    'books_clean.csv': '13OPdG-3ZTrjyDl0pg3-HtNGjGpYaHO3u',
}

MODEL_FILES = {
    'models/title_norm_to_row.json': '1KVMRYmv9aGuaHl8RCciDPQGS4y2phh_r',
    'models/X_desc_reduced.npy': '1q_wT_WMt6ASRwja-fH9uKY2ztoUIilvq',
    'models/X_review_reduced.npy': '1G2hPzdlLNPjzwj7q0ovnzfs1_bqgV38r',
}

def download_from_gdrive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {output_path.name}...")
    gdown.download(url, str(output_path), quiet=False)
    print(f"✓ Downloaded {output_path.name}")

def setup_project_data():
    #project root
    script_dir = Path(__file__).resolve().parent  # src/
    streamlit_dir = script_dir.parent  # streamlit-hello-world/
    project_root = streamlit_dir.parent  # data-science-mini-project/
    models_dir = project_root / 'models'
    
    print(f"Project root: {project_root}")
    
    # Download books_clean.csv if it doesn't exist
    for filename, file_id in GOOGLE_DRIVE_IDS.items():
        output_path = project_root / filename

        if output_path.exists():
            print(f"✓ {filename} already exists")
            continue

        download_from_gdrive(file_id, output_path)

    # Download pre-built model files if they don't exist
    models_dir.mkdir(exist_ok=True)

    for model_path, file_id in MODEL_FILES.items():
        output_path = project_root / model_path

        if output_path.exists():
            print(f"✓ {model_path} already exists")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        download_from_gdrive(file_id, output_path)
    
    print("\nSetup complete, now run streamlit app")
    return True

if __name__ == '__main__':
    setup_project_data()
