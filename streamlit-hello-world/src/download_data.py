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
    'authors_clean.csv': '1aeCRRKSYiAxCiZ1uunqHII3J53bOsZzo',
    'reviews_clean.csv': '1VAtguMu85ccUQapCZlL5xEB26EPbI9AS',
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
    
    #Download cleaned CSV files if they don't exist
    for filename, file_id in GOOGLE_DRIVE_IDS.items():
        output_path = project_root / filename
        
        if output_path.exists():
            print(f"✓ {filename} already exists")
            continue
            
        if 'YOUR_' in file_id:
            raise ValueError(f"Please update Google Drive file ID for {filename}")
            
        download_from_gdrive(file_id, output_path)
    
    #Build models if they don't exist
    required_models = [
        'X_desc_reduced.npy',
        'X_review_reduced.npy',
        'title_norm_to_row.json',
        'df_books_text.parquet'
    ]
    
    models_exist = all((models_dir / m).exists() for m in required_models)
    
    if not models_exist:
        print("\nBuilding recommendation models...")
        print("This will take a few minutes on first run...")
        
        from similarity import load_csvs, prepare_book_texts, build_tfidf_and_svd
        
        # Load data
        df_books, df_authors, df_reviews = load_csvs(
            books_path=str(project_root / 'books_clean.csv'),
            authors_path=str(project_root / 'authors_clean.csv'),
            reviews_path=str(project_root / 'reviews_clean.csv')
        )
        
        # Prepare text data
        df_books_text, title_map = prepare_book_texts(df_books, df_reviews)
        
        # Build and save models
        build_tfidf_and_svd(df_books_text, out_dir=models_dir)
        
        print("Models built successfully!")
    else:
        print("Models already exist")
    
    print("\nSetup complete, now run streamlit app")
    return True

if __name__ == '__main__':
    setup_project_data()
