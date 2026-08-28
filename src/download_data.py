from pathlib import Path;
import kagglehub; 


ROOT_DIR = Path(__file__).resolve().parent.parent;
DATA_DIR = ROOT_DIR / "data" / "raw";

path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset", output_dir=str(DATA_DIR));

print(path);