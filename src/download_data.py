from pathlib import Path;
import kagglehub; 


ROOT_DIR = Path(__file__).resolve().parent.parent;
DATA_DIR = ROOT_DIR / "data" / "raw";
GITKEEP = DATA_DIR / ".gitkeep"

if GITKEEP.exists():
    GITKEEP.unlink();

path = kagglehub.dataset_download(
        "maharshipandya/-spotify-tracks-dataset", 
        output_dir=str(DATA_DIR), 
        force_download=True
    );

GITKEEP.touch();

print(path);