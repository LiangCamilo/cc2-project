from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent;
DATASET_PATH = ROOT_DIR / "data" / "raw" / "dataset.csv"

#Reading Spotify dataset with pandas
ds = pd.read_csv(DATASET_PATH)

#app for 



