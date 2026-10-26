import pandas as pd
from zta.models.inference import load_artifact, predict_scores, top_genres

# Una canción inventada: tú pones los 9 números
cancion = pd.DataFrame({
    "danceability": [0.85], "energy": [0.80], "valence": [0.75],
    "acousticness": [0.05], "instrumentalness": [0.0], "liveness": [0.10],
    "loudness": [-5.0], "tempo": [124.0], "speechiness": [0.05],
}, index=["mi_cancion"])

for nombre in ["linear", "knn", "rf", "svm"]:
    modelo = load_artifact(f"data/processed/models/genre/{nombre}.joblib")
    print(nombre, top_genres(predict_scores(modelo, cancion), 3).values.tolist())
