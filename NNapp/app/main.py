from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.inference import run_inference
from utils.preprocess import preprocess_tabular
from pathlib import Path
import pandas as pd
import json
from utils.data import load_and_split
from models import trainer_tf

app = FastAPI()

# Configurar CORS para que Streamlit pueda hacer requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(tabular_data: str = Form(...)):
    """Predicción con datos tabulares de Breast Cancer"""
    data = json.loads(tabular_data)
    df = pd.DataFrame(data)
    X, _ = preprocess_tabular(df)
    result = run_inference(X, "tensorflow")
    return {"predictions": result}

@app.post("/train")
async def train_model(csv_file: UploadFile):
    """Entrenar modelo con dataset CSV"""
    temp_path = Path("uploads") / csv_file.filename
    Path("uploads").mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await csv_file.read())

    X_train, X_test, y_train, y_test = load_and_split(temp_path)
    model_path = trainer_tf.train_tabular(X_train, y_train, X_test, y_test, epochs=20)

    return {"status": "ok", "saved_model": str(model_path)}
