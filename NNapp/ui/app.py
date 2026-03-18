import streamlit as st
import requests
import pandas as pd
import json

FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🏥 Clasificador de Cáncer de Mama")
st.markdown("Dataset: Breast Cancer Wisconsin (Diagnostic)")

col1, col2 = st.columns(2)

with col1:
    st.header("🔬 Entrenar Modelo")
    st.write("Sube un CSV con datos de entrenamiento")
    
    csv_file = st.file_uploader("Selecciona CSV de entrenamiento", type="csv", key="train_file")
    if csv_file:
        if st.button("Entrenar Modelo", key="train_btn"):
            files = {"csv_file": (csv_file.name, csv_file.getvalue(), "text/csv")}
            with st.spinner("🔄 Entrenando modelo..."):
                res = requests.post(f"{FASTAPI_URL}/train", files=files)
            if res.status_code == 200:
                st.success("✅ Modelo entrenado exitosamente")
                st.json(res.json())
            else:
                st.error(f"❌ Error: {res.text}")

with col2:
    st.header("🔮 Hacer Predicción")
    st.write("Carga un CSV con datos a predecir")
    
    pred_file = st.file_uploader("Selecciona CSV de predicción", type="csv", key="pred_file")
    if pred_file:
        df = pd.read_csv(pred_file)
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Predecir", key="pred_btn"):
            payload = {"tabular_data": df.to_json(orient="records")}
            with st.spinner("🔄 Realizando predicciones..."):
                r = requests.post(f"{FASTAPI_URL}/predict/", data=payload)
            
            if r.status_code == 200:
                predictions = r.json()["predictions"]
                
                st.success("✅ Predicciones completadas")
                
                # Mostrar resultados
                results_df = pd.DataFrame({
                    "Predicción (1: Maligno, 0: Benigno)": predictions,
                    "Diagnóstico": ["🔴 MALIGNO" if p == 1 else "🟢 BENIGNO" for p in predictions]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Estadísticas
                benignos = sum(1 for p in predictions if p == 0)
                malignos = sum(1 for p in predictions if p == 1)
                
                st.metric("Casos Benignos", benignos)
                st.metric("Casos Malignos", malignos)
            else:
                st.error(f"❌ Error: {r.text}")
