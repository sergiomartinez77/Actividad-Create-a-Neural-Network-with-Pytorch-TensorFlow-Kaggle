import numpy as np

def run_inference(x: np.ndarray, framework: str = "tensorflow") -> list:
    """
    Ejecuta inferencia con el modelo de Breast Cancer Wisconsin.
    Retorna lista de predicciones (0: Benigno, 1: Maligno).
    """
    if framework == "tensorflow":
        from models.tensorflow_arch import build_tabular_model
        from pathlib import Path
        
        model_path = Path("models/saved/tf_tabular.keras")
        if model_path.exists():
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
        else:
            model = build_tabular_model(input_dim=30)
        
        preds = model.predict(x, verbose=0)
        # Convertir probabilidades a clases (0 o 1)
        return (preds > 0.5).astype(int).flatten().tolist()
    
    return []
