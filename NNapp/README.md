#  Clasificador de Cáncer de Mama - Breast Cancer Wisconsin

Aplicación web para clasificación de diagnóstico de cáncer de mama usando redes neuronales con TensorFlow y Keras.

## Dataset

**Breast Cancer Wisconsin ** - Dataset de sklearn
- **Muestras**: 569 casos
- **Features**: 30 características (propiedades celulares)
- **Target**: Binario (0: Benigno, 1: Maligno)

## Arquitectura de la Red Neuronal

```
Input (30 features)
    ↓
Dense(32, relu)
    ↓
Dense(16, relu)
    ↓
Dense(1, sigmoid)  ← Output (probabilidad)
```
## Instalación

### 1. Crear Entorno Virtual

```bash
python -m venv venv
```

Activar el entorno:
- **Windows**: `venv\Scripts\activate`
- **Linux/Mac**: `source venv/bin/activate`

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Proyecto (Generar datasets y entrenar modelo)

```bash
python setup.py
```

Este script:
- ✓ Genera `datasets/breast_cancer_train.csv`
- ✓ Genera `datasets/breast_cancer_pred.csv`
- ✓ Entrena el modelo automáticamente

## Ejecución

### 1. Iniciar Backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verificar: http://localhost:8000/docs

### 2. Iniciar Frontend (Streamlit)

En otra terminal:

```bash
streamlit run ui/app.py
```

La aplicación abrirá en: http://localhost:8501

## Uso

### Entrenar el Modelo
1. En la sección izquierda, sube `datasets/breast_cancer_train.csv`
2. Haz clic en "Entrenar Modelo"
3. El modelo se guardará en `models/saved/tf_tabular.keras`

### Realizar Predicciones
1. En la sección derecha, sube `datasets/breast_cancer_pred.csv`
2. Haz clic en "Predecir"
3. Visualiza los resultados (Benigno/Maligno) con estadísticas

## Estructura del Proyecto

```
NNapp/
├── app/
│   ├── __init__.py
│   └── main.py              # Backend FastAPI
│
├── models/
│   ├── __init__.py
│   ├── tensorflow_arch.py   # Arquitectura del modelo
│   ├── trainer_tf.py        # Entrenamiento
│   └── saved/
│       └── tf_tabular.keras # Modelo entrenado
│
├── ui/
│   ├── __init__.py
│   └── app.py               # Frontend Streamlit
│
├── utils/
│   ├── __init__.py
│   ├── data.py              # Carga y división de datos
│   ├── preprocess.py        # Normalización
│   └── inference.py         # Predicción
│
├── datasets/
│   ├── breast_cancer_train.csv
│   └── breast_cancer_pred.csv
│
├── generate_datasets.py     # Script para generar CSVs
├── requirements.txt
└── README.md
```

## Dependencias Principales

- **TensorFlow 2.16+** - Framework de deep learning
- **FastAPI 0.110+** - Backend API
- **Streamlit 1.36+** - Frontend interactivo
- **pandas** - Manipulación de datos
- **scikit-learn** - Dataset y preprocesamiento

## Equipo

- Desarrollado Sergio Andres Martinez 2220231060
               Jhoan Ortiz            2220231054
               Diego Vargas           2220231068
- Proyecto: Red Neuronal Tabular - Diagnóstico de Cáncer de Mama

## Notas

- El modelo usa normalización MinMax (0-1) en los features
- Las predicciones se realizan con probabilidades > 0.5 como umbral para clase positiva (Maligno)
- Los datos se dividen 80% entrenamiento / 20% validación

---
