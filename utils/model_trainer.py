from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset
from surprise.model_selection import train_test_split
import joblib
import os

def train_model(model_name, dataset_name='ml-100k', params=None):
    """Entrena un modelo y lo guarda"""
    # Cargar datos
    data = Dataset.load_builtin(dataset_name)
    trainset = data.build_full_trainset()
    
    # Inicializar modelo
    if model_name == 'KNNBasic':
        model = KNNBasic(**params)
    elif model_name == 'KNNWithMeans':
        model = KNNWithMeans(**params)
    elif model_name == 'KNNWithZScore':
        model = KNNWithZScore(**params)
    elif model_name == 'KNNBaseline':
        model = KNNBaseline(**params)
    elif model_name == 'SVD':
        model = SVD(**params)
    elif model_name == 'SVDpp':
        model = SVDpp(**params)
    elif model_name == 'NMF':
        model = NMF(**params)
    elif model_name == 'SlopeOne':
        model = SlopeOne(**params)
    elif model_name == 'CoClustering':
        model = CoClustering(**params)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
    
    # Entrenar
    model.fit(trainset)
    
    # Guardar modelo
    model_dir = os.path.join('models', 'trained_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_{dataset_name}.joblib")
    joblib.dump(model, model_path)
    
    return model