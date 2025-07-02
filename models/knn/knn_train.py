from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import Dataset
import joblib
import os

def train_knn_model(model_type='KNNBasic', dataset='ml-100k'):
    """Entrena y guarda modelos KNN"""
    
    # Configuración de similitud
    sim_options = {
        'name': 'msd',
        'user_based': True,
        'min_support': 3
    }
    
    # Selección de modelo
    models = {
        'KNNBasic': KNNBasic,
        'KNNWithMeans': KNNWithMeans,
        'KNNWithZScore': KNNWithZScore,
        'KNNBaseline': KNNBaseline
    }
    
    # Cargar datos
    data = Dataset.load_builtin(dataset)
    trainset = data.build_full_trainset()
    
    # Crear y entrenar modelo
    model = models[model_type](sim_options=sim_options)
    model.fit(trainset)
    
    # Guardar modelo
    os.makedirs('./models/trained_models', exist_ok=True)
    joblib.dump(model, f'./models/trained_models/{model_type}_{dataset}.joblib')
    
    return model