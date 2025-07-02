from surprise import Dataset
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd
import time
import joblib
import os

def load_or_train_model(model_name, dataset_name):
    """Carga modelo pre-entrenado o lo entrena si no existe"""
    model_path = f"models/trained_models/{model_name}_{dataset_name}.joblib"
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Importar y ejecutar la función de entrenamiento adecuada
        if model_name.startswith('KNN'):
            from models.knn.knn_train import train_knn_model
            return train_knn_model(model_type=model_name, dataset=dataset_name)
        elif model_name in ['SVD', 'SVDpp']:
            from models.svd.svd_train import train_svd_model
            return train_svd_model(model_type=model_name, dataset=dataset_name)
        elif model_name == 'NMF':
            from models.nmf.nmf_train import train_nmf_model
            return train_nmf_model(dataset=dataset_name)
        elif model_name == 'SlopeOne':
            from models.slope_one.slope_one_train import train_slope_one
            return train_slope_one(dataset=dataset_name)

def compare_models(model_names, dataset_name='ml-100k', cv=5, test_size=0.2):
    """Compara múltiples modelos con métricas consistentes"""
    data = Dataset.load_builtin(dataset_name)
    results = {}
    
    for name in model_names:
        try:
            model = load_or_train_model(name, dataset_name)
            
            # Validación cruzada
            cv_results = cross_validate(
                model, data,
                measures=['rmse', 'mae'],
                cv=cv,
                verbose=False
            )
            
            # Test set fijo para tiempos
            trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
            
            start_time = time.time()
            model.fit(trainset)
            fit_time = time.time() - start_time
            
            start_time = time.time()
            predictions = model.test(testset)
            pred_time = (time.time() - start_time) / len(testset) * 1000  # ms por predicción
            
            results[name] = {
                'RMSE': cv_results['test_rmse'].mean(),
                'MAE': cv_results['test_mae'].mean(),
                'Fit Time': fit_time,
                'Test Time': pred_time
            }
            
        except Exception as e:
            print(f"Error evaluando {name}: {str(e)}")
            continue
    
    return pd.DataFrame.from_dict(results, orient='index')