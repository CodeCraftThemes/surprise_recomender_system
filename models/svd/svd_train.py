from surprise import SVD, SVDpp, Dataset
import joblib
import os

def train_svd_model(model_type='SVD', dataset='ml-100k', n_factors=100, n_epochs=20):
    """Entrena modelos de factorización matricial"""
    
    models = {
        'SVD': SVD,
        'SVDpp': SVDpp
    }
    
    data = Dataset.load_builtin(dataset)
    trainset = data.build_full_trainset()
    
    model = models[model_type](n_factors=n_factors, n_epochs=n_epochs)
    model.fit(trainset)
    
    os.makedirs('./models/trained_models', exist_ok=True)
    joblib.dump(model, f'./models/trained_models/{model_type}_{dataset}.joblib')
    
    return model