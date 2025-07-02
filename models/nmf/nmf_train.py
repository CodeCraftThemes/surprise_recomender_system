from surprise import NMF, Dataset
import joblib
import os

def train_nmf_model(dataset='ml-100k', n_factors=15, n_epochs=50):
    """Entrena modelo NMF"""
    
    data = Dataset.load_builtin(dataset)
    trainset = data.build_full_trainset()
    
    model = NMF(n_factors=n_factors, n_epochs=n_epochs)
    model.fit(trainset)
    
    os.makedirs('./models/trained_models', exist_ok=True)
    joblib.dump(model, f'./models/trained_models/NMF_{dataset}.joblib')
    
    return model