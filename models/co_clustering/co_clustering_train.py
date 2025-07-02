from surprise import CoClustering, Dataset
import joblib
import os

def train_co_clustering(dataset='ml-100k', n_cltr_u=3, n_cltr_i=3, n_epochs=20):
    """Entrena modelo CoClustering"""
    
    data = Dataset.load_builtin(dataset)
    trainset = data.build_full_trainset()
    
    model = CoClustering(n_cltr_u=n_cltr_u, n_cltr_i=n_cltr_i, n_epochs=n_epochs)
    model.fit(trainset)
    
    os.makedirs('./models/trained_models', exist_ok=True)
    joblib.dump(model, f'./models/trained_models/CoClustering_{dataset}.joblib')
    
    return model