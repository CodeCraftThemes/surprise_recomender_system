from surprise import SlopeOne, Dataset
import joblib
import os

def train_slope_one(dataset='ml-100k'):
    """Entrena modelo SlopeOne"""
    
    data = Dataset.load_builtin(dataset)
    trainset = data.build_full_trainset()
    
    model = SlopeOne()
    model.fit(trainset)
    
    os.makedirs('./models/trained_models', exist_ok=True)
    joblib.dump(model, f'./models/trained_models/SlopeOne_{dataset}.joblib')
    
    return model