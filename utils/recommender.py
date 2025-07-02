import joblib
import os
from surprise import Dataset
import pandas as pd

def load_trained_model(model_name, dataset_name='ml-100k'):
    """Carga un modelo entrenado"""
    model_path = os.path.join('models', 'trained_models', f"{model_name}_{dataset_name}.joblib")
    return joblib.load(model_path)

def get_top_n_recommendations(model, user_id, movies_df, n=10, rating_threshold=3.5):
    """Genera las top N recomendaciones para un usuario"""
    # Obtener todas las predicciones
    predictions = []
    for movie_id in movies_df['movie_id']:
        pred = model.predict(str(user_id), str(movie_id))
        predictions.append((movie_id, pred.est))
    
    # Crear DataFrame y ordenar
    recs = pd.DataFrame(predictions, columns=['movie_id', 'rating'])
    recs = recs[recs['rating'] >= rating_threshold]
    recs = recs.sort_values('rating', ascending=False).head(n)
    
    # Unir con información de películas
    recs = pd.merge(recs, movies_df[['movie_id', 'title']], on='movie_id')
    
    return recs[['title', 'rating']]