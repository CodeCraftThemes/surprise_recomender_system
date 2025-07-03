import pandas as pd
import os
from surprise import Dataset

def get_movies_data(dataset_name='ml-100k'):
    """Carga la información de películas según el dataset seleccionado"""
    if dataset_name == 'ml-100k':
        movies_path = os.path.join('data', 'ml-100k', 'u.item')
        movies = pd.read_csv(
            movies_path,
            sep='|',
            encoding='latin-1',
            header=None,
            names=['movie_id', 'title', 'release_date', 'video_release', 'imdb_url',
                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )
    
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")
    
    return movies[['movie_id', 'title']]

def load_dataset(dataset_name='ml-100k'):
    """Carga el dataset de ratings"""
    return Dataset.load_builtin(dataset_name)

def get_users_data(dataset_name='ml-100k'):
    """Carga y formatea los datos de usuarios"""
    if dataset_name == 'ml-100k':
        users_path = os.path.join('data', 'ml-100k', 'u.user')
        users = pd.read_csv(
            users_path,
            sep='|',
            encoding='latin-1',
            header=None,
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        users['user_info'] = users['user_id'].astype(str) + '-' + \
                             users['age'].astype(str) + '-' + \
                             users['gender'] + '-' + \
                             users['occupation']
        
        return users['user_info'].tolist()  # Devuelve una lista de strings
    
    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")