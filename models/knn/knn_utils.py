from surprise import accuracy
from surprise.model_selection import cross_validate

def evaluate_knn(model, data, cv=5):
    """Evalúa modelo KNN con validación cruzada"""
    return cross_validate(
        model, 
        data, 
        measures=['RMSE', 'MAE'], 
        cv=cv, 
        verbose=True
    )

def get_knn_recommendations(model, user_id, movies_df, n=10):
    """Genera recomendaciones para un usuario"""
    predictions = []
    for movie_id in movies_df['movie_id']:
        pred = model.predict(str(user_id), str(movie_id))
        predictions.append((movie_id, pred.est))
    
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]