import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

# Datos simulados (en producción cargarías ml-100k)
def load_mock_data():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5],
        "title": ["Toy Story", "The Godfather", "Inception", "Pulp Fiction", "The Dark Knight"],
        "genre": ["Animation|Comedy", "Crime|Drama", "Sci-Fi|Thriller", "Crime|Drama", "Action|Crime"]
    })
    
    users = pd.DataFrame({
        "user_id": [101, 102, 103],
        "age": [25, 30, 35],
        "gender": ["Masculino", "Femenino", "Masculino"],
        "occupation": ["Student", "Engineer", "Doctor"]
    })
    
    ratings = pd.DataFrame({
        "user_id": [101, 101, 102, 103],
        "movie_id": [1, 2, 3, 5],
        "rating": [5, 4, 3, 5]
    })
    return movies, users, ratings

movies, users, ratings = load_mock_data()

# Modelo SVD (simplificado para el ejemplo)
def train_model():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

model = train_model()

def show():
    st.title("🎬 Sistema de Recomendación para Nuevos Usuarios")
    
    # --- SECCIÓN 1: Formulario demográfico ---
    with st.form("user_profile"):
        st.header("📝 Perfil Demográfico")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad:", 1, 100, 25)
            gender = st.selectbox("Género:", ["M", "F"])
        
        with col2:
            occupation = st.selectbox(
                "Ocupación:", 
                ["administrator","artist","doctor","educator","engineer",
                 "entertainment","executive","healthcare","homemaker","lawyer",
                 "librarian","marketing","none","other","programmer","retired",
                 "salesman","scientist","student","technician","writer"]
            )
            zip_code = st.text_input("Código Postal (opcional):", "00000")
        
        liked_genres = st.multiselect(
            "Géneros favoritos:", 
            ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
             "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
             "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        )
        
        submitted_profile = st.form_submit_button("Continuar")
    
    # --- SECCIÓN 2: Cold Start (calificar películas) ---
    if submitted_profile:
        st.header("🌟 Califica 5 películas para personalizar recomendaciones")
        selected_movies = movies.sample(5)
        user_ratings = {}
        
        with st.form("movie_ratings"):
            for _, movie in selected_movies.iterrows():
                user_ratings[movie["movie_id"]] = st.slider(
                    f"Califica '{movie['title']}':",
                    1, 5, 3,
                    key=f"rate_{movie['movie_id']}"
                )
            
            submitted_ratings = st.form_submit_button("Generar Recomendaciones")
        
        # --- SECCIÓN 3: Recomendaciones ---
        if submitted_ratings:
            # Paso 1: Encontrar usuarios similares (ejemplo simplificado)
            similar_users = users[
                (users["age"].between(age-5, age+5)) &
                (users["gender"] == gender)
            ]["user_id"].tolist()
            
            # Paso 2: Obtener películas mejor calificadas por similares
            top_movies = ratings[
                ratings["user_id"].isin(similar_users) &
                (ratings["rating"] >= 4)
            ]["movie_id"].value_counts().head(10).index.tolist()
            
            recommended_movies = movies[
                movies["movie_id"].isin(top_movies) &
                ~movies["movie_id"].isin(user_ratings.keys())  # Excluir ya calificadas
            ]
            
            # Paso 3: Mostrar resultados
            st.success("🎉 Recomendaciones personalizadas para ti:")
            
            # Opción 1: Lista simple
            st.dataframe(recommended_movies[["title", "genre"]])
            
            
if __name__ == "__main__":
    show()