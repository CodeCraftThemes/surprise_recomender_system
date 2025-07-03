from utils.data_loader import get_movies_data, load_dataset, get_users_data
from utils.recommender import load_trained_model, get_top_n_recommendations
import streamlit as st
import pandas as pd

def show():
    """Función principal que muestra la vista de recomendaciones"""
    # Configuración de página
    st.title("🎬 Sistema de Recomendación")
    
    # Sidebar para controles
    with st.sidebar:
        st.header("Configuración")
        dataset = st.selectbox("Dataset:", ["ml-100k"])
        model_type = st.selectbox(
            "Modelo:",
            ["KNNBasic", "KNNWithMeans", "SVD", "SVDpp", "NMF", "SlopeOne", "CoClustering"]
        )
        rigor = st.slider("Nivel de rigor (rating mínimo):", 1.0, 5.0, 3.5, 0.5)
    
    # Cargar datos
    movies = get_movies_data(dataset)
    users_list = get_users_data(dataset)
    
    # Selección de usuario
    selected_user = st.selectbox(
        "Selecciona un usuario:", 
        options=users_list,
        index=None,
        placeholder="Elige un usuario...",
        key="user_select"
    )
    
    # Extraer user_id cuando se selecciona un usuario
    user_id = int(selected_user.split('-')[0]) if selected_user else None
    
    if st.button("Generar Recomendaciones"):
        if not user_id:
            st.warning("Por favor selecciona un usuario primero")
            return
            
        try:
            model = load_trained_model(model_type, dataset)
            recs = get_top_n_recommendations(model, user_id, movies, rating_threshold=rigor)
            
            # Mostrar resultados
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader(f"Top 10 para Usuario {user_id}")
                st.dataframe(
                    recs.style.highlight_max(axis=0),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Distribución de Ratings")
                st.bar_chart(recs.set_index("title")["rating"])
                
        except FileNotFoundError:
            st.error(f"⚠️ Modelo {model_type} no encontrado. Entrénalo primero.")

if __name__ == "__main__":
    show()