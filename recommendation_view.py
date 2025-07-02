from utils.data_loader import get_movies_data, load_dataset
from utils.recommender import load_trained_model, get_top_n_recommendations
import streamlit as st

def show():
    """Función principal que muestra la vista de recomendaciones"""
    # Configuración de página (esto podría moverse al app principal)
    #st.set_page_config(layout="wide", page_title="Sistema de Recomendación")
    
    # Sidebar para controles
    with st.sidebar:
        st.header("Configuración")
        dataset = st.selectbox("Dataset:", ["ml-100k"])
        model_type = st.selectbox(
            "Modelo:",
            ["KNNBasic", "KNNWithMeans", "SVD", "SVDpp", "NMF", "SlopeOne", "CoClustering"]
        )
        rigor = st.slider("Nivel de rigor (rating mínimo):", 1.0, 5.0, 3.5, 0.5)
    
    # Vista principal
    st.title(f"🎬 Recomendaciones con {model_type}")
    
    # Cargar datos
    movies = get_movies_data(dataset)
    user_id = st.number_input("ID de Usuario:", min_value=1, max_value=1000, value=1)
    
    if st.button("Generar Recomendaciones"):
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

# Para pruebas independientes (opcional)
if __name__ == "__main__":
    show()