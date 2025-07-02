# interface/app.py
import streamlit as st

# Configuración de página - DEBE SER LO PRIMERO
st.set_page_config(
    layout="wide",
    page_title="Sistema de Recomendación",
    page_icon="🎬"
)

from recommendation_view import show as show_recommendations
from evaluation_view import show as show_evaluation
from new_user_view import show as show_new_user

# Menú de navegación
st.sidebar.title("Navegación")
view = st.sidebar.radio(
    "Ir a:",
    ("🎬 Recomendaciones", "📊 Evaluación", "👤 Nuevo Usuario")
)

# Mostrar vista seleccionada
if view == "🎬 Recomendaciones":
    show_recommendations()
elif view == "📊 Evaluación":
    show_evaluation()
else:
    show_new_user()