import streamlit as st

def show():  # Cambiar el nombre de la función
    st.title("👤 Recomendaciones para Nuevos Usuarios")
    
    with st.form("user_form"):
        st.write("Complete sus preferencias:")
        
        age = st.number_input("Edad:", min_value=1, max_value=100, value=25)
        gender = st.radio("Género:", ["Masculino", "Femenino", "Otro"])
        
        genres = [
            "Action", "Comedy", "Drama", "Horror", 
            "Sci-Fi", "Thriller", "Romance"
        ]
        liked_genres = st.multiselect("Géneros favoritos:", genres)
        
        submitted = st.form_submit_button("Obtener Recomendaciones")
        
        if submitted:
            # Lógica para encontrar usuarios similares
            st.success(f"Recomendaciones basadas en usuarios similares ({', '.join(liked_genres)})")
            # Aquí integrarías la lógica de recomendación híbrida

if __name__ == "__main__":
    show()