import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.title("📊 Evaluación de Modelos")
    
    # Sidebar con controles
    with st.sidebar:
        st.header("Configuración")
        dataset = st.selectbox("Dataset:", ["ml-100k"], key="eval_dataset")
        
        # Diccionario de funciones de entrenamiento
        model_functions = {
            "KNNBasic": get_knn_train_function("KNNBasic"),
            "KNNWithMeans": get_knn_train_function("KNNWithMeans"),
            "SVD": get_svd_train_function("SVD"),
            "SVDpp": get_svd_train_function("SVDpp"),
            "NMF": get_nmf_train_function(),
            "SlopeOne": get_slope_one_train_function()
        }
        
        selected_models = st.multiselect(
            "Modelos a evaluar:",
            list(model_functions.keys()),
            default=list(model_functions.keys())
        )
    
    # Vista principal
    if st.button("✅ Ejecutar Evaluación Comparativa"):
        if not selected_models:
            st.warning("⚠️ Por favor selecciona al menos un modelo")
            return
            
        with st.spinner(f"Evaluando {len(selected_models)} modelos. Por favor espere..."):
            from utils.evaluator import compare_models
            
            # Cargar solo los modelos seleccionados
            models_to_evaluate = {name: model_functions[name] for name in selected_models}
            
            try:
                results = compare_models(models_to_evaluate, dataset)
                display_metrics(results)
            except Exception as e:
                st.error(f"❌ Error durante la evaluación: {str(e)}")

def get_knn_train_function(model_type):
    """Devuelve función de entrenamiento para modelos KNN"""
    from models.knn.knn_train import train_knn_model
    return lambda: train_knn_model(model_type=model_type)

def get_svd_train_function(model_type):
    """Devuelve función de entrenamiento para modelos SVD"""
    from models.svd.svd_train import train_svd_model
    return lambda: train_svd_model(model_type=model_type)

def get_nmf_train_function():
    """Devuelve función de entrenamiento para NMF"""
    from models.nmf.nmf_train import train_nmf_model
    return train_nmf_model

def get_slope_one_train_function():
    """Devuelve función de entrenamiento para SlopeOne"""
    from models.slope_one.slope_one_train import train_slope_one
    return train_slope_one

def display_metrics(results_df):
    """Muestra métricas con gráficos y análisis"""
    st.subheader("Resultados de Evaluación")
    
    # Métricas principales
    metrics = [
        ('RMSE', 'Error Cuadrático Medio (menor es mejor)'),
        ('MAE', 'Error Absoluto Medio (menor es mejor)'),
        ('Fit Time', 'Tiempo de Entrenamiento (segundos)'),
        ('Test Time', 'Tiempo por Predicción (ms)')
    ]
    
    for metric, description in metrics:
        if metric in results_df.columns:
            # Gráfico de barras
            st.markdown(f"### {description}")
            fig, ax = plt.subplots(figsize=(10, 4))
            results_df[metric].sort_values().plot(
                kind='bar', 
                ax=ax,
                color=sns.color_palette("viridis", len(results_df))
            )
            plt.xticks(rotation=45)
            ax.set_xlabel("Modelos")
            ax.set_ylabel(metric)
            st.pyplot(fig)
            
            # Análisis interpretativo
            is_lower_better = metric in ['RMSE', 'MAE', 'Fit Time', 'Test Time']
            best_value = results_df[metric].min() if is_lower_better else results_df[metric].max()
            best_model = results_df[metric].idxmin() if is_lower_better else results_df[metric].idxmax()
            
            st.markdown(f"""
            **Mejor modelo**: `{best_model}` ({best_value:.4f})
            
            **Interpretación**:
            {get_metric_interpretation(metric, best_model, best_value)}
            """)
            
            st.divider()
    
    # Resumen ejecutivo
    st.subheader("📌 Resumen Comparativo")
    st.dataframe(
        results_df.style.highlight_min(
            subset=['RMSE', 'MAE', 'Fit Time', 'Test Time'],
            color='lightgreen'
        )
    )

def get_metric_interpretation(metric, best_model, best_value):
    """Provee explicaciones técnicas contextualizadas"""
    interpretations = {
        'RMSE': (
            f"El modelo {best_model} tiene el menor error cuadrático medio ({best_value:.4f}). "
            "Valores típicos buenos para ml-100k están entre 0.85-1.00. "
            "Diferencias >0.05 entre modelos son consideradas significativas."
        ),
        'MAE': (
            f"El modelo {best_model} tiene el menor error absoluto ({best_value:.4f}). "
            "Esta métrica es más fácil de interpretar directamente que RMSE. "
            "Por ejemplo, un MAE de 0.75 significa que en promedio las predicciones "
            "se desvían ±0.75 estrellas del rating real."
        ),
        'Fit Time': (
            f"El modelo {best_model} fue el más rápido en entrenarse ({best_value:.2f} segundos). "
            "KNN y SVDpp suelen ser los más lentos, mientras que SlopeOne y CoClustering "
            "son generalmente los más rápidos."
        ),
        'Test Time': (
            f"El modelo {best_model} tiene el menor tiempo por predicción ({best_value:.4f} ms). "
            "Importante para sistemas que requieren recomendaciones en tiempo real."
        )
    }
    return interpretations.get(metric, "Métrica sin interpretación específica.")

if __name__ == "__main__":
    show()