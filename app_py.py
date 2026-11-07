
from transformers import pipeline
import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="💬",
    layout="wide"
)

# Cargar el modelo de análisis
@st.cache_resource
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )

# Inicializar historial
if "historial" not in st.session_state:
    st.session_state.historial = []

# Título
st.title("💬 Análisis de Sentimientos")

st.write(
    "Aplicación para identificar sentimientos "
    "en comentarios escritos en español."
)

# Área para escribir comentarios
with st.form("formulario_sentimientos"):
    texto = st.text_area(
        "Escribe un comentario:",
        placeholder="Ejemplo: Me encanta esta aplicación."
    )

    analizar = st.form_submit_button("Analizar sentimiento")

# Analizar comentario
if analizar:

    if texto.strip():

        try:
            with st.spinner("Analizando comentario..."):

                analizador = cargar_modelo()
                resultado = analizador(texto)[0]

                nombres = {
                    "POS": "Positivo",
                    "NEG": "Negativo",
                    "NEU": "Neutral"
                }

                etiqueta = nombres.get(
                    resultado["label"],
                    resultado["label"]
                )

                confianza = resultado["score"]

                st.subheader("Resultado del análisis")

                st.metric(
                    "Sentimiento identificado",
                    etiqueta
                )

                st.metric(
                    "Nivel de confianza",
                    f"{confianza:.2%}"
                )

                st.session_state.historial.append({
                    "Comentario": texto,
                    "Sentimiento": etiqueta,
                    "Confianza": confianza
                })

        except Exception as e:
            st.error(f"Error al analizar el comentario: {e}")

    else:
        st.warning("Por favor, escribe un comentario.")

# Mostrar historial
if st.session_state.historial:

    st.subheader("Historial de comentarios")

    df = pd.DataFrame(st.session_state.historial)

    st.dataframe(
        df,
        use_container_width=True
    )

    # Contar sentimientos
    conteo = df["Sentimiento"].value_counts().reindex(
        ["Positivo", "Negativo", "Neutral"],
        fill_value=0
    )

    # Crear gráfica
    figura = px.bar(
        x=conteo.index,
        y=conteo.values,
        title="Resultados del análisis de sentimientos",
        labels={
            "x": "Tipo de sentimiento",
            "y": "Cantidad de comentarios"
        },
        color=conteo.index
    )

    st.plotly_chart(
        figura,
        use_container_width=True
    )
