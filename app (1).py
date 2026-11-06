
from transformers import pipeline
import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="💬",
    layout="wide"
)

# 2. Cargar el modelo de inteligencia artificial
@st.cache_resource
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )

analizador = cargar_modelo()

# 3. Título y descripción
st.title("💬 Análisis de Sentimientos")
st.markdown("""
Esta aplicación utiliza inteligencia artificial
para identificar sentimientos en comentarios escritos en español.
""")

# 4. Entrada de usuario
texto = st.text_area(
    "Escribe un comentario:",
    placeholder="Ejemplo: Me encanta esta aplicación."
)

# 5. Crear historial de resultados
if "historial" not in st.session_state:
    st.session_state.historial = []

# 6. Botón para analizar el comentario
if st.button("Analizar sentimiento"):

    if texto.strip():

        resultado = analizador(texto)[0]

        etiqueta = resultado["label"]
        confianza = resultado["score"]

        nombres = {
            "POS": "Positivo",
            "NEG": "Negativo",
            "NEU": "Neutral"
        }

        st.subheader("Resultado del análisis")

        st.metric(
            "Sentimiento identificado",
            nombres.get(etiqueta, etiqueta)
        )

        st.metric(
            "Nivel de confianza",
            f"{confianza:.2%}"
        )

        # Guardar resultado
        st.session_state.historial.append({
            "Comentario": texto,
            "Sentimiento": nombres.get(etiqueta, etiqueta),
            "Confianza": confianza
        })

    else:
        st.warning("Por favor, escribe un comentario.")

# 7. Mostrar historial y gráfico
if st.session_state.historial:

    st.subheader("Historial de comentarios")

    df = pd.DataFrame(st.session_state.historial)

    st.dataframe(df, use_container_width=True)

    conteo = df["Sentimiento"].value_counts().reindex(
        ["Positivo", "Negativo", "Neutral"],
        fill_value=0
    )

    figura = px.bar(
        x=conteo.index,
        y=conteo.values,
        title="Resultados del análisis de sentimientos",
        labels={
            "x": "Tipo de sentimiento",
            "y": "Cantidad de comentarios"
        }
    )

    st.plotly_chart(figura, use_container_width=True)
