import streamlit as st
import numpy as np
import pandas as pd
from core.stats_engine import generar_datos_sinteticos, calcular_prueba_z
from ui.plots import renderizar_eda, renderizar_curva_z
from ai.gemini_client import consultar_oraculo

st.set_page_config(page_title="Divergence Meter Z",
                   page_icon="📊", layout="wide")
st.title("Divergence Meter Z: Oráculo Estadístico")
st.markdown("---")


@st.cache_data(show_spinner=False)
def cargar_datos_sinteticos(n: int, mu: float, sigma: float):
    return generar_datos_sinteticos(n, mu, sigma)


@st.cache_data(show_spinner=False)
def purgar_y_preparar_csv(df_raw: pd.DataFrame, target_col: str):
    df_clean = df_raw.dropna(subset=[target_col]).copy()
    return df_clean.rename(columns={target_col: 'Valor'})


st.sidebar.header("Parámetros de Ingesta")
tipo_datos = st.sidebar.radio(
    "Fuente de datos:", ["Generación Sintética", "Cargar CSV"])

df = None
desv_pob = 15.0

if tipo_datos == "Generación Sintética":
    st.sidebar.subheader("Parámetros Poblacionales")
    n_muestras = st.sidebar.slider(
        "Tamaño de muestra (n)", min_value=10, max_value=1000, value=50, step=10)
    media_pob = st.sidebar.number_input("Media (μ)", value=100.0)
    desv_pob = st.sidebar.number_input(
        "Desviación Estándar (σ)", value=15.0, min_value=0.01)

    df = cargar_datos_sinteticos(n_muestras, media_pob, desv_pob)

    st.subheader("Datos de la Muestra")
    st.dataframe(df.head(10), use_container_width=True)

elif tipo_datos == "Cargar CSV":
    st.sidebar.subheader("Carga de Archivo")
    archivo_subido = st.sidebar.file_uploader(
        "Sube tu dataset (.csv)", type=["csv"])

    if archivo_subido is not None:
        try:
            df_raw = pd.read_csv(archivo_subido)
            columnas_numericas = df_raw.select_dtypes(
                include=np.number).columns.tolist()

            if not columnas_numericas:
                st.error(
                    "Violación de integridad: El CSV no contiene columnas numéricas válidas.")
                st.stop()

            columna_objetivo = st.sidebar.selectbox(
                "Selecciona la variable a analizar:", columnas_numericas)
            desv_pob = st.sidebar.number_input(
                "Desviación Est. Poblacional (σ) asumida", value=15.0, min_value=0.01)

            df = purgar_y_preparar_csv(df_raw, columna_objetivo)

            if df.empty or len(df) < 30:
                st.error(
                    "Bloqueo de seguridad: Tras limpiar datos nulos, la muestra es menor a 30 registros.")
                st.stop()

            st.subheader(f"Datos de la Muestra (Variable: {columna_objetivo})")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Fallo crítico en el parseo del archivo: {e}")
            st.stop()
    else:
        st.info("Esperando inyección de archivo CSV para proceder con el pipeline...")
        st.stop()

st.plotly_chart(renderizar_eda(df, 'Valor'), use_container_width=True)

st.markdown("---")
st.header("Motor de Hipótesis (Z-Test)")

col1, col2, col3 = st.columns(3)
with col1:
    mu_0 = st.number_input("Hipótesis Nula (μ0)", value=100.0)
with col2:
    alpha = st.selectbox("Nivel de Significancia (α)",
                         [0.01, 0.05, 0.10], index=1)
with col3:
    tipo_prueba = st.selectbox(
        "Lateralidad", ["Bilateral", "Cola izquierda", "Cola derecha"])

try:
    resultados_stats = calcular_prueba_z(
        df=df,
        columna='Valor',
        mu_0=mu_0,
        sigma=desv_pob,
        alpha=alpha,
        tipo_prueba_str=tipo_prueba
    )

    st.subheader("Resultados Determinísticos")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Estadístico Z", f"{resultados_stats['z_stat']:.4f}")
    r_col2.metric("P-Value", f"{resultados_stats['p_value']:.4e}")
    r_col3.metric(
        "Rechaza H0", "Sí" if resultados_stats['rechaza_H0'] else "No")

    criticos_str = ""
    if resultados_stats['z_critico']['izq'] is not None:
        criticos_str += f"Z ≤ {resultados_stats['z_critico']['izq']:.2f} "
    if resultados_stats['z_critico']['der'] is not None:
        criticos_str += f"Z ≥ {resultados_stats['z_critico']['der']:.2f}"

    r_col4.metric("Región Crítica", criticos_str)

    st.plotly_chart(renderizar_curva_z(resultados_stats),
                    use_container_width=True)

    st.markdown("---")
    st.header("🔮 Análisis Semántico (Gemini AI)")

    if st.button("Generar Interpretación Automática", type="primary"):
        with st.spinner("Conectando con el oráculo bajo canal seguro..."):
            respuesta_ia = consultar_oraculo(resultados_stats)

            if "⚠️ [ALERTA ZERO TRUST]" in respuesta_ia or "Error" in respuesta_ia:
                st.error(respuesta_ia)
            else:
                st.success("Análisis validado y completado.")
                st.markdown(f"> *{respuesta_ia}*")

except ValueError as e:
    st.error(f"Bloqueo de Seguridad Matemática: {e}")
