import streamlit as st
from core.stats_engine import generar_datos_sinteticos, calcular_prueba_z
from ui.plots import renderizar_eda, renderizar_curva_z
from ai.gemini_client import consultar_oraculo

st.set_page_config(
    page_title="Divergence Meter Z",
    page_icon="📊",
    layout="wide"
)

st.title("Divergence Meter Z: Oráculo Estadístico")
st.markdown("---")

st.sidebar.header("Parámetros de Ingesta")
tipo_datos = st.sidebar.radio(
    "Fuente de datos:", ["Generación Sintética", "Cargar CSV"])

if tipo_datos == "Generación Sintética":
    st.sidebar.subheader("Parámetros Poblacionales")
    n_muestras = st.sidebar.slider(
        "Tamaño de muestra (n)", min_value=10, max_value=1000, value=50, step=10)
    media_pob = st.sidebar.number_input("Media (μ)", value=100.0)
    desv_pob = st.sidebar.number_input("Desviación Estándar (σ)", value=15.0)

    df = generar_datos_sinteticos(n_muestras, media_pob, desv_pob)

    st.subheader("Datos de la Muestra")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(
        f"Mostrando los primeros 10 registros de un total de {n_muestras}.")

    st.plotly_chart(renderizar_eda(df, 'Valor'), use_container_width=True)

    st.markdown("**Evaluación humana de la distribución:**")
    c1, c2 = st.columns(2)
    with c1:
        st.radio("¿La distribución parece normal?", ["Sí", "No", "Incierto"])
    with c2:
        st.multiselect("Anomalías detectadas", [
                       "Sesgo a la izquierda", "Sesgo a la derecha", "Outliers visibles", "Ninguna"])

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
            tipo_prueba=tipo_prueba
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

        st.session_state['resultados_stats'] = resultados_stats

        st.markdown("---")
        st.header("🔮 Análisis Semántico (Gemini AI)")
        st.info(
            "El oráculo analizará el vector de resultados sin acceso a los datos crudos.")

        if st.button("Generar Interpretación Automática"):
            with st.spinner("Estableciendo conexión con la API de Gemini..."):
                # Recuperamos el estado inmutable desde la sesión
                payload = st.session_state['resultados_stats']
                respuesta_ia = consultar_oraculo(payload)

                st.success("Análisis completado.")
                st.markdown(f"> *{respuesta_ia}*")

    except ValueError as e:
        st.error(f"Error de Integridad: {e}")
