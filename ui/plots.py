import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import norm


def renderizar_eda(df: pd.DataFrame, columna: str):
    """
    Renderiza un Histograma acoplado con un Boxplot superior.
    Al encapsular ambos en un solo objeto Figure, minimizamos las llamadas al DOM.
    """
    fig = px.histogram(
        df,
        x=columna,
        marginal="box",
        title="Distribución de la Muestra (Análisis Exploratorio)",
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(bargap=0.1)
    return fig


def renderizar_curva_z(resultados_stats: dict):
    """
    Proyecta la distribución normal estándar y mapea las zonas de divergencia (rechazo).
    """
    z_stat = resultados_stats['z_stat']
    z_critico = resultados_stats['z_critico']

    x = np.linspace(-4, 4, 500)
    y = norm.pdf(x)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Distribución Normal (H0)',
        line=dict(color='white')
    ))

    fig.add_trace(go.Scatter(
        x=[z_stat, z_stat],
        y=[0, norm.pdf(z_stat)],
        mode='lines',
        name=f'Z Calculado ({z_stat:.2f})',
        line=dict(color='cyan', width=3, dash='dash')
    ))

    if z_critico['izq'] is not None:
        x_izq = np.linspace(-4, z_critico['izq'], 100)
        y_izq = norm.pdf(x_izq)
        fig.add_trace(go.Scatter(
            x=x_izq, y=y_izq,
            fill='tozeroy', mode='none',
            name='Rechazo Izq',
            fillcolor='rgba(255, 50, 50, 0.5)'
        ))

    if z_critico['der'] is not None:
        x_der = np.linspace(z_critico['der'], 4, 100)
        y_der = norm.pdf(x_der)
        fig.add_trace(go.Scatter(
            x=x_der, y=y_der,
            fill='tozeroy', mode='none',
            name='Rechazo Der',
            fillcolor='rgba(255, 50, 50, 0.5)'
        ))

    fig.update_layout(
        title="Mapeo de Divergencia (Prueba Z)",
        xaxis_title="Valor Z",
        yaxis_title="Densidad de Probabilidad",
        showlegend=True
    )
    return fig
