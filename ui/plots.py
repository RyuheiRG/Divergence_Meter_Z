import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import norm
from core.stats_engine import ZTestResult


def renderizar_eda(df: pd.DataFrame, columna: str):
    """
    Renderiza un Histograma acoplado con un Boxplot superior.
    """
    fig = px.histogram(
        df,
        x=columna,
        marginal="box",
        title=f"Distribución de la Muestra: {columna}",
        color_discrete_sequence=['#3b82f6']
    )
    fig.update_layout(bargap=0.1)
    return fig


def renderizar_curva_z(resultados_stats: ZTestResult):
    """
    Proyecta la distribución normal estándar con un motor de renderizado dinámico.
    Se adapta automáticamente a vectores Z extremos y previene colapsos visuales.
    """
    z_stat = resultados_stats['z_stat']
    z_critico = resultados_stats['z_critico']

    limite_max = max(4.0, abs(z_stat) + 1.5)
    if z_critico['izq'] is not None:
        limite_max = max(limite_max, abs(z_critico['izq']) + 1.5)
    if z_critico['der'] is not None:
        limite_max = max(limite_max, abs(z_critico['der']) + 1.5)

    x = np.linspace(-limite_max, limite_max, 500)
    y = norm.pdf(x)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Distribución Normal (H0)',
        line=dict(color='gray', width=2)
    ))

    altura_maxima = max(y)

    fig.add_trace(go.Scatter(
        x=[z_stat, z_stat],
        y=[0, altura_maxima],
        mode='lines',
        name=f'Z Calculado ({z_stat:.2f})',
        line=dict(color='cyan', width=3, dash='dash')
    ))

    if z_critico['izq'] is not None:
        x_izq = np.linspace(-limite_max, z_critico['izq'], 100)
        fig.add_trace(go.Scatter(
            x=x_izq, y=norm.pdf(x_izq),
            fill='tozeroy', mode='none',
            name='Rechazo Izq',
            fillcolor='rgba(255, 50, 50, 0.5)'
        ))

    if z_critico['der'] is not None:
        x_der = np.linspace(z_critico['der'], limite_max, 100)
        fig.add_trace(go.Scatter(
            x=x_der, y=norm.pdf(x_der),
            fill='tozeroy', mode='none',
            name='Rechazo Der',
            fillcolor='rgba(255, 50, 50, 0.5)'
        ))

    fig.update_layout(
        title="Mapeo de Divergencia (Prueba Z)",
        xaxis_title="Valor Z",
        yaxis_title="Densidad de Probabilidad",
        showlegend=True,
        hovermode="x unified"
    )
    return fig
