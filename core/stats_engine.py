import numpy as np
import pandas as pd
from scipy import stats


def generar_datos_sinteticos(n: int, media: float, desviacion: float, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame determinístico con una distribución normal.
    El uso del seed garantiza la reproducibilidad del experimento.
    """
    np.random.seed(seed)
    datos = np.random.normal(loc=media, scale=desviacion, size=n)
    df = pd.DataFrame({'Valor': datos})
    return df


def calcular_prueba_z(df: pd.DataFrame, columna: str, mu_0: float, sigma: float, alpha: float, tipo_prueba: str) -> dict:
    """
    Motor matemático para la prueba Z.
    Retorna un diccionario determinístico con los resultados listos para inyectarse en UI o en el LLM.
    """
    n = len(df[columna])

    if n < 30:
        raise ValueError(
            f"Violación de supuesto: El tamaño de muestra (n={n}) es menor a 30. La prueba Z no es estadísticamente robusta.")

    media_muestral = df[columna].mean()

    error_estandar = sigma / np.sqrt(n)
    z_stat = (media_muestral - mu_0) / error_estandar

    if tipo_prueba == "Bilateral":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_critico_izq = stats.norm.ppf(alpha / 2)
        z_critico_der = stats.norm.ppf(1 - alpha / 2)
        rechazo = p_value < alpha
        region = {"izq": z_critico_izq, "der": z_critico_der}

    elif tipo_prueba == "Cola izquierda":
        p_value = stats.norm.cdf(z_stat)
        z_critico_izq = stats.norm.ppf(alpha)
        rechazo = p_value < alpha
        region = {"izq": z_critico_izq, "der": None}

    elif tipo_prueba == "Cola derecha":
        p_value = 1 - stats.norm.cdf(z_stat)
        z_critico_der = stats.norm.ppf(1 - alpha)
        rechazo = p_value < alpha
        region = {"izq": None, "der": z_critico_der}

    else:
        raise ValueError("Tipo de prueba no reconocido.")

    return {
        "n": int(n),
        "media_muestral": float(media_muestral),
        "mu_0": float(mu_0),
        "sigma": float(sigma),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "tipo_prueba": tipo_prueba,
        "rechaza_H0": bool(rechazo),
        "z_critico": region
    }
