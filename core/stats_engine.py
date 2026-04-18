import numpy as np
import pandas as pd
from scipy import stats
from enum import Enum
from typing import TypedDict, Optional


class TipoPrueba(Enum):
    BILATERAL = "Bilateral"
    COLA_IZQUIERDA = "Cola izquierda"
    COLA_DERECHA = "Cola derecha"


class RegionCritica(TypedDict):
    izq: Optional[float]
    der: Optional[float]


class ZTestResult(TypedDict):
    n: int
    media_muestral: float
    mu_0: float
    sigma: float
    z_stat: float
    p_value: float
    alpha: float
    tipo_prueba: str
    rechaza_H0: bool
    z_critico: RegionCritica


def generar_datos_sinteticos(n: int, media: float, desviacion: float, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame determinístico con distribución normal.
    FIX: Uso de RNG local para no contaminar el estado global de Numpy.
    """
    rng = np.random.default_rng(seed)
    datos = rng.normal(loc=media, scale=desviacion, size=n)
    return pd.DataFrame({'Valor': datos})


def calcular_prueba_z(df: pd.DataFrame, columna: str, mu_0: float, sigma: float, alpha: float, tipo_prueba_str: str) -> ZTestResult:
    """
    Motor matemático con validación estricta (Zero Trust).
    """
    if columna not in df.columns:
        raise ValueError(
            f"Violación de integridad: La columna '{columna}' no existe en el dataset.")

    datos_limpios = df[columna].dropna().to_numpy()
    n = len(datos_limpios)

    if n < 30:
        raise ValueError(
            f"Violación de supuesto: n={n} < 30. Prueba Z no robusta.")
    if sigma <= 0:
        raise ValueError(
            "Violación matemática: Sigma (desviación) debe ser mayor a 0.")
    if not (0 < alpha < 1):
        raise ValueError(
            "Violación probabilística: Alpha debe estar estrictamente entre 0 y 1.")

    try:
        tipo_prueba = TipoPrueba(tipo_prueba_str)
    except ValueError:
        raise ValueError(f"Tipo de prueba no reconocido: {tipo_prueba_str}")

    media_muestral = np.mean(datos_limpios)
    error_estandar = sigma / np.sqrt(n)
    z_stat = (media_muestral - mu_0) / error_estandar

    region: RegionCritica = {"izq": None, "der": None}

    if tipo_prueba == TipoPrueba.BILATERAL:
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        region["izq"] = stats.norm.ppf(alpha / 2)
        region["der"] = stats.norm.ppf(1 - alpha / 2)
        rechazo = p_value < alpha

    elif tipo_prueba == TipoPrueba.COLA_IZQUIERDA:
        p_value = stats.norm.cdf(z_stat)
        region["izq"] = stats.norm.ppf(alpha)
        rechazo = p_value < alpha

    elif tipo_prueba == TipoPrueba.COLA_DERECHA:
        p_value = 1 - stats.norm.cdf(z_stat)
        region["der"] = stats.norm.ppf(1 - alpha)
        rechazo = p_value < alpha

    # Retorno blindado bajo el contrato TypedDict
    return {
        "n": int(n),
        "media_muestral": float(media_muestral),
        "mu_0": float(mu_0),
        "sigma": float(sigma),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "tipo_prueba": tipo_prueba.value,
        "rechaza_H0": bool(rechazo),
        "z_critico": region
    }
