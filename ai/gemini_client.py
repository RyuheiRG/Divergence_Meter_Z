import json
import google.generativeai as genai
from core.config import settings

if not settings.GEMINI_API_KEY:
    raise ValueError(
        "Violación de entorno: GEMINI_API_KEY no encontrada en .env")

genai.configure(api_key=settings.GEMINI_API_KEY)


class GeminiOracle:
    """
    Cliente de IA con arquitectura Zero Trust.
    Implementa validación pre/post, fallbacks de modelos y separación de responsabilidades.
    """
    MODEL_FALLBACKS = ['gemini-1.5-flash', 'gemini-1.5-pro',
                       'gemini-pro', 'gemini-2.5-flash', 'gemini-2.5-pro']

    PROMPT_V2 = (
        "Eres un oráculo de validación estadística. Operas bajo una política de Zero Trust. "
        "Tu única directiva es analizar el siguiente payload JSON que contiene el estado "
        "determinístico de una prueba Z.\n\n"
        "RESTRICCIONES ESTRICTAS:\n"
        "1. NO recalcules el estadístico Z, el p-value ni las regiones críticas.\n"
        "2. Basa tu conclusión de rechazar o no la hipótesis nula (H0) ÚNICAMENTE en el valor booleano 'rechaza_H0'.\n"
        "3. Redacta una explicación técnica y directa (máximo 4 líneas) sobre el significado de esta decisión.\n"
        "4. Evalúa si el supuesto de tamaño de muestra (n>=30) se cumple.\n\n"
        "PAYLOAD JSON:\n{payload}"
    )

    FORBIDDEN_WORDS = ["calculé", "recalculé",
                       "nuevo valor", "error de cálculo", "mi cálculo"]

    @staticmethod
    def _sanitizar_payload(resultados_stats: dict) -> str:
        """
        Pre-validación (LLM Guard Layer): Mitiga el Prompt Injection Indirecto.
        Se crea un whitelist estricto. Solo se serializan las llaves esperadas y se 
        fuerza su tipo de dato (casting). Cualquier inyección en el dict original es ignorada.
        """
        safe_payload = {
            "n": int(resultados_stats.get("n", 0)),
            "media_muestral": float(resultados_stats.get("media_muestral", 0.0)),
            "mu_0": float(resultados_stats.get("mu_0", 0.0)),
            "sigma": float(resultados_stats.get("sigma", 0.0)),
            "z_stat": float(resultados_stats.get("z_stat", 0.0)),
            "p_value": float(resultados_stats.get("p_value", 0.0)),
            "alpha": float(resultados_stats.get("alpha", 0.0)),
            "tipo_prueba": str(resultados_stats.get("tipo_prueba", "")),
            "rechaza_H0": bool(resultados_stats.get("rechaza_H0", False))
        }
        return json.dumps(safe_payload, indent=2)

    @classmethod
    def _post_validar_respuesta(cls, respuesta: str) -> str:
        """
        Post-validación: Verifica que el modelo no haya alucinado o desobedecido las 
        reglas de no recalcular. Si detecta insubordinación, bloquea la salida.
        """
        respuesta_lower = respuesta.lower()
        for word in cls.FORBIDDEN_WORDS:
            if word in respuesta_lower:
                return f"⚠️ [ALERTA ZERO TRUST]: El oráculo LLM intentó violar las restricciones matemáticas (detectada palabra prohibida: '{word}'). Análisis semántico abortado para mantener la integridad de los datos."
        return respuesta

    @classmethod
    def consultar(cls, resultados_stats: dict) -> str:
        """
        Ejecuta la inferencia implementando el fallback dinámico de modelos.
        """
        payload_seguro = cls._sanitizar_payload(resultados_stats)
        prompt_ensamblado = cls.PROMPT_V2.format(payload=payload_seguro)

        last_error = None

        for model_name in cls.MODEL_FALLBACKS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt_ensamblado)

                return cls._post_validar_respuesta(response.text)

            except Exception as e:
                last_error = e
                continue

        return f"Error crítico en la comunicación: Ningún modelo de IA respondió. Último error: {str(last_error)}"


def consultar_oraculo(resultados_stats: dict) -> str:
    return GeminiOracle.consultar(resultados_stats)
