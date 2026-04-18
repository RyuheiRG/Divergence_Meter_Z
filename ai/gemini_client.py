import google.generativeai as genai
import json
from core.config import settings

if not settings.GEMINI_API_KEY:
    raise ValueError(
        "Violación de entorno: GEMINI_API_KEY no encontrada en .env")

genai.configure(api_key=settings.GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash')


def consultar_oraculo(resultados_stats: dict) -> str:
    """
    Arquitectura Prompt V2 directo.
    Separa el identity_core (rol de la IA) del scene_state (payload matemático).
    """
    scene_state = json.dumps(resultados_stats, indent=2)

    identity_core = (
        "Eres un oráculo de validación estadística. Operas bajo una política de Zero Trust. "
        "Tu única directiva es analizar el siguiente payload JSON que contiene el estado "
        "determinístico de una prueba Z.\n\n"
        "RESTRICCIONES ESTRICTAS:\n"
        "1. NO recalcules el estadístico Z, el p-value ni las regiones críticas.\n"
        "2. Basa tu conclusión de rechazar o no la hipótesis nula (H0) ÚNICAMENTE en el valor booleano 'rechaza_H0'.\n"
        "3. Redacta una explicación técnica y directa (máximo 4 líneas) sobre el significado de esta decisión "
        "en el contexto de los datos.\n"
        "4. Evalúa si el supuesto de tamaño de muestra (n>=30) se cumple según el JSON.\n\n"
    )

    prompt_ensamblado = f"{identity_core}PAYLOAD JSON:\n{scene_state}"

    try:
        response = model.generate_content(prompt_ensamblado)
        return response.text
    except Exception as e:
        return f"Error crítico en la comunicación con la API: {str(e)}"
