import os
from dotenv import load_dotenv


class Settings:
    """
    Configuración centralizada y blindada bajo Zero Trust.
    Garantiza que el sistema falle rápido (Fail-Fast) si el entorno no es seguro.
    """

    def __init__(self):
        self.APP_ENV = os.getenv("APP_ENV", "development").lower()

        if self.APP_ENV == "development":
            load_dotenv()

        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        self._validar_estado()

    def _validar_estado(self):
        """Auditoría temprana del estado de las variables de entorno."""
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "⚠️ [ALERTA CRÍTICA]: GEMINI_API_KEY no detectada en el entorno. Abortando inicio.")

        if not isinstance(self.GEMINI_API_KEY, str) or not self.GEMINI_API_KEY.startswith("AQ."):
            raise ValueError(
                "⚠️ [ALERTA CRÍTICA]: El formato de la GEMINI_API_KEY es inválido o corrupto.")

    def __str__(self):
        """
        Mitigación de fuga de datos en memoria. 
        Si alguien intenta imprimir la configuración por error (ej. print(settings)),
        la llave real nunca se volcará en los logs.
        """
        return f"Settings(APP_ENV='{self.APP_ENV}', API_KEY='***MASKED***')"


settings = Settings()
