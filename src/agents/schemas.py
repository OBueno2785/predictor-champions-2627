"""Esquema del veredicto del Juez (salida estructurada con Pydantic)."""
from typing import Literal

from pydantic import BaseModel, Field

DELTA_MAX = 0.25  # tope en log-xG por lado (~±28% en ritmo de gol, ~±8pp en 1X2)


class VeredictoJuez(BaseModel):
    delta_log_xg_home: float = Field(
        description="Ajuste al log del ritmo de gol del local. 0 = sin ajuste. "
                    f"Rango permitido [-{DELTA_MAX}, +{DELTA_MAX}].")
    delta_log_xg_away: float = Field(
        description="Ajuste al log del ritmo de gol de la visita. 0 = sin ajuste. "
                    f"Rango permitido [-{DELTA_MAX}, +{DELTA_MAX}].")
    confianza: Literal["alta", "media", "baja"]
    bajas_confirmadas: bool = Field(
        default=False,
        description="true SOLO si el ajuste se debe a lesiones, suspensiones o "
                    "ausencias CONFIRMADAS de jugadores TITULARES que el modelo no "
                    "ve (no rumores, no dudas, no factores blandos). Activa que el "
                    "debate pese más que su 20% habitual en la predicción final.")
    factores: list[str] = Field(
        description="Cada ajuste debe citar el factor concreto que lo justifica "
                    "(lesión, cuota, conflicto, etc.). Vacío si delta=0.")
    resumen: str = Field(description="Síntesis del debate en 2-3 frases.")
