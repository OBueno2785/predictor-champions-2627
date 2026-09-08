"""Orquestación del debate multiagente sobre el prior del Dixon-Coles.

Transporte: Claude Code en modo headless (`claude -p`), autenticado con la
suscripción del usuario — NO requiere ANTHROPIC_API_KEY ni paga por token
(consume cuota del plan). Los agentes de noticias usan la búsqueda web
integrada de Claude Code (WebSearch).

Flujo: ronda 1 (agentes en paralelo) → ronda 2 (réplicas) → Juez (JSON
validado con Pydantic, deltas acotados en log-xG) → matriz de marcadores
recalculada con los ritmos ajustados.
"""
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv

from src.agents import prompts
from src.agents.schemas import DELTA_MAX, VeredictoJuez

load_dotenv()

# sonnet para los debatientes (cuota ~5x menor), opus para el Juez
MODEL = os.environ.get("DEBATE_MODEL", "sonnet")
MODEL_JUEZ = os.environ.get("DEBATE_MODEL_JUEZ", "opus")
TIMEOUT_S = 300   # una llamada colgada degrada en 5 min, no 10
MAX_WORKERS = 4


class CuotaAgotada(RuntimeError):
    """Límite de sesión de la suscripción de Claude Code (HTTP 429)."""

JSON_JUEZ = """

Responde ÚNICAMENTE con un objeto JSON válido (sin markdown, sin texto extra):
{"delta_log_xg_home": <número>, "delta_log_xg_away": <número>,
 "confianza": "alta"|"media"|"baja", "bajas_confirmadas": true|false,
 "factores": [<strings>], "resumen": <string>}"""


@dataclass
class ResultadoDebate:
    veredicto: VeredictoJuez
    delta_home: float
    delta_away: float
    rondas: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


def _claude_exe() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("CLI 'claude' no encontrado en PATH. Instalar Claude Code "
                           "o agregarlo al PATH para ejecutar el debate.")
    return exe


def _call_claude(prompt: str, allow_web: bool = False, max_turns: int = 12,
                 model: str = MODEL) -> tuple[str, dict]:
    cmd = [_claude_exe(), "-p", "--output-format", "json",
           "--model", model, "--max-turns", str(max_turns)]
    if allow_web:
        cmd += ["--allowedTools", "WebSearch,WebFetch"]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                       encoding="utf-8", timeout=TIMEOUT_S)
    try:
        data = json.loads(r.stdout)
    except (json.JSONDecodeError, TypeError):
        data = {}
    if data.get("api_error_status") == 429:
        raise CuotaAgotada(data.get("result", "límite de sesión alcanzado"))
    if r.returncode != 0:
        raise RuntimeError(f"claude -p falló ({r.returncode}): "
                           f"{(r.stderr or r.stdout)[:500]}")
    if data.get("is_error"):
        raise RuntimeError(f"claude -p devolvió error: {data.get('result', '')[:500]}")
    return data.get("result", ""), data


def _acumular_usage(total: dict, meta: dict) -> None:
    total["llamadas"] = total.get("llamadas", 0) + 1
    total["duracion_ms"] = total.get("duracion_ms", 0) + meta.get("duration_ms", 0)
    total["turnos"] = total.get("turnos", 0) + meta.get("num_turns", 0)


def render_contexto(ctx: dict) -> str:
    """Contexto compartido del partido para todos los agentes."""
    lines = [
        f"# Partido: {ctx['home']} vs {ctx['away']}",
        f"{ctx['group']} · {ctx['fecha']} · {ctx['estadio']} ({ctx['sede_pais']})",
        f"Ventaja de localía aplicada a: {ctx['ventaja'] or 'nadie (cancha neutral)'}",
        "",
        "## Prior del modelo Dixon-Coles",
        f"- xG esperado: {ctx['home']} {ctx['xg_home']:.2f} — {ctx['xg_away']:.2f} {ctx['away']}",
        f"- Probabilidades: local {ctx['p_home']:.1%} / empate {ctx['p_draw']:.1%} / visita {ctx['p_away']:.1%}",
        f"- Marcadores más probables: {ctx['top_scores']}",
        "",
        f"## Forma reciente {ctx['home']} (últimos 5)",
        ctx["form_home"],
        f"  promedio: {ctx.get('form5_home', '—')}",
        "",
        f"## Forma reciente {ctx['away']} (últimos 5)",
        ctx["form_away"],
        f"  promedio: {ctx.get('form5_away', '—')}",
        "",
        "## Situación de grupo (incentivos)",
        f"- {ctx['home']}: posición {ctx.get('pos_home', '—')} · {ctx.get('outlook_home', '—')}",
        f"- {ctx['away']}: posición {ctx.get('pos_away', '—')} · {ctx.get('outlook_away', '—')}",
    ]
    if ctx.get("odds"):
        o = ctx["odds"]
        lines += ["", "## Mercado (probabilidades implícitas sin margen, "
                  f"mediana de {o['n_bookies']} casas)",
                  f"- local {o['p_home']:.1%} / empate {o['p_draw']:.1%} / visita {o['p_away']:.1%}"]
    else:
        lines += ["", "## Mercado", "Sin datos de cuotas en esta ejecución."]
    return "\n".join(lines)


def _parse_veredicto(texto: str) -> VeredictoJuez:
    s = texto.strip()
    s = re.sub(r"^```(json)?|```$", "", s, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError(f"El Juez no devolvió JSON: {texto[:300]}")
    return VeredictoJuez(**json.loads(m.group(0)))


def run_debate(ctx: dict, rounds: int = 2) -> ResultadoDebate:
    contexto = render_contexto(ctx)
    usage_total = {}
    tarea_r1 = "Emite tu análisis inicial del partido según tu rol."

    # El mercado se incorpora mecánicamente después del debate (src.blend),
    # por eso NO hay agente de Mercado aquí: sería doble conteo. El panel se
    # enfoca en información que el mercado aún no refleja.
    roles = [
        ("Estadístico", prompts.ESTADISTICO, False),
        ("Plantel/Noticias", prompts.PLANTEL, True),
        ("Sentimiento", prompts.SENTIMIENTO, True),
    ]

    rondas = {}

    def _ronda1(rol):
        nombre, system, web = rol
        texto, meta = _call_claude(
            f"{system}\n\n{contexto}\n\n{tarea_r1}",
            allow_web=web, max_turns=12 if web else 4)
        return nombre, texto, meta

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        r1 = list(ex.map(_ronda1, roles))
    for _, _, meta in r1:
        _acumular_usage(usage_total, meta)
    rondas["ronda1"] = {n: t for n, t, _ in r1}

    posiciones_finales = rondas["ronda1"]
    if rounds >= 2:
        def _ronda2(rol):
            nombre, system, web = rol
            otras = "\n\n".join(f"### {n}\n{t}" for n, t in rondas["ronda1"].items()
                                if n != nombre)
            prompt = (f"{system}\n\n{contexto}\n\n"
                      f"## Tu análisis inicial\n{rondas['ronda1'][nombre]}\n\n"
                      + prompts.REPLICA.format(posiciones=otras))
            texto, meta = _call_claude(prompt, allow_web=web,
                                       max_turns=8 if web else 4)
            return nombre, texto, meta

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            r2 = list(ex.map(_ronda2, roles))
        for _, _, meta in r2:
            _acumular_usage(usage_total, meta)
        rondas["ronda2"] = {n: t for n, t, _ in r2}
        posiciones_finales = rondas["ronda2"]

    panel = "\n\n".join(f"### {n}\n{t}" for n, t in posiciones_finales.items())
    texto, meta = _call_claude(
        f"{prompts.JUEZ}\n\n{contexto}\n\n## Posiciones finales del panel\n\n{panel}"
        f"\n\nEmite tu veredicto.{JSON_JUEZ}",
        allow_web=False, max_turns=4, model=MODEL_JUEZ)
    _acumular_usage(usage_total, meta)
    veredicto = _parse_veredicto(texto)

    dh = float(np.clip(veredicto.delta_log_xg_home, -DELTA_MAX, DELTA_MAX))
    da = float(np.clip(veredicto.delta_log_xg_away, -DELTA_MAX, DELTA_MAX))
    return ResultadoDebate(veredicto=veredicto, delta_home=dh, delta_away=da,
                           rondas=rondas, usage=usage_total)


def guardar(resultado: ResultadoDebate, ctx: dict, dest) -> None:
    dest.write_text(json.dumps({
        "partido": f"{ctx['home']} vs {ctx['away']}",
        "fecha": str(ctx["fecha"]),
        "prior": {k: ctx[k] for k in ["xg_home", "xg_away", "p_home", "p_draw", "p_away"]},
        "odds": ctx.get("odds"),
        "model_final": ctx.get("model_final"),
        "blend_weight_market": ctx.get("blend_weight_market"),
        "final": ctx.get("final"),
        "veredicto": resultado.veredicto.model_dump(),
        "delta_aplicado": {"home": resultado.delta_home, "away": resultado.delta_away},
        "rondas": resultado.rondas,
        "usage": resultado.usage,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
