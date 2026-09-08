"""Combinación de pronósticos 1X2 en la probabilidad final.

Pool LINEAL (promedio ponderado): combinar dos pronósticos calibrados por
promedio es notoriamente robusto y difícil de batir. El mercado entra UNA sola
vez, aquí, de forma mecánica — por eso el debate ya NO lleva un agente de
mercado (sería doble conteo); el panel se concentra en noticias que el mercado
aún no refleja.

Por qué no se combina con el ranking de poder (FIFA/Elo): el Dixon-Coles ya se
entrena con resultados, que es la misma información de la que sale el ranking
(correlación ~0.9). Promediarlos reduce varianza marginal pero no agrega señal;
queda como mejora futura vía prior de contracción para selecciones con pocos
datos (hoy se filtran con MIN_MATCHES).

PESO DEL MERCADO (W_MARKET): provisional. El mercado es un benchmark fuerte,
pero el peso óptimo solo se fija con datos. Cada predicción registra las tres
señales (modelo / mercado / mezcla) para ajustarlo tras ~30-40 partidos.
El peso vive en outputs/blend_config.json (versionado) y lo fija por datos
src.fit_market_weight (minimiza RPS sobre los partidos jugados, con tope de
movimiento por corrida). Si no existe el archivo, default 0.65.
"""
import json
from pathlib import Path

import numpy as np

_CFG = Path(__file__).resolve().parent.parent / "outputs" / "blend_config.json"
DEFAULT_W_MARKET = 0.65
# Peso EFECTIVO del debate cuando detecta ausencias confirmadas de titulares.
# Normalmente el debate pesa (1-W_MARKET); ante bajas sube a este valor, sumando
# el efecto del debate (model-prior) por encima de la mezcla normal.
DEFAULT_W_DEBATE_OVERRIDE = 0.50


def _cfg(key: str, default: float) -> float:
    try:
        return float(json.loads(_CFG.read_text(encoding="utf-8"))[key])
    except Exception:
        return default


W_MARKET = _cfg("w_market", DEFAULT_W_MARKET)
W_DEBATE_OVERRIDE = _cfg("w_debate_override", DEFAULT_W_DEBATE_OVERRIDE)


def linear_pool(probs_list, weights) -> np.ndarray:
    P = np.array(probs_list, dtype=float)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    blended = (P * w[:, None]).sum(axis=0)
    return blended / blended.sum()


def blend_market(model_probs, market_probs, w_market: float = W_MARKET):
    """Devuelve (probs_finales, uso_mercado). Si no hay cuotas, pasa el modelo."""
    model_probs = np.array(model_probs, dtype=float)
    if market_probs is None:
        return model_probs / model_probs.sum(), False
    return linear_pool([model_probs, market_probs], [1 - w_market, w_market]), True


def blend_final(model_cal, prior_cal, market_probs, bajas: bool = False):
    """Mezcla final. Normal: (1-W_MARKET)·modelo + W_MARKET·mercado.

    Override CONSCIENTE DEL MERCADO: ante bajas confirmadas, amplifica el efecto
    del debate (d = modelo−prior) PERO solo la fracción que el mercado aún NO
    refleja. Si el mercado ya se movió en la dirección del debate tanto o más que
    el debate (proyección de m=mercado−prior sobre d ≥ |d|), la baja YA está
    precificada → no se aplica override (evita doble conteo, como el caso Portugal
    donde el mercado ya sabía de la baja). Si el mercado no se movió (o se movió en
    contra), la señal es información no precificada → override pleno.

    Devuelve (probs_finales, uso_mercado, override_aplicado).
    """
    blended, usa = blend_market(model_cal, market_probs)
    override_aplicado = False
    if usa and bajas and prior_cal is not None:
        d = np.array(model_cal, float) - np.array(prior_cal, float)
        m = np.array(market_probs, float) - np.array(prior_cal, float)
        dd = float(d @ d)
        if dd > 1e-9:
            residual = float(np.clip(1.0 - (m @ d) / dd, 0.0, 1.0))  # frac no precificada
            extra = max(W_DEBATE_OVERRIDE - (1 - W_MARKET), 0.0)
            if residual > 0.05 and extra > 0:
                b = np.array(blended, float) + extra * residual * d
                b = np.clip(b, 1e-9, None)
                blended = b / b.sum()
                override_aplicado = True
    return blended, usa, override_aplicado


def rescale_matrix(P, target_1x2) -> np.ndarray:
    """Reescala las regiones (local/empate/visita) de la matriz de marcadores
    para que sus sumas igualen target_1x2=(p_home,p_draw,p_away), preservando la
    forma dentro de cada región. Así el marcador refleja la prob FINAL (debate +
    mercado), no solo el modelo+debate."""
    P = np.array(P, dtype=float)
    lower = np.tril(P, -1)            # gana local
    diag = np.diag(np.diag(P))        # empate
    upper = np.triu(P, 1)             # gana visita
    out = np.zeros_like(P)
    for region, target in ((lower, target_1x2[0]), (diag, target_1x2[1]),
                           (upper, target_1x2[2])):
        s = region.sum()
        if s > 0:
            out += region * (target / s)
    return out / out.sum()


def score_summary(P, n_top: int = 3) -> dict:
    """Resumen de marcadores de una matriz: marcador modal, top-n, goles
    esperados y P(el favorito gana por 2+)."""
    P = np.array(P, dtype=float)
    dim = P.shape[0]
    flat = sorted(((P[i, j], i, j) for i in range(dim) for j in range(dim)), reverse=True)
    top = [(i, j, p) for p, i, j in flat[:n_top]]
    idx = np.arange(dim)
    xg_home = float((P.sum(axis=1) * idx).sum())
    xg_away = float((P.sum(axis=0) * idx).sum())
    margen = idx[:, None] - idx[None, :]
    p_home_2 = float(P[margen >= 2].sum())
    p_away_2 = float(P[margen <= -2].sum())
    return {
        "score_pred": f"{top[0][0]}-{top[0][1]}",
        "top_scores": "; ".join(f"{i}-{j} ({p:.1%})" for i, j, p in top),
        "xg_home": round(xg_home, 2), "xg_away": round(xg_away, 2),
        "p_home_by2": round(p_home_2, 4), "p_away_by2": round(p_away_2, 4),
    }
