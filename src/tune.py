"""Ajusta los pesos del pipeline con los partidos de CL 26-27 ya jugados.

Con muestra chica (arranque de la fase de liga) usa contracción fuerte y
defaults apoyados en el mercado. Escribe outputs/calibration.json (temperature,
goals_mult) y outputs/blend_config.json (w_market).

Uso:  python -m src.tune
"""
import json
import sys
from pathlib import Path

import numpy as np

from src.calibration import metrics, temperature
from src.ingest import odds, openfootball as of
from src.model import dixon_coles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SHRINK_K = 30
GOALS_CAP = (0.85, 1.30)
W_MARKET_DEFAULT = 0.75
MIN_FOR_T = 20
MIN_FOR_W = 15


def _played_covered(model):
    """Partidos CL 26-27 jugados donde el modelo cubre ambos equipos."""
    res = odds.results_as_training(DATA)
    filas = []
    for r in res.itertuples():
        if r.home_team in model.attack and r.away_team in model.attack:
            filas.append(r)
    return res, filas


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(exist_ok=True)
    of.download_all(DATA)
    base = of.load_training(DATA)
    res = odds.results_as_training(DATA)
    df = base if res.empty else \
        __import__("pandas").concat([base, res], ignore_index=True)
    model = dixon_coles.fit(df)

    cov = [r for r in res.itertuples()
           if r.home_team in model.attack and r.away_team in model.attack] if not res.empty else []
    n = len(cov)
    print(f"CL 26-27 jugados: {0 if res.empty else len(res)} | con modelo (ambos equipos): {n}")

    # goals_mult (lo único con algo de señal)
    goals_mult = 1.0
    if n >= 3:
        pred_tot = real_tot = 0.0
        for r in cov:
            lam, mu = model.rates(r.home_team, r.away_team, adv_side=1)
            pred_tot += lam + mu
            real_tot += r.home_score + r.away_score
        ratio = real_tot / pred_tot
        shrink = n / (n + SHRINK_K)
        goals_mult = float(np.clip(1 + shrink * (ratio - 1), *GOALS_CAP))
        print(f"  goles: ratio real/predicho ×{ratio:.2f} → goals_mult ×{goals_mult:.3f} "
              f"(contracción {shrink:.2f})")

    # temperature: solo con muestra suficiente
    T = 1.0
    if n >= MIN_FOR_T:
        probs = np.array([model.outcome_probs(model.score_matrix(r.home_team, r.away_team, adv_side=1))
                          for r in cov])
        outs = np.array([0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
                         for r in cov])
        T = temperature.fit(probs, outs)
        print(f"  temperature ajustada: {T:.3f}")
    else:
        print(f"  temperature: 1.0 (muestra {n} < {MIN_FOR_T}, aún no se ajusta)")

    (OUT / "calibration.json").write_text(json.dumps(
        {"temperature": T, "goals_mult": goals_mult, "n_cl": n}, indent=1), encoding="utf-8")

    # w_market: default apoyado en el mercado hasta tener muestra
    w = W_MARKET_DEFAULT
    print(f"  w_market: {w:.2f} (default apoyado en el mercado; se ajustará con ≥{MIN_FOR_W} partidos)")
    (OUT / "blend_config.json").write_text(json.dumps({"w_market": w}, indent=1), encoding="utf-8")
    print(f"\nEscrito calibration.json (T={T:.3f}, goles×{goals_mult:.3f}) y blend_config.json (w={w:.2f})")


if __name__ == "__main__":
    main()
