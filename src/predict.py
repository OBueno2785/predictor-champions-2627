"""Pipeline completo del predictor de la Champions (fase de liga).

Entrena el Dixon-Coles (ligas + Champions + resultados CL 26-27), calibra
(temperature, goles), mezcla con el mercado (cuotas sin margen) y produce el
marcador con goles calibrados. Los equipos sin datos de liga (Porto, AEK...) se
predicen 100% con el mercado.

Uso:
  python -m src.predict                     # predicciones de próximos partidos + tabla
  python -m src.predict "Real Madrid" "Inter"   # un cruce puntual
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import blend, league
from src.calibration import temperature
from src.ingest import odds, openfootball as of
from src.model import dixon_coles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def ingest(offline: bool = False) -> None:
    if not offline:
        of.download_all(DATA)
        odds.fetch_results(DATA)


def train():
    base = of.load_training(DATA)
    res = odds.results_as_training(DATA)
    df = base if res.empty else pd.concat([base, res], ignore_index=True)
    return dixon_coles.fit(df), df


def load_cal() -> tuple[float, float, float]:
    T = g = 1.0
    w = 0.75
    if (OUT / "calibration.json").exists():
        c = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
        T, g = c.get("temperature", 1.0), c.get("goals_mult", 1.0)
    if (OUT / "blend_config.json").exists():
        w = json.loads((OUT / "blend_config.json").read_text(encoding="utf-8")).get("w_market", 0.75)
    return T, g, w


def _outc(P):
    return np.array([np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()])


def _p_over(lam, mu, rho, line):
    P = dixon_coles.matrix_from_rates(lam, mu, rho)
    dim = P.shape[0]
    return sum(P[i, j] for i in range(dim) for j in range(dim) if i + j > line)


def _market_rates(target_1x2, total, rho):
    """Ritmos (lam, mu) implícitos en el mercado. Ajusta el 1X2 (ganador) y, si
    hay mercado de totales, también P(Over línea) → así el marcador hereda el
    VOLUMEN de goles de las casas, no solo el ganador."""
    th, _, ta = target_1x2

    def loss(x):
        lam, mu = np.exp(x)
        ph, _, pa = _outc(dixon_coles.matrix_from_rates(lam, mu, rho))
        e = (ph - th) ** 2 + (pa - ta) ** 2
        if total is not None:
            e += (_p_over(lam, mu, rho, total["line"]) - total["p_over"]) ** 2
        return e

    r = minimize(loss, [np.log(1.4), np.log(1.1)], method="Nelder-Mead")
    return np.exp(r.x)


# compat: ajuste de ritmos solo con 1X2 (sin totales)
def _implied_rates(target, rho):
    return _market_rates(target, None, rho)


def predict_match(model, home, away, market, cal, neutral=False, market_total=None):
    T, g, w = cal
    side = 0 if neutral else 1
    cubierto = home in model.attack and away in model.attack
    if cubierto:
        P = model.score_matrix(home, away, adv_side=side)
        mf = temperature.apply(_outc(P), T)
        lam, mu = model.rates(home, away, adv_side=side)
        lam_md, mu_md = lam * g, mu * g
        if market is not None:
            mkt_1x2 = np.array([market["p_home"], market["p_draw"], market["p_away"]])
            fin = (1 - w) * mf + w * mkt_1x2
            fin = fin / fin.sum()
            fuente = "modelo+mercado"
            # marcador con ritmos mezclados (volumen de goles del mercado al mismo w)
            lam_m, mu_m = _market_rates(mkt_1x2, market_total, model.rho)
            P_sc = dixon_coles.matrix_from_rates((1 - w) * lam_md + w * lam_m,
                                                 (1 - w) * mu_md + w * mu_m, model.rho)
        else:
            fin = mf
            fuente = "modelo"
            P_sc = dixon_coles.matrix_from_rates(lam_md, mu_md, model.rho)
    elif market is not None:
        fin = np.array([market["p_home"], market["p_draw"], market["p_away"]])
        lam, mu = _market_rates(fin, market_total, model.rho)
        P_sc = dixon_coles.matrix_from_rates(lam, mu, model.rho)
        fuente = "mercado"
    else:
        return None
    s = blend.score_summary(blend.rescale_matrix(P_sc, fin), 3)
    return {"home": home, "away": away, "p_home": float(fin[0]), "p_draw": float(fin[1]),
            "p_away": float(fin[2]), "fuente": fuente, **s}


def predict_upcoming(model, cal) -> list:
    events = odds.fetch_odds() or []
    rows = []
    for e in events:
        th = odds.match_team(e["home_team"], model.teams)
        ta = odds.match_team(e["away_team"], model.teams)
        mk = odds.implied_probs(e)
        mt = odds.implied_total(e)
        h = th or e["home_team"]
        a = ta or e["away_team"]
        # si no está en el modelo, predecir solo con mercado
        p = predict_match(model, th if th else "?", ta if ta else "?", mk, cal, market_total=mt) \
            if (th and ta) else (predict_match(model, "?", "?", mk, cal, market_total=mt) if mk else None)
        if p is None:
            continue
        p["home"], p["away"] = e["home_team"], e["away_team"]
        p["commence"] = e["commence_time"]
        rows.append(p)
    return sorted(rows, key=lambda r: r["commence"])


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    offline = "--offline" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    print("[1/3] Ingesta + entrenamiento")
    ingest(offline)
    model, df = train()
    cal = load_cal()
    print(f"  {model.n_matches} partidos, {len(model.teams)} equipos · "
          f"T={cal[0]:.3f} goles×{cal[1]:.3f} w_mercado={cal[2]:.2f}")

    if len(args) >= 2:
        home = next((t for t in model.teams if args[0].lower() in t.lower()), None)
        away = next((t for t in model.teams if args[1].lower() in t.lower()), None)
        if not home or not away:
            sys.exit("Equipo no encontrado en el modelo.")
        p = predict_match(model, home, away, None, cal, neutral="--neutral" in sys.argv)
        print(f"\n{p['home']} vs {p['away']}: {p['p_home']:.0%}/{p['p_draw']:.0%}/{p['p_away']:.0%} "
              f"· xG {p['xg_home']:.1f}-{p['xg_away']:.1f} · {p['score_pred']}")
        return

    print("[2/3] Predicciones de próximos partidos (modelo + mercado)")
    ups = predict_upcoming(model, cal)
    for p in ups:
        print(f"  {p['home']} vs {p['away']}: {p['p_home']:.0%}/{p['p_draw']:.0%}/{p['p_away']:.0%} "
              f"· xG {p['xg_home']:.1f}-{p['xg_away']:.1f} · {p['score_pred']} [{p['fuente']}]")

    print("[3/3] Tabla (formato suizo) con resultados CL 26-27")
    res = odds.fetch_results(DATA)
    if not res.empty:
        m = res.rename(columns={"home": "home_team", "away": "away_team"})
        (ROOT / "TABLA.md").write_text(
            league.standings_markdown(m, "Champions 2026-27 — fase de liga"), encoding="utf-8")
        t = league.standings(m)
        print(f"  {len(res)} partidos jugados; tabla escrita en TABLA.md (líder: "
              f"{t.iloc[0]['team']}, {int(t.iloc[0]['pts'])} pts)")


if __name__ == "__main__":
    main()
