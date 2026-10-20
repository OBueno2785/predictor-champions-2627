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

# fit del peso del MARCADOR-mercado (volumen de goles del marcador)
SCORE_SHRINK_K = 20
W_SCORE_MOVE_CAP = 0.15
MIN_FOR_WSCORE = 5


def _played_covered(model):
    """Partidos CL 26-27 jugados donde el modelo cubre ambos equipos."""
    res = odds.results_as_training(DATA)
    filas = []
    for r in res.itertuples():
        if r.home_team in model.attack and r.away_team in model.attack:
            filas.append(r)
    return res, filas


def _usable_market_results(model):
    """(th, ta, [ph,pd,pa]_mercado, hs, as) de partidos jugados con cuotas 1X2
    guardadas en outputs/debates y resultado real en data/cl_results.json."""
    store = DATA / "cl_results.json"
    res = {(r["home"], r["away"]): (r["home_score"], r["away_score"])
           for r in json.loads(store.read_text(encoding="utf-8"))} if store.exists() else {}

    def _result_for(home, away):
        if (home, away) in res:
            return res[(home, away)]
        hm, am = odds.match_team(home, model.teams), odds.match_team(away, model.teams)
        for (h, a), s in res.items():
            if odds.match_team(h, model.teams) == hm and odds.match_team(a, model.teams) == am:
                return s
        return None

    uniq = {}
    for f in sorted((OUT / "debates").glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d.get("odds") or " vs " not in d.get("partido", ""):
            continue
        home, away = d["partido"].split(" vs ", 1)
        th, ta = odds.match_team(home, model.teams), odds.match_team(away, model.teams)
        if not (th and ta and th in model.attack and ta in model.attack):
            continue  # el endpoint "modelo" necesita cobertura de ambos
        sc = _result_for(home, away)
        if sc is None:
            continue
        o = d["odds"]
        uniq[(th, ta)] = ([o["p_home"], o["p_draw"], o["p_away"]], int(sc[0]), int(sc[1]))
    return [(th, ta, mk, hs, a) for (th, ta), (mk, hs, a) in uniq.items()]


def fit_score_weight(model, goals_mult, w_market):
    """Ajusta w_score: peso del mercado en el VOLUMEN de goles del marcador.
    Minimiza el NLL del marcador real bajo la matriz de goles con ritmos
    mezclados (1-w)·modelo + w·mercado. Contracción hacia w_market y tope de
    movimiento (muestra chica). Devuelve (w_final, n, w_opt|None)."""
    from scipy.optimize import minimize_scalar
    from src.predict import _market_rates

    data = _usable_market_results(model)
    n = len(data)
    if n < MIN_FOR_WSCORE:
        return w_market, n, None
    rho = model.rho
    pts = []
    for th, ta, mk, hs, a in data:
        lam_md, mu_md = model.rates(th, ta, adv_side=1)
        lam_md, mu_md = lam_md * goals_mult, mu_md * goals_mult
        lam_m, mu_m = _market_rates(np.array(mk, float), None, rho)
        pts.append((lam_md, mu_md, lam_m, mu_m, hs, a))

    def nll(w):
        s = 0.0
        for lam_md, mu_md, lam_m, mu_m, hs, a in pts:
            P = dixon_coles.matrix_from_rates((1 - w) * lam_md + w * lam_m,
                                              (1 - w) * mu_md + w * mu_m, rho)
            dim = P.shape[0]
            p = P[hs, a] if hs < dim and a < dim else 1e-12
            s -= np.log(max(float(p), 1e-12))
        return s

    w_opt = float(minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded").x)
    shrink = n / (n + SCORE_SHRINK_K)
    w_shr = w_market + shrink * (w_opt - w_market)
    w_final = float(np.clip(w_shr, max(0.0, w_market - W_SCORE_MOVE_CAP),
                            min(1.0, w_market + W_SCORE_MOVE_CAP)))
    return w_final, n, w_opt


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

    # w_market (1X2): default apoyado en el mercado hasta tener muestra
    w = W_MARKET_DEFAULT
    print(f"  w_market: {w:.2f} (default apoyado en el mercado; se ajustará con ≥{MIN_FOR_W} partidos)")

    # w_score: peso del mercado en el VOLUMEN de goles del marcador (fit propio)
    w_score, n_ws, w_opt = fit_score_weight(model, goals_mult, w)
    if w_opt is None:
        print(f"  w_score: {w_score:.2f} (= w_market; muestra {n_ws} < {MIN_FOR_WSCORE})")
    else:
        print(f"  w_score: óptimo {w_opt:.2f} → {w_score:.3f} "
              f"(contracción n/(n+{SCORE_SHRINK_K}), n={n_ws}, tope ±{W_SCORE_MOVE_CAP})")

    (OUT / "blend_config.json").write_text(
        json.dumps({"w_market": w, "w_score": w_score}, indent=1), encoding="utf-8")
    print(f"\nEscrito calibration.json (T={T:.3f}, goles×{goals_mult:.3f}) y "
          f"blend_config.json (w_market={w:.2f}, w_score={w_score:.3f})")


if __name__ == "__main__":
    main()
