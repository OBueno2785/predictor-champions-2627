"""Entrena el Dixon-Coles con datos de clubes (ligas + Champions) y predice cruces.

Uso:
  python -m src.predict                      # ranking de fuerza + sanity check
  python -m src.predict "Real Madrid" "Bayern"   # predice un cruce (subcadena)
"""
import sys
from pathlib import Path

import numpy as np

from src.ingest import openfootball as of
from src.model import dixon_coles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def ingest(offline: bool = False) -> None:
    if not offline:
        of.download_all(DATA)


def train():
    df = of.load_training(DATA)
    if df.empty:
        sys.exit("Sin datos. Ejecuta la ingesta (quita --offline).")
    model = dixon_coles.fit(df)
    return model, df


def find_team(model, query: str) -> str:
    q = query.lower()
    exact = [t for t in model.teams if t.lower() == q]
    if exact:
        return exact[0]
    subs = [t for t in model.teams if q in t.lower()]
    if len(subs) == 1:
        return subs[0]
    if not subs:
        sys.exit(f"No hay equipo que contenga '{query}'. Usa 'python -m src.predict' para ver el ranking.")
    sys.exit(f"'{query}' es ambiguo: {subs[:8]}")


def predict_match(model, home: str, away: str, neutral: bool = False) -> dict:
    side = 0 if neutral else 1
    P = model.score_matrix(home, away, adv_side=side)
    ph, pd_, pa = model.outcome_probs(P)
    lam, mu = model.rates(home, away, adv_side=side)
    top = model.top_scores(P, 3)
    return {
        "home": home, "away": away, "xg_home": float(lam), "xg_away": float(mu),
        "p_home": float(ph), "p_draw": float(pd_), "p_away": float(pa),
        "top_scores": "; ".join(f"{i}-{j} ({p:.1%})" for i, j, p in top),
        "score_pred": f"{top[0][0]}-{top[0][1]}",
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    offline = "--offline" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    print("[1/2] Ingesta + entrenamiento (clubes: ligas + Champions)")
    ingest(offline)
    model, df = train()
    print(f"  {model.n_matches} partidos ({df['date'].min().date()} a {df['date'].max().date()}), "
          f"{len(model.teams)} equipos · ventaja localía={model.home_adv:.3f} · rho={model.rho:.4f}")

    if len(args) >= 2:
        home = find_team(model, args[0])
        away = find_team(model, args[1])
        p = predict_match(model, home, away, neutral="--neutral" in sys.argv)
        print(f"\n{p['home']}  vs  {p['away']}")
        print(f"  1 {p['p_home']:.0%} · X {p['p_draw']:.0%} · 2 {p['p_away']:.0%}")
        print(f"  xG {p['xg_home']:.2f}-{p['xg_away']:.2f} · marcador {p['score_pred']}")
        print(f"  top: {p['top_scores']}")
        return

    print("\n[2/2] Ranking de fuerza (ataque+defensa) — sanity check:")
    rk = model.strength_ranking().head(20)
    for r in rk.itertuples():
        print(f"  {r.Index + 1:2d}. {r.team:<32} {r.strength:+.3f}")


if __name__ == "__main__":
    main()
