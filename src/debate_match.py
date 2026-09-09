"""Debate multiagente + pipeline completo para UN partido de la Champions.

Uso:  python -m src.debate_match "Real Madrid" "Inter" [--rounds 2]
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import blend, league
from src.agents import debate
from src.agents.schemas import VeredictoJuez
from src.calibration import temperature
from src.ingest import odds, openfootball as of
from src.model import dixon_coles
from src.predict import DATA, OUT, load_cal, train

DEBATES = OUT / "debates"


def _pos_texto(res, team) -> str:
    if res.empty:
        return "arranque de la fase de liga (sin partidos)"
    m = res.rename(columns={"home": "home_team", "away": "away_team"})
    t = league.standings(m)
    row = t[t["team"] == team]
    if row.empty:
        return "aún sin partidos"
    r = row.iloc[0]
    return f"{int(r['pos'])}º, {int(r['pts'])} pts, DG {int(r['dg']):+d} ({r['zona']})"


def build_context(model, df, home, away, cal, market):
    T, g, _ = cal
    P = model.score_matrix(home, away, adv_side=1)
    ph, pd_, pa = model.outcome_probs(P)
    lam, mu = model.rates(home, away, adv_side=1)
    top = model.top_scores(P, 3)
    res = odds.fetch_results(DATA)
    return {
        "home": home, "away": away, "fecha": datetime.now().strftime("%Y-%m-%d"),
        "jornada": "fase de liga",
        "xg_home": float(lam), "xg_away": float(mu),
        "p_home": float(ph), "p_draw": float(pd_), "p_away": float(pa),
        "top_scores": "; ".join(f"{i}-{j} ({p:.1%})" for i, j, p in top),
        "form_home": of.recent_form(df, home), "form_away": of.recent_form(df, away),
        "pos_home": _pos_texto(res, home), "pos_away": _pos_texto(res, away),
        "odds": market,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit('Uso: python -m src.debate_match "Local" "Visita"')
    rounds = 2 if "--rounds" in sys.argv and sys.argv[sys.argv.index("--rounds") + 1] == "2" else 1

    print("[1/3] Entrenamiento + contexto")
    of.download_all(DATA)
    odds.fetch_results(DATA)
    model, df = train()
    cal = load_cal()
    T, g, w = cal
    home = next((t for t in model.teams if args[0].lower() in t.lower()), None)
    away = next((t for t in model.teams if args[1].lower() in t.lower()), None)
    if not home or not away:
        sys.exit("Equipo no encontrado en el modelo (¿liga no disponible?).")

    events = odds.fetch_odds() or []
    market = None
    for e in events:
        if odds.match_team(e["home_team"], model.teams) == home:
            market = odds.implied_probs(e)
            break
    ctx = build_context(model, df, home, away, cal, market)
    print(f"  {home} vs {away} · prior {ctx['p_home']:.0%}/{ctx['p_draw']:.0%}/{ctx['p_away']:.0%}")

    print(f"[2/3] Debate ({rounds} ronda(s), agentes={debate.MODEL}, juez={debate.MODEL_JUEZ})")
    degraded = None
    try:
        resultado = debate.run_debate(ctx, rounds=rounds)
    except Exception as e:
        degraded = str(e)[:120]
        resultado = debate.ResultadoDebate(
            veredicto=VeredictoJuez(delta_log_xg_home=0, delta_log_xg_away=0,
                                    confianza="baja", bajas_confirmadas=False,
                                    factores=[], resumen=f"Degradado: {degraded}"),
            delta_home=0.0, delta_away=0.0)
    if degraded:
        print(f"  AVISO degradado: {degraded}")

    print("[3/3] Predicción final")
    lam, mu = model.rates(home, away, adv_side=1,
                          delta_home=resultado.delta_home, delta_away=resultado.delta_away)
    P = dixon_coles.matrix_from_rates(lam, mu, model.rho)
    mf_cal = temperature.apply(np.array(model.outcome_probs(P)), T)
    prior_cal = temperature.apply(np.array([ctx["p_home"], ctx["p_draw"], ctx["p_away"]]), T)
    mkt = [market["p_home"], market["p_draw"], market["p_away"]] if market else None
    bajas = bool(resultado.veredicto.bajas_confirmadas)
    blended, usa_mkt, override_ap = blend.blend_final(mf_cal, prior_cal, mkt, bajas)
    bh, bd, ba = (float(x) for x in blended)
    P_sc = dixon_coles.matrix_from_rates(lam * g, mu * g, model.rho)
    resumen = blend.score_summary(blend.rescale_matrix(P_sc, [bh, bd, ba]), 3)

    v = resultado.veredicto
    print(f"  Veredicto: Δ home={resultado.delta_home:+.3f} away={resultado.delta_away:+.3f}"
          + ("  ⚑ override no precificado" if override_ap else ""))
    for f in v.factores:
        print(f"    - {f}")
    print(f"  Prior:  {ctx['p_home']:.0%}/{ctx['p_draw']:.0%}/{ctx['p_away']:.0%}")
    if usa_mkt:
        print(f"  Mercado:{mkt[0]:.0%}/{mkt[1]:.0%}/{mkt[2]:.0%}")
    print(f"  FINAL:  {bh:.0%}/{bd:.0%}/{ba:.0%} · xG {resumen['xg_home']:.1f}-{resumen['xg_away']:.1f}"
          f" · marcador {resumen['score_pred']} (top: {resumen['top_scores']})")

    DEBATES.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    ctx["model_final"] = {"p_home": float(mf_cal[0]), "p_draw": float(mf_cal[1]), "p_away": float(mf_cal[2])}
    ctx["blend_weight_market"] = w if usa_mkt else 0.0
    ctx["final"] = {"p_home": bh, "p_draw": bd, "p_away": ba, **resumen}
    debate.guardar(resultado, ctx, DEBATES / f"{home}_{away}_{ts}.json".replace(" ", "_"))


if __name__ == "__main__":
    main()
