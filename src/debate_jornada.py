"""Debate multiagente + pipeline completo para TODOS los partidos de una fecha.

Versión por lotes de src.debate_match. Los deltas del debate se aplican sobre los
ritmos del modelo (equipos cubiertos) o sobre los implícitos del mercado (equipos
fuera del pool de ligas). El marcador usa ritmos mezclados modelo+mercado al peso
fiteado w_score y una matriz coherente con el 1X2 final. Solo debate partidos que
aún no empezaron (hora de America/Lima).

Salidas: outputs/debates/<local>_<visita>_<ts>.json (cada debate) y
outputs/jornadas/<fecha>.json (resumen de la fecha, se reescribe partido a partido).

Uso:  python -m src.debate_jornada [today|tomorrow|YYYY-MM-DD]
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from src import blend, league
from src.agents import debate
from src.agents.schemas import VeredictoJuez
from src.calibration import temperature
from src.ingest import odds, openfootball as of
from src.model import dixon_coles
from src.predict import DATA, OUT, _implied_rates, _market_rates, _outc, load_cal, train

DEBATES = OUT / "debates"
JORNADAS = OUT / "jornadas"
LIMA = ZoneInfo("America/Lima")
ROUNDS = 1


def _fecha(arg: str) -> str:
    ahora = datetime.now(LIMA)
    if arg == "today":
        return ahora.strftime("%Y-%m-%d")
    if arg == "tomorrow":
        return (ahora + timedelta(days=1)).strftime("%Y-%m-%d")
    return arg


def _hora(e) -> datetime:
    return datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(LIMA)


def _pos_texto(res, team) -> str:
    if team is None or res.empty:
        return "—"
    t = league.standings(res.rename(columns={"home": "home_team", "away": "away_team"}))
    row = t[t["team"] == team]
    if row.empty:
        return "aún sin partidos"
    r = row.iloc[0]
    return f"{int(r['pos'])}º, {int(r['pts'])} pts, DG {int(r['dg']):+d} ({r['zona']})"


def _top_scores(P, n=3) -> str:
    dim = P.shape[0]
    flat = sorted(((P[i, j], i, j) for i in range(dim) for j in range(dim)), reverse=True)
    return "; ".join(f"{i}-{j} ({p:.1%})" for p, i, j in flat[:n])


def _degradado(motivo: str):
    return debate.ResultadoDebate(
        veredicto=VeredictoJuez(delta_log_xg_home=0, delta_log_xg_away=0, confianza="baja",
                                bajas_confirmadas=False, factores=[], resumen=motivo),
        delta_home=0.0, delta_away=0.0)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    fecha = _fecha(sys.argv[1] if len(sys.argv) > 1 else "today")
    DEBATES.mkdir(parents=True, exist_ok=True)
    JORNADAS.mkdir(parents=True, exist_ok=True)
    resumen_path = JORNADAS / f"{fecha}.json"

    print(f"[setup] ingesta + entrenamiento (fecha={fecha})")
    of.download_all(DATA)
    res = odds.fetch_results(DATA)
    model, df = train()
    T, g, w, w_score = load_cal()
    print(f"[setup] {model.n_matches} partidos, {len(model.teams)} equipos · "
          f"T={T:.3f} goles x{g:.3f} w={w:.2f} w_score={w_score:.2f}")

    ahora = datetime.now(LIMA)
    events = sorted((e for e in (odds.fetch_odds() or [])
                     if _hora(e).strftime("%Y-%m-%d") == fecha and _hora(e) > ahora),
                    key=lambda e: e["commence_time"])
    print(f"[setup] {len(events)} partidos sin empezar en {fecha}\n")

    rows = []
    cuota_agotada = False
    for k, e in enumerate(events, 1):
        home_lbl, away_lbl = e["home_team"], e["away_team"]
        th = odds.match_team(home_lbl, model.teams)
        ta = odds.match_team(away_lbl, model.teams)
        market = odds.implied_probs(e)
        market_total = odds.implied_total(e)
        mkt = [market["p_home"], market["p_draw"], market["p_away"]] if market else None
        cov_h = bool(th) and th in model.attack
        cov_a = bool(ta) and ta in model.attack

        if cov_h and cov_a:
            lam, mu = model.rates(th, ta, adv_side=1)
            fuente = "modelo"
        elif mkt is not None:
            lam, mu = _implied_rates(np.array(mkt, float), model.rho)
            fuente = "mercado"
        else:
            print(f"[{k}/{len(events)}] {home_lbl} vs {away_lbl}: sin modelo ni cuotas, se omite")
            continue
        P = dixon_coles.matrix_from_rates(lam, mu, model.rho)
        prior_probs = _outc(P)
        ph, pd_, pa = prior_probs

        ctx = {
            "home": home_lbl, "away": away_lbl, "fecha": fecha, "jornada": "fase de liga",
            "xg_home": float(lam), "xg_away": float(mu),
            "p_home": float(ph), "p_draw": float(pd_), "p_away": float(pa),
            "top_scores": _top_scores(P, 3),
            "form_home": of.recent_form(df, th) if cov_h else "liga local fuera del pool (ratea vía CL/mercado)",
            "form_away": of.recent_form(df, ta) if cov_a else "liga local fuera del pool (ratea vía CL/mercado)",
            "pos_home": _pos_texto(res, th), "pos_away": _pos_texto(res, ta),
            "odds": market,
        }

        print(f"[{k}/{len(events)}] {home_lbl} vs {away_lbl} · prior {ph:.0%}/{pd_:.0%}/{pa:.0%} "
              f"[{fuente}] — debatiendo...")
        degraded = None
        if cuota_agotada:
            degraded = "cuota agotada antes de este partido"
            resultado = _degradado(degraded)
        else:
            try:
                resultado = debate.run_debate(ctx, rounds=ROUNDS)
            except debate.CuotaAgotada as ex:
                cuota_agotada = True
                degraded = f"cuota agotada: {str(ex)[:80]}"
                resultado = _degradado(degraded)
            except Exception as ex:
                degraded = f"degradado: {str(ex)[:80]}"
                resultado = _degradado(degraded)

        dh, da = resultado.delta_home, resultado.delta_away
        lam2, mu2 = lam * np.exp(dh), mu * np.exp(da)
        # T calibra al Dixon-Coles; si el "modelo" es el propio mercado no se aplica
        T_eff = T if fuente == "modelo" else 1.0
        mf_cal = temperature.apply(_outc(dixon_coles.matrix_from_rates(lam2, mu2, model.rho)), T_eff)
        prior_cal = temperature.apply(prior_probs, T_eff)
        bajas = bool(resultado.veredicto.bajas_confirmadas)
        blended, usa_mkt, override_ap = blend.blend_final(mf_cal, prior_cal, mkt, bajas)
        bh, bd, ba = (float(x) for x in blended)

        # marcador: volumen de goles mezclado modelo+mercado al peso fiteado w_score
        if mkt is not None:
            lam_m, mu_m = _market_rates(np.array(mkt), market_total, model.rho)
            if fuente == "modelo":
                lam_sc = (1 - w_score) * lam2 * g + w_score * lam_m
                mu_sc = (1 - w_score) * mu2 * g + w_score * mu_m
            else:  # la base ya es implícita del mercado
                lam_sc, mu_sc = lam_m * np.exp(dh), mu_m * np.exp(da)
        else:
            lam_sc, mu_sc = lam2 * g, mu2 * g
        resumen = blend.score_summary(blend.coherent_matrix(lam_sc, mu_sc, model.rho, [bh, bd, ba]), 3)

        v = resultado.veredicto
        tag = " ⚑override" if override_ap else (f" ⚠{degraded}" if degraded else "")
        print(f"      Δ home={dh:+.3f} away={da:+.3f} bajas={bajas}{tag}")
        for f in v.factores:
            print(f"        - {f}")
        print(f"      FINAL {bh:.0%}/{bd:.0%}/{ba:.0%} · marcador {resumen['score_pred']} "
              f"· xG {resumen['xg_home']:.1f}-{resumen['xg_away']:.1f}\n")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
        ctx["model_final"] = {"p_home": float(mf_cal[0]), "p_draw": float(mf_cal[1]),
                              "p_away": float(mf_cal[2])}
        ctx["blend_weight_market"] = w if usa_mkt else 0.0
        ctx["final"] = {"p_home": bh, "p_draw": bd, "p_away": ba, **resumen}
        debate.guardar(resultado, ctx,
                       DEBATES / f"{home_lbl}_{away_lbl}_{ts}.json".replace(" ", "_").replace("/", "-"))

        rows.append({
            "hora": _hora(e).strftime("%H:%M"), "home": home_lbl, "away": away_lbl, "fuente": fuente,
            "p_home": bh, "p_draw": bd, "p_away": ba,
            "score": resumen["score_pred"], "xg_home": resumen["xg_home"], "xg_away": resumen["xg_away"],
            "delta_home": dh, "delta_away": da, "bajas": bajas, "override": override_ap,
            "factores": v.factores, "resumen": v.resumen, "degraded": degraded,
        })
        resumen_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[fin] listo. Resumen en {resumen_path}" if rows else "[fin] sin partidos que debatir")


if __name__ == "__main__":
    main()
