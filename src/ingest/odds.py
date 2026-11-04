"""Cuotas y resultados de la Champions vía The Odds API (y TheSportsDB).

- fetch_odds(): partidos próximos con cuotas 1X2 (implícitas sin margen).
- fetch_results(): partidos ya jugados (endpoint /scores, solo últimos 3 días),
  se acumulan en data/cl_results.json para entrenar el modelo y ajustar los pesos.
- fetch_results_fechas(): resultados de las últimas N jornadas vía TheSportsDB,
  sin la ventana de 3 días; completa data/cl_results.json sin duplicar.
- match_team(): empareja el nombre de la API con el del modelo (openfootball).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.ingest.openfootball import normalize_team

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

SPORT = "soccer_uefa_champs_league"
BASE = f"https://api.the-odds-api.com/v4/sports/{SPORT}"

# alias nombre-API -> forma normalizada del modelo
_ALIAS_API = {
    "inter milan": "Inter", "aek athens": "AEK", "manchester city": "Manchester City",
    "paris saint germain": "Paris SG", "sporting lisbon": "Sporting CP",
    "psv eindhoven": "PSV", "bayern munich": "Bayern Munich",
    "atletico madrid": "Atlético Madrid", "atlético madrid": "Atlético Madrid",
    # nombres de TheSportsDB -> los de The Odds API normalizados
    "fenerbahçe": "Fenerbahce", "sabah baku": "Sabah", "slavia prague": "Slavia Praha",
    "slovan bratislava": "ŠK Slovan Bratislava", "sporting cp": "Sporting CP",
}

TSDB = "https://www.thesportsdb.com/api/v1/json/123"
TSDB_LIGA_CL = 4480
TSDB_TEMPORADA = "2026-2027"
INICIO_FASE_LIGA = "2026-09-01"  # en TheSportsDB cada ronda mezcla la previa de verano con la jornada


def _key() -> str | None:
    return os.environ.get("ODDS_API_KEY")


def _norm(name: str) -> str:
    return _ALIAS_API.get(name.lower().strip(), normalize_team(name))


def match_team(api_name: str, model_teams) -> str | None:
    n = _norm(api_name)
    low = {t.lower(): t for t in model_teams}
    if n.lower() in low:
        return low[n.lower()]
    cand = [t for t in model_teams if n.lower() in t.lower() or t.lower() in n.lower()]
    return cand[0] if len(cand) == 1 else (cand[0] if cand else None)


def fetch_odds() -> list | None:
    key = _key()
    if not key:
        return None
    r = requests.get(f"{BASE}/odds", params={
        "apiKey": key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal",
    }, timeout=60)
    r.raise_for_status()
    return r.json()


def implied_total(event: dict) -> dict | None:
    """Línea de goles del mercado (Over/Under) agregada por las casas: devuelve
    la línea modal y la mediana de P(Over) sin margen. Informa el VOLUMEN de
    goles del marcador (el 1X2 solo informa el ganador)."""
    from collections import Counter
    por_linea: dict = {}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != "totals":
                continue
            pr = {(o["name"], o.get("point")): o["price"] for o in mk["outcomes"]}
            puntos = {p for (_, p) in pr}
            for pt in puntos:
                ov, un = pr.get(("Over", pt)), pr.get(("Under", pt))
                if ov and un:
                    po = (1 / ov) / ((1 / ov) + (1 / un))  # devig
                    por_linea.setdefault(pt, []).append(po)
    if not por_linea:
        return None
    line = Counter({pt: len(v) for pt, v in por_linea.items()}).most_common(1)[0][0]
    return {"line": float(line), "p_over": float(np.median(por_linea[line])),
            "n_bookies": len(por_linea[line])}


def implied_probs(event: dict) -> dict | None:
    per = []
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != "h2h":
                continue
            pr = {o["name"]: o["price"] for o in mk["outcomes"]}
            h, d, a = pr.get(event["home_team"]), pr.get("Draw"), pr.get(event["away_team"])
            if h and d and a:
                inv = np.array([1 / h, 1 / d, 1 / a])
                per.append(inv / inv.sum())
    if not per:
        return None
    m = np.median(np.array(per), axis=0)
    m = m / m.sum()
    return {"p_home": float(m[0]), "p_draw": float(m[1]), "p_away": float(m[2]),
            "n_bookies": len(per)}


def fetch_results(data_dir: Path, days_from: int = 3) -> pd.DataFrame:
    """Descarga resultados recientes y los acumula en data/cl_results.json."""
    store = data_dir / "cl_results.json"
    prev = {}
    if store.exists():
        prev = {r["id"]: r for r in json.loads(store.read_text(encoding="utf-8"))}
    key = _key()
    if key:
        try:
            r = requests.get(f"{BASE}/scores", params={"apiKey": key, "daysFrom": days_from},
                             timeout=60)
            r.raise_for_status()
            for e in r.json():
                if not e.get("completed"):
                    continue
                sc = {s["name"]: s["score"] for s in (e.get("scores") or [])}
                hs, as_ = sc.get(e["home_team"]), sc.get(e["away_team"])
                if hs is None or as_ is None:
                    continue
                prev[e["id"]] = {
                    "id": e["id"], "date": e["commence_time"][:10],
                    "home": e["home_team"], "away": e["away_team"],
                    "home_score": int(hs), "away_score": int(as_),
                }
        except Exception:
            pass
    rows = list(prev.values())
    store.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    if not rows:
        return pd.DataFrame(columns=["date", "home", "away", "home_score", "away_score"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _tsdb_ronda(rnd: int) -> list:
    url = f"{TSDB}/eventsround.php"
    for _ in range(3):
        try:
            r = requests.get(url, params={"id": TSDB_LIGA_CL, "r": rnd, "s": TSDB_TEMPORADA},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code == 429:  # límite por minuto del plan gratuito
                time.sleep(30)
                continue
            r.raise_for_status()
            return (r.json() or {}).get("events") or []
        except Exception:
            time.sleep(5)
    return []


def fetch_results_fechas(data_dir: Path, n_fechas: int = 3) -> int:
    """Agrega a data/cl_results.json los resultados de las últimas `n_fechas`
    jornadas de la fase de liga con partidos terminados (TheSportsDB). No tiene
    la ventana de 3 días del /scores de The Odds API, así que una actualización
    tardía no pierde resultados. Devuelve cuántos partidos nuevos agregó."""
    store = data_dir / "cl_results.json"
    rows = json.loads(store.read_text(encoding="utf-8")) if store.exists() else []
    vistos = {(r["date"], _norm(r["home"]), _norm(r["away"])) for r in rows}
    nuevos, jornadas = 0, 0
    for rnd in range(8, 0, -1):
        eventos = [e for e in _tsdb_ronda(rnd)
                   if (e.get("dateEvent") or "") >= INICIO_FASE_LIGA
                   and e.get("strStatus") == "FT"
                   and e.get("intHomeScore") not in (None, "") and e.get("intAwayScore") not in (None, "")]
        time.sleep(2.5)
        if not eventos:
            continue
        jornadas += 1
        for e in eventos:
            clave = (e["dateEvent"], _norm(e["strHomeTeam"]), _norm(e["strAwayTeam"]))
            if clave in vistos:
                continue
            vistos.add(clave)
            rows.append({"id": f"tsdb-{e['idEvent']}", "date": e["dateEvent"],
                         "home": e["strHomeTeam"], "away": e["strAwayTeam"],
                         "home_score": int(e["intHomeScore"]), "away_score": int(e["intAwayScore"])})
            nuevos += 1
        if jornadas >= n_fechas:
            break
    store.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return nuevos


def results_as_training(data_dir: Path, model_teams=None) -> pd.DataFrame:
    """Resultados CL 26-27 en el esquema de entrenamiento (nombres normalizados)."""
    df = fetch_results(data_dir)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "date": df["date"],
        "home_team": df["home"].map(_norm), "away_team": df["away"].map(_norm),
        "home_score": df["home_score"], "away_score": df["away_score"],
        "competition": "CL", "stage": "league", "neutral": False,
        "weight_importance": 1.0,
    })
    return out
