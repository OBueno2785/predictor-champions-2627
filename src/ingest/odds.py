"""Cuotas y resultados de la Champions vía The Odds API.

- fetch_odds(): partidos próximos con cuotas 1X2 (implícitas sin margen).
- fetch_results(): partidos ya jugados (endpoint /scores), se acumulan en
  data/cl_results.json para entrenar el modelo y ajustar los pesos.
- match_team(): empareja el nombre de la API con el del modelo (openfootball).
"""
import json
import os
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
}


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
        "apiKey": key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal",
    }, timeout=60)
    r.raise_for_status()
    return r.json()


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
