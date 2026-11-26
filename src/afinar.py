"""Afinado walk-forward de los hiperparámetros del Dixon-Coles.

Para cada semana con partidos de Champions (desde 2024-25) reentrena solo con
partidos anteriores y predice los de esa semana. Mide el modelo puro (sin
mercado ni debate): log-loss del 1X2 (crudo y con la temperatura óptima de la
configuración), RPS y log-verosimilitud del marcador exacto.

Uso:  python -m src.afinar <salida.json> '<json con lista de configs>'
      config = {"xi": 0.00095, "l2": 0.001, "min": 8}
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import odds

odds._key = lambda: None  # sin llamadas a The Odds API: los resultados ya están en data/

from src.calibration import temperature  # noqa: E402
from src.model import dixon_coles  # noqa: E402
from src.predict import lado_local, train  # noqa: E402

DESDE = "2024-07-01"
_DF = None
_EXILIO = None


def _init():
    global _DF, _EXILIO
    _, _DF = train()
    _EXILIO = set(_DF.loc[_DF["competition"] == "UKR", "home_team"])


def _semana(args):
    cfg, inicio = args
    dixon_coles.MIN_MATCHES = cfg["min"]
    ini = pd.Timestamp(inicio)
    fin = ini + pd.Timedelta(days=7)
    base = _DF[_DF["date"] < ini]
    m = dixon_coles.fit(base, ref_date=ini, xi=cfg["xi"], l2=cfg["l2"])
    m.exilio = _EXILIO
    sem = _DF[(_DF["competition"] == "CL") & (_DF["date"] >= ini) & (_DF["date"] < fin)]
    out = []
    for r in sem.itertuples():
        if r.home_team not in m.attack or r.away_team not in m.attack:
            continue
        P = m.score_matrix(r.home_team, r.away_team, adv_side=lado_local(m, r.home_team))
        dim = P.shape[0]
        ps = [float(x) for x in m.outcome_probs(P)]
        hs, a = int(r.home_score), int(r.away_score)
        out.append({"fecha": str(r.date.date()), "home": r.home_team, "away": r.away_team,
                    "p": ps, "y": 0 if hs > a else (1 if hs == a else 2),
                    "ll_score": float(np.log(max(P[hs, a] if hs < dim and a < dim else 1e-12, 1e-12)))})
    return out


def metricas(filas) -> dict:
    p = np.array([f["p"] for f in filas])
    y = np.array([f["y"] for f in filas])
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    T = temperature.fit(p, y)
    pt = temperature.apply(p, T)
    llT = -np.log(np.clip(pt[np.arange(len(y)), y], 1e-12, 1))
    o = np.eye(3)[y]
    rps = 0.5 * ((p[:, 0] - o[:, 0]) ** 2 + (p[:, :2].sum(1) - o[:, :2].sum(1)) ** 2)
    return {"n": len(y), "logloss": float(ll.mean()), "logloss_T": float(llT.mean()), "T": T,
            "rps": float(rps.mean()), "ll_score": float(np.mean([f["ll_score"] for f in filas])),
            "signo": float((p.argmax(1) == y).mean())}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    salida = Path(sys.argv[1])
    configs = json.loads(sys.argv[2])
    previo = json.loads(salida.read_text(encoding="utf-8")) if salida.exists() else {}
    _init()
    cl = _DF[(_DF["competition"] == "CL") & (_DF["date"] >= DESDE)]
    semanas = sorted({str((d - pd.Timedelta(days=d.weekday())).date()) for d in cl["date"]})
    print(f"[afinar] {len(cl)} partidos CL desde {DESDE} en {len(semanas)} semanas")
    with ProcessPoolExecutor(max_workers=7, initializer=_init) as ex:
        for cfg in configs:
            clave = f"xi={cfg['xi']} l2={cfg['l2']} min={cfg['min']}"
            if clave in previo:
                print(f"[afinar] {clave}: ya calculada")
                continue
            t0 = time.time()
            filas = [f for sem in ex.map(_semana, [(cfg, s) for s in semanas]) for f in sem]
            previo[clave] = {"cfg": cfg, "metricas": metricas(filas), "filas": filas}
            salida.write_text(json.dumps(previo, ensure_ascii=False), encoding="utf-8")
            mt = previo[clave]["metricas"]
            print(f"[afinar] {clave}: n={mt['n']} logloss={mt['logloss']:.4f} logloss_T={mt['logloss_T']:.4f} "
                  f"(T={mt['T']:.2f}) rps={mt['rps']:.4f} ll_score={mt['ll_score']:.4f} "
                  f"signo={mt['signo']:.1%} · {time.time() - t0:.0f} s")
    print("[afinar] fin")


if __name__ == "__main__":
    main()
