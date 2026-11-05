"""Completa con TheSportsDB lo que openfootball trae incompleto.

openfootball es la fuente principal (nombres de club coherentes entre ligas y
copas). Una liga-temporada cuyo archivo trae menos del 95% de los partidos que
anuncia su cabecera (la temporada en curso, o archivos a medio cargar) se
reemplaza entera por los resultados de TheSportsDB, consultados jornada por
jornada con la clave gratuita. Las copas europeas que openfootball no tiene
(fase de liga 2025-26, temporada 2026-27) también salen de aquí.

Los nombres de TheSportsDB se traducen a los de openfootball de la misma liga;
lo que no se puede emparejar con seguridad queda en el reporte para revisar.

Escribe data/tsdb_<comp>_<temporada>.json, que openfootball.load_training usa
en lugar del archivo incompleto. Uso:  python -m src.ingest.tsdb
"""
import difflib
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests

from src.ingest import openfootball as of

API = "https://www.thesportsdb.com/api/v1/json/123"
LIGAS = {"ESP": 4335, "ENG": 4328, "GER": 4331, "ITA": 4332, "AUT": 4621,
         "FRA": 4334, "POR": 4344, "NED": 4337, "TUR": 4339, "GRE": 4336, "BEL": 4338,
         "NOR": 4358, "CZE": 4631, "UKR": 4354, "SVK": 4672, "AZE": 4693}
# copa -> (id, comp de sus previas); antes del 1 de septiembre son previas
COPAS = {"CL": (4480, "CLQ"), "EL": (4481, "ELQ"), "UECL": (5071, "UECLQ")}
COMPLETA = 0.95
MAX_RONDA, VACIAS_FIN = 50, 5
# copas: jornadas de fase de liga/grupos + ronda de 32, octavos, cuartos (125) y
# semis (150); la final (200) es en cancha neutral y se omite
RONDAS_COPA = list(range(1, 9)) + [16, 32, 125, 150]
PAUSA = 2.5
# nombre de TheSportsDB -> nombre canónico, revisado a mano (5-nov-2026); los
# que apuntan a sí mismos son clubes distintos que el emparejamiento automático
# confundía con otro (Poltava ≠ Vorskla Poltava, CSKA Sofia ≠ CSKA 1948, ...)
ALIAS = {
    "Athletic Bilbao": "Athletic Club", "Inter Milan": "Inter",
    "Austria Vienna": "Austria Wien", "Rapid Vienna": "Rapid Wien",
    "Brest": "Stade Brestois", "Lyon": "Olympique Lyonnais",
    "Olympiacos": "Olympiakos Piraeus", "AEL": "AEL Larissa",
    "Sint-Truiden": "Sint-Truidense VV",
    "Slavia Prague": "Slavia Praha", "Sparta Prague": "Sparta Praha",
    "Dynamo Kyiv": "Dinamo Kiev", "Poltava": "Poltava", "Veres Rivne": "NK Veres",
    "Kəpəz": "Kapaz PFK", "Qəbələ": "Gabala",
    "Drita": "Drita Gjilan", "Petrocub Hîncești": "CS Petrocub",
    "Ferencváros": "Ferencvárosi TC", "Paks": "Paksi SE",
    "FC Copenhagen": "København", "Levadia Tallinn": "FCI Levadia",
    "CSKA Sofia": "CSKA Sofia", "Inter Turku": "Inter Turku",
    "Bohemians": "Bohemians Dublin",  # Bohemian FC (Irlanda), no Bohemians Praha 1905
}


def _plano(n: str) -> str:
    n = unicodedata.normalize("NFKD", of.normalize_team(n)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", n).split())


def _temporada_tsdb(s: str) -> str:
    return s if "-" not in s else f"{s[:4]}-{int(s[:4]) + 1}"


def _get(params: dict, endpoint: str = "eventsround.php"):
    for _ in range(6):
        try:
            r = requests.get(f"{API}/{endpoint}", params=params, timeout=30,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 429:
                time.sleep(60)
                continue
            time.sleep(PAUSA)
            r.raise_for_status()
            return r.json() or {}
        except Exception:
            time.sleep(10)
    return None


def _final(e) -> bool:
    return e.get("strStatus") in ("FT", "Match Finished") \
        and e.get("intHomeScore") not in (None, "") and e.get("intAwayScore") not in (None, "")


def _ronda(cache: Path, lid: int, temp: str, r: int, en_curso: bool) -> list | None:
    """Eventos de una jornada. Se guardan en caché, de temporadas ya cerradas,
    todas; de la temporada en curso, las jornadas cuyos partidos terminaron o
    quedaron atrás más de una semana (aplazados: no se vuelve a consultar por uno)."""
    f = cache / f"{lid}_{temp}_{r}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    d = _get({"id": lid, "r": r, "s": temp})
    if d is None:
        return None
    ev = d.get("events") or []
    hace_semana = date.fromordinal(date.today().toordinal() - 7).isoformat()
    if not en_curso or (ev and all(_final(e) or (e.get("dateEvent") or "9999") < hace_semana
                                   for e in ev)):
        f.write_text(json.dumps(ev, ensure_ascii=False), encoding="utf-8")
    return ev


def _eventos(cache: Path, lid: int, temp: str, rondas=None, en_curso: bool = False) -> list:
    """Todas las jornadas. En la temporada en curso se detiene en la primera
    jornada que todavía no empezó (no hay nada más que bajar)."""
    hoy = date.today().isoformat()
    out, vacias, visto = {}, 0, False
    for r in rondas or range(1, MAX_RONDA + 1):
        ev = _ronda(cache, lid, temp, r, en_curso)
        if ev is None:
            continue
        if ev:
            visto, vacias = True, 0
            for e in ev:
                out[e["idEvent"]] = e
            if en_curso and not any(_final(e) for e in ev) \
                    and min(e.get("dateEvent") or "9999" for e in ev) > hoy:
                break
        elif en_curso and rondas is not None and visto:
            break  # copas: ronda eliminatoria aún sin sortear
        else:
            vacias += 1
            if visto and vacias >= VACIAS_FIN and rondas is None:
                break
    return list(out.values())


def _emparejar(nombres: set, candidatos: set, reporte: list, etiqueta: str) -> dict:
    """nombre TheSportsDB -> canónico. Exacto, sin acentos, contenido o muy parecido;
    si no, queda como está (club sin historial en openfootball)."""
    por_plano = defaultdict(set)
    for c in candidatos:
        por_plano[_plano(c)].add(c)
    m = {}
    for n in sorted(nombres):
        if n in ALIAS:
            m[n] = ALIAS[n]
            continue
        canon = of.normalize_team(n)
        if canon in candidatos:
            m[n] = canon
            continue
        p = _plano(n)
        if len(por_plano.get(p, ())) == 1:
            m[n] = next(iter(por_plano[p]))
            reporte.append(f"{etiqueta}: {n!r} → {m[n]!r} (sin acentos)")
            continue
        tok = set(p.split())
        cont = {c for q, cs in por_plano.items() for c in cs
                if tok and (tok <= set(q.split()) or set(q.split()) <= tok)}
        if len(cont) == 1:
            m[n] = next(iter(cont))
            reporte.append(f"{etiqueta}: {n!r} → {m[n]!r} (contenido)")
            continue
        punt = sorted(((difflib.SequenceMatcher(None, p, q).ratio(), q) for q in por_plano), reverse=True)
        if punt and punt[0][0] >= 0.85 and (len(punt) == 1 or punt[0][0] - punt[1][0] >= 0.05) \
                and len(por_plano[punt[0][1]]) == 1:
            m[n] = next(iter(por_plano[punt[0][1]]))
            reporte.append(f"{etiqueta}: {n!r} → {m[n]!r} (parecido {punt[0][0]:.2f})")
            continue
        m[n] = canon
        reporte.append(f"{etiqueta}: {n!r} sin pareja → queda {canon!r}"
                       + (f" (ambiguo: {sorted(cont)})" if len(cont) > 1 else ""))
    return m


def _filas(eventos, mapa) -> list:
    return [{"date": e["dateEvent"], "home_team": mapa[e["strHomeTeam"]],
             "away_team": mapa[e["strAwayTeam"]],
             "home_score": int(e["intHomeScore"]), "away_score": int(e["intAwayScore"])}
            for e in eventos if _final(e)]


def actualizar(data_dir: Path, verbose: bool = True) -> list:
    """Reemplaza las ligas-temporadas incompletas y agrega las copas faltantes.
    Devuelve el reporte de emparejamientos no exactos."""
    cache = data_dir / "tsdb_cache"
    cache.mkdir(parents=True, exist_ok=True)
    reporte = []

    # openfootball por liga-temporada: filas y si está completa
    of_filas, completa = defaultdict(list), {}
    for _, nombre, comp, s in of._archivos():
        f = data_dir / nombre
        if not f.exists():
            completa[(comp, s)] = False
            continue
        txt = f.read_text(encoding="utf-8")
        filas = of._parse_file(txt, comp, s)
        of_filas[comp] += filas
        esp = of.esperados(txt)
        completa[(comp, s)] = bool(esp) and len(filas) >= COMPLETA * esp
    universo = {x[k] for v in of_filas.values() for x in v for k in ("home_team", "away_team")}

    for comp, lid in LIGAS.items():
        cands = {x[k] for x in of_filas[comp] for k in ("home_team", "away_team")}
        for s in of.temporadas(comp):
            destino = data_dir / f"tsdb_{comp}_{s}.json"
            if completa.get((comp, s)):
                destino.unlink(missing_ok=True)
                continue
            ev = _eventos(cache, lid, _temporada_tsdb(s), en_curso=s == of.temporadas(comp)[-1])
            ev = [e for e in ev if _final(e)]
            if not ev:
                continue
            mapa = _emparejar({e["strHomeTeam"] for e in ev} | {e["strAwayTeam"] for e in ev},
                              cands or universo, reporte, f"{comp} {s}")
            filas = _filas(ev, mapa)
            destino.write_text(json.dumps(filas, ensure_ascii=False, indent=0), encoding="utf-8")
            if verbose:
                print(f"  {comp} {s}: {len(filas)} partidos de TheSportsDB "
                      f"(openfootball incompleto)", flush=True)

    # copas: en TheSportsDB cada jornada mezcla las previas de verano con la fase
    # de liga; se separan por fecha. La fase de liga de la CL nunca sale de aquí
    # (openfootball + data/cl_results.json).
    def previa(e):
        return "06-15" <= e["dateEvent"][5:] < "09-01"

    for comp, (lid, comp_q) in COPAS.items():
        for s in of.SEASONS:
            ev = None
            for c, es_previa in ((comp, False), (comp_q, True)):
                destino = data_dir / f"tsdb_{c}_{s}.json"
                if c == "CL" or (data_dir / f"of_{c}_{s}.txt").exists():
                    destino.unlink(missing_ok=True)
                    continue
                if ev is None:
                    ev = [e for e in _eventos(cache, lid, _temporada_tsdb(s), RONDAS_COPA,
                                              en_curso=s == of.SEASONS[-1]) if _final(e)]
                sel = [e for e in ev if previa(e) == es_previa]
                if not sel:
                    continue
                mapa = _emparejar({e["strHomeTeam"] for e in sel} | {e["strAwayTeam"] for e in sel},
                                  universo, reporte, f"{c} {s}")
                filas = _filas(sel, mapa)
                destino.write_text(json.dumps(filas, ensure_ascii=False, indent=0), encoding="utf-8")
                if verbose:
                    print(f"  {c} {s}: {len(filas)} partidos de TheSportsDB", flush=True)
    return reporte


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raiz = Path(__file__).resolve().parents[2] / "data"
    rep = actualizar(raiz)
    print("\nEmparejamientos no exactos:")
    for linea in rep:
        print("  " + linea)
