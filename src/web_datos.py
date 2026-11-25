"""Datos de la web pública (GitHub Pages, carpeta docs/).

La web lee docs/data/partidos.json, modelo.json y estadisticas.json y los
recarga cada minuto. Este módulo aplica cambios a partidos.json (por `id`) y
los publica con un commit + push solo de docs/data.

Uso:  python -m src.web_datos aplicar <cambios.json> "<mensaje del commit>"
      python -m src.web_datos jornada <fecha> "<Jornada N · día k>" <JN> <notas.json> <salida.json>
        (arma los partidos de una fecha desde outputs/jornadas/<fecha>.json; luego `aplicar`)
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "docs" / "data"
LIMA = ZoneInfo("America/Lima")
FIRMA = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def aplicar_partidos(cambios: list) -> int:
    """Fusiona cada cambio en el partido con el mismo `id`. Solo crea partidos
    a partir de cambios completos (con local/visita): un marcador en vivo de un
    partido sin pronóstico no se publica."""
    ruta = DATOS / "partidos.json"
    doc = json.loads(ruta.read_text(encoding="utf-8"))
    por_id = {p["id"]: p for p in doc["partidos"]}
    n = 0
    for c in cambios:
        if c["id"] not in por_id and not {"local", "visita", "escL", "escV"} <= c.keys():
            continue
        por_id.setdefault(c["id"], {}).update(c)
        n += 1
    doc["partidos"] = sorted(por_id.values(), key=lambda p: p["id"])
    doc["actualizado"] = datetime.now(LIMA).isoformat(timespec="seconds")
    ruta.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return n


def _porcentajes(ps) -> list:
    """Probabilidades a enteros que suman 100 (mayor resto)."""
    base = [int(p * 100) for p in ps]
    resto = sorted(range(len(ps)), key=lambda i: ps[i] * 100 - base[i], reverse=True)
    for i in resto[:100 - sum(base)]:
        base[i] += 1
    return base


def desde_jornada(fecha: str, etiqueta: str, corta: str, notas: dict) -> list:
    """Partidos (formato de la web y de la app) desde outputs/jornadas/<fecha>.json.
    `notas`: {id: texto corto} redactado a partir del debate. En partidos ya
    publicados no toca el estado en vivo/final ni la nota si no hay una nueva."""
    from src.en_vivo import _slug
    previos = json.loads((DATOS / "partidos.json").read_text(encoding="utf-8"))["partidos"]
    publicados = {p["id"] for p in previos}
    nombre = {}
    for p in previos:
        if {"local", "visita", "escL", "escV"} <= p.keys():
            nombre[p["escL"]], nombre[p["escV"]] = p["local"], p["visita"]
    filas = json.loads((RAIZ / "outputs" / "jornadas" / f"{fecha}.json").read_text(encoding="utf-8"))
    out = []
    for r in filas:
        sl, sv = _slug(r["home"]), _slug(r["away"])
        pid = f"{fecha}-{sl}-{sv}"
        pl, pe, pv = _porcentajes([r["p_home"], r["p_draw"], r["p_away"]])
        p = {
            "id": pid, "fecha": fecha, "hora": r["hora"], "jornada": etiqueta, "jornadaCorta": corta,
            "local": nombre.get(sl, r["home"]), "visita": nombre.get(sv, r["away"]), "escL": sl, "escV": sv,
            "pl": pl, "pe": pe, "pv": pv, "marcador": r["score"],
            "xgL": round(r["xg_home"], 2), "xgV": round(r["xg_away"], 2), "q": r.get("q"),
            "fuente": r["fuente"].capitalize(), "ajuste": bool(r["override"]),
            "nota": notas.get(pid, ""), "estado": "por_empezar", "real": None, "minuto": None, "stats": {},
        }
        if pid in publicados:
            for k in ("estado", "real", "minuto", "stats"):
                del p[k]
            if pid not in notas:
                del p["nota"]
        out.append(p)
    return out


def publicar(mensaje: str) -> bool:
    """Commit + push de docs/data. Devuelve False si git falla."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True)
    git("add", "docs/data")
    if git("diff", "--cached", "--quiet", "--", "docs/data").returncode == 0:
        return True
    if git("commit", "-q", "-m", mensaje + FIRMA, "--", "docs/data").returncode != 0:
        return False
    return git("push", "-q").returncode == 0


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "jornada":
        notas = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8")) if Path(sys.argv[5]).exists() else {}
        ps = desde_jornada(sys.argv[2], sys.argv[3], sys.argv[4], notas)
        Path(sys.argv[6]).write_text(json.dumps(ps, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(ps)} partidos en {sys.argv[6]}")
        sys.exit(0)
    if len(sys.argv) < 4 or sys.argv[1] != "aplicar":
        sys.exit('Uso: python -m src.web_datos aplicar <cambios.json> "<mensaje>"')
    n = aplicar_partidos(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    print(f"{n} cambios aplicados; publicado: {publicar(sys.argv[3])}")
