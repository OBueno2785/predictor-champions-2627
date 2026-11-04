"""Marcadores en vivo de la Champions para la app (artifact).

Consulta The Odds API (/scores, 1 crédito por consulta) solo mientras haya
partidos de la fecha empezados y sin terminar; antes del primer inicio y entre
tandas duerme hasta la siguiente hora de inicio. Escribe una línea JSON en la
salida estándar por cada consulta con cambios (inicio, gol, final) para que
Claude los publique en la base de la app. El minuto lo estima la propia página
a partir de la hora de inicio. Guarda el último estado en
outputs/en_vivo/<fecha>.json, así que puede reiniciarse sin repetir eventos.

Con --web, además aplica cada cambio a docs/data/partidos.json y lo publica
con un commit (la web de GitHub Pages lo muestra en uno o dos minutos).

Uso:  python -m src.en_vivo [today|YYYY-MM-DD] [--cada 240] [--web]
"""
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from src import web_datos
from src.ingest import odds

LIMA = ZoneInfo("America/Lima")
SALIDA = Path(__file__).resolve().parent.parent / "outputs" / "en_vivo"
# nombre en The Odds API -> slug del escudo / id del documento en la app
SLUG = {
    "AEK Athens": "aek-athens", "Arsenal": "arsenal", "AS Roma": "as-roma", "Aston Villa": "aston-villa",
    "Atlético Madrid": "atletico-madrid", "Barcelona": "barcelona", "Bayern Munich": "bayern-munich",
    "Bodø/Glimt": "bod-glimt", "Borussia Dortmund": "borussia-dortmund", "Club Brugge": "club-brugge",
    "Como": "como", "Fenerbahce": "fenerbahce", "Feyenoord": "feyenoord", "Galatasaray": "galatasaray",
    "Inter Milan": "inter", "LASK": "lask", "Lille": "lille", "Liverpool": "liverpool",
    "Manchester City": "manchester-city", "Manchester United": "manchester-united", "Napoli": "napoli",
    "Paris Saint Germain": "paris-sg", "Porto": "porto", "PSV Eindhoven": "psv-eindhoven",
    "RB Leipzig": "rb-leipzig", "RC Lens": "rc-lens", "Real Betis": "real-betis", "Real Madrid": "real-madrid",
    "Sabah FK": "sabah-fk", "Shakhtar Donetsk": "shakhtar-donetsk", "ŠK Slovan Bratislava": "sk-slovan-bratislava",
    "Slavia Praha": "slavia-praha", "Sporting Lisbon": "sporting-cp", "VfB Stuttgart": "vfb-stuttgart",
    "Viking FK": "viking-fk", "Villarreal": "villarreal",
}


def _slug(nombre: str) -> str:
    if nombre in SLUG:
        return SLUG[nombre]
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def _inicio(e) -> datetime:
    return datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(LIMA)


def _consultar(dias: int | None = None):
    params = {"apiKey": odds._key()}
    if dias:
        params["daysFrom"] = dias  # incluye terminados (2 créditos)
    r = requests.get(f"{odds.BASE}/scores", params=params, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def _marcador(e) -> str | None:
    sc = {s["name"]: s["score"] for s in (e.get("scores") or [])}
    h, a = sc.get(e["home_team"]), sc.get(e["away_team"])
    return f"{int(h)}-{int(a)}" if h is not None and a is not None else None


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    arg = args[0] if args else "today"
    fecha = datetime.now(LIMA).strftime("%Y-%m-%d") if arg == "today" else arg
    cada = int(sys.argv[sys.argv.index("--cada") + 1]) if "--cada" in sys.argv else 240
    web = "--web" in sys.argv
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA / f"{fecha}.json"
    estado = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    esperados: dict = {}  # id -> hora de inicio, de todos los partidos de la fecha vistos

    while True:
        try:
            eventos, creditos = _consultar()
        except Exception as ex:
            print(json.dumps({"error": str(ex)[:160]}, ensure_ascii=False))
            time.sleep(cada)
            continue
        ahora = datetime.now(LIMA)
        delDia = [e for e in eventos if _inicio(e).strftime("%Y-%m-%d") == fecha]
        cambios, vivos = [], set()
        for e in delDia:
            did = f"{fecha}-{_slug(e['home_team'])}-{_slug(e['away_team'])}"
            esperados[did] = _inicio(e)
            real = _marcador(e)
            if real is None:
                continue
            nuevo = {"estado": "final" if e.get("completed") else "en_vivo", "real": real}
            if nuevo["estado"] == "en_vivo":
                vivos.add(did)
            if estado.get(did) != nuevo:
                estado[did] = nuevo
                cambios.append({"id": did, **nuevo})

        # sin daysFrom el feed deja de listar los terminados: confirmar su final
        sin_cerrar = [d for d, s in estado.items() if s["estado"] == "en_vivo" and d not in vivos]
        if sin_cerrar:
            try:
                todos, creditos = _consultar(1)
                for e in todos:
                    did = f"{fecha}-{_slug(e['home_team'])}-{_slug(e['away_team'])}"
                    real = _marcador(e)
                    if did in sin_cerrar and e.get("completed") and real:
                        nuevo = {"estado": "final", "real": real}
                        if estado.get(did) != nuevo:
                            estado[did] = nuevo
                            cambios.append({"id": did, **nuevo})
            except Exception as ex:
                print(json.dumps({"error": str(ex)[:160]}, ensure_ascii=False))

        if cambios:
            ruta.write_text(json.dumps(estado, ensure_ascii=False, indent=1), encoding="utf-8")
            print(json.dumps({"hora": ahora.strftime("%H:%M"), "cambios": cambios, "creditos": creditos},
                             ensure_ascii=False))
            if web:
                try:
                    web_datos.aplicar_partidos(cambios)
                    if not web_datos.publicar(f"datos: marcadores en vivo {fecha} {ahora:%H:%M}"):
                        print(json.dumps({"error": "no se pudo publicar en la web (git)"}))
                except Exception as ex:
                    print(json.dumps({"error": f"web: {str(ex)[:140]}"}, ensure_ascii=False))

        terminados = [d for d in esperados if estado.get(d, {}).get("estado") == "final"]
        if esperados and len(terminados) == len(esperados):
            print(json.dumps({"fin": True, "partidos": len(esperados), "creditos": creditos}))
            return
        if esperados and ahora > max(esperados.values()) + timedelta(hours=3, minutes=30):
            print(json.dumps({"fin": True, "aviso": "tiempo agotado sin cerrar todos los partidos",
                              "sin_cerrar": [d for d in esperados if d not in terminados]}, ensure_ascii=False))
            return

        en_juego = [d for d, h in esperados.items() if h <= ahora and estado.get(d, {}).get("estado") != "final"]
        if en_juego:
            time.sleep(cada)
        else:
            futuros = [h for h in esperados.values() if h > ahora]
            espera = (min(futuros) - ahora).total_seconds() + 60 if futuros else cada
            time.sleep(max(60, min(espera, 3 * 3600)))


if __name__ == "__main__":
    main()
