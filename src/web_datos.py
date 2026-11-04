"""Datos de la web pública (GitHub Pages, carpeta docs/).

La web lee docs/data/partidos.json, modelo.json y estadisticas.json y los
recarga cada minuto. Este módulo aplica cambios a partidos.json (por `id`) y
los publica con un commit + push solo de docs/data.

Uso:  python -m src.web_datos aplicar <cambios.json> "<mensaje del commit>"
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
    """Fusiona cada cambio en el partido con el mismo `id` (lo crea si no existe)."""
    ruta = DATOS / "partidos.json"
    doc = json.loads(ruta.read_text(encoding="utf-8"))
    por_id = {p["id"]: p for p in doc["partidos"]}
    for c in cambios:
        por_id.setdefault(c["id"], {}).update(c)
    doc["partidos"] = sorted(por_id.values(), key=lambda p: p["id"])
    doc["actualizado"] = datetime.now(LIMA).isoformat(timespec="seconds")
    ruta.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(cambios)


def publicar(mensaje: str) -> bool:
    """Commit + push de docs/data. Devuelve False si git falla."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True)
    git("add", "docs/data")
    if git("diff", "--cached", "--quiet").returncode == 0:
        return True
    if git("commit", "-q", "-m", mensaje + FIRMA).returncode != 0:
        return False
    return git("push", "-q").returncode == 0


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "aplicar":
        sys.exit('Uso: python -m src.web_datos aplicar <cambios.json> "<mensaje>"')
    n = aplicar_partidos(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    print(f"{n} cambios aplicados; publicado: {publicar(sys.argv[3])}")
