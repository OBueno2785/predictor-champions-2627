"""Fase de liga de la Champions (formato suizo desde 2024-25).

36 equipos, una sola tabla, 8 partidos cada uno (4 local + 4 visita vs 8 rivales
distintos). Clasificación:
  - Puestos 1-8   → octavos de final directo
  - Puestos 9-24  → playoff de eliminación (ida y vuelta)
  - Puestos 25-36 → eliminados

Desempates (aprox. UEFA): puntos, diferencia de gol, goles a favor.
"""
import sys
from pathlib import Path

import pandas as pd

PTS = {"W": 3, "D": 1, "L": 0}


def zona(pos: int) -> str:
    if pos <= 8:
        return "octavos directo"
    if pos <= 24:
        return "playoff"
    return "eliminado"


def standings(matches: pd.DataFrame) -> pd.DataFrame:
    """matches: home_team, away_team, home_score, away_score (solo jugados)."""
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    st = {t: {"team": t, "pj": 0, "pts": 0, "gf": 0, "gc": 0} for t in teams}
    for r in matches.itertuples():
        hs, as_ = int(r.home_score), int(r.away_score)
        for t, f, c in [(r.home_team, hs, as_), (r.away_team, as_, hs)]:
            st[t]["pj"] += 1
            st[t]["gf"] += f
            st[t]["gc"] += c
            st[t]["pts"] += PTS["W" if f > c else ("D" if f == c else "L")]
    df = pd.DataFrame(st.values())
    df["dg"] = df["gf"] - df["gc"]
    df = df.sort_values(["pts", "dg", "gf"], ascending=False).reset_index(drop=True)
    df["pos"] = df.index + 1
    df["zona"] = df["pos"].map(zona)
    return df[["pos", "team", "pj", "pts", "gf", "gc", "dg", "zona"]]


def standings_markdown(matches: pd.DataFrame, titulo: str) -> str:
    t = standings(matches)
    lines = [f"# {titulo}", "",
             "Formato suizo: 1-8 octavos directo · 9-24 playoff · 25-36 eliminados.", "",
             "| Pos | Equipo | PJ | Pts | GF | GC | DG | Zona |",
             "|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        sep = {8: "|—— octavos ——|", 24: "|—— playoff ——|"}.get(r.pos)
        marca = "**" if r.pos <= 8 else ("" if r.pos <= 24 else "_")
        lines.append(f"| {r.pos} | {marca}{r.team}{marca} | {r.pj} | {r.pts} | "
                     f"{r.gf} | {r.gc} | {r.dg:+d} | {r.zona} |")
        if sep:
            lines.append(f"| | {sep} | | | | | | |")
    return "\n".join(lines)


def _extract_league_phase(df: pd.DataFrame, ini: str, fin: str) -> pd.DataFrame:
    m = df[(df["competition"] == "CL") & (df["stage"] == "league")
           & (df["date"] >= ini) & (df["date"] <= fin)]
    return m


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from src.ingest import openfootball as of
    root = Path(__file__).resolve().parent.parent
    df = of.load_training(root / "data")
    # test: fase de liga CL 2025-26
    m = _extract_league_phase(df, "2025-08-01", "2026-02-15")
    print(f"Partidos de fase de liga CL 25-26 detectados: {len(m)} "
          f"({m['home_team'].nunique()} equipos locales)")
    print()
    print(standings_markdown(m, "Tabla CL 2025-26 (fase de liga) — reconstruida"))


if __name__ == "__main__":
    main()
