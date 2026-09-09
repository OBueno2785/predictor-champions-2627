"""Ingesta de resultados de clubes desde openfootball (GitHub raw).

Reúne varias temporadas de las grandes ligas (España, Inglaterra, Alemania,
Italia) MÁS la Champions League, que es la que enlaza los equipos de distintas
ligas en una sola escala (el reto del fútbol de clubes). Devuelve un DataFrame
homogéneo (date, home_team, away_team, home_score, away_score, competition,
neutral, weight_importance) apto para src.model.dixon_coles.
"""
import re
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/openfootball"
SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
# repo -> (archivo, etiqueta de competición)
LIGAS = {
    "espana": ("1-liga.txt", "ESP"),
    "england": ("1-premierleague.txt", "ENG"),
    "deutschland": ("1-bundesliga.txt", "GER"),
    "italy": ("1-seriea.txt", "ITA"),
    "austria": ("1-bundesliga.txt", "AUT"),
    "champions-league": ("cl.txt", "CL"),
}

_MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_DIA = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?")
# Liga doméstica: <local>  <hs>-<as> [ (ht) ]  <visita>
_PARTIDO = re.compile(
    r"^(?:\d{1,2}[:.]\d{2}\s+)?(?P<home>[^\d].*?)\s+(?P<hs>\d{1,2})-(?P<as>\d{1,2})"
    r"(?:\s*\(\d+-\d+\))?\s+(?P<away>[^\d(].*?)\s*$")
# Champions: <local> (PAÍS)  v  <visita> (PAÍS)  <hs>-<as> [ (ht) ]
_PARTIDO_CL = re.compile(
    r"^(?:\d{1,2}:\d{2}\s+)?(?P<home>.+?)\s+\([A-Z]{3}\)\s+v\s+"
    r"(?P<away>.+?)\s+\([A-Z]{3}\)\s+(?P<hs>\d{1,2})-(?P<as>\d{1,2})"
    r"(?:\s*\(\d+-\d+\))?\s*$")


# tokens de tipo de club a eliminar de los extremos del nombre para unificar
# variantes ("Arsenal FC"/"Arsenal", "FC Barcelona"/"Barcelona", "Real Madrid CF"/"C.F.")
_TOKENS = {"fc", "cf", "afc", "ac", "ssc", "sc", "cd", "ud", "rc", "rcd", "sl",
           "as", "sk", "kv", "bc", "bsc", "fk", "rb", "bv", "us", "ssd", "cp",
           "calcio", "1", "sd", "as", "ac", "sv", "vfb", "vfl", "tsg", "sco"}
_ALIAS = {
    "Internazionale Milano": "Inter", "Internazionale": "Inter",
    "Bayern München": "Bayern Munich", "Bayern Múnich": "Bayern Munich",
    "Atlético de Madrid": "Atlético Madrid", "Atletico de Madrid": "Atlético Madrid",
    "Sporting Clube de Portugal": "Sporting CP", "Sport Lisboa e Benfica": "Benfica",
    "Paris Saint-Germain": "Paris SG", "Paris Saint Germain": "Paris SG",
}


def normalize_team(name: str) -> str:
    n = name.replace(".", "").strip()  # "C.F."->"CF", "1."->"1"
    parts = n.split()
    while parts and parts[0].lower() in _TOKENS:
        parts.pop(0)
    while parts and parts[-1].lower() in _TOKENS:
        parts.pop()
    base = " ".join(parts).strip() or name.strip()
    return _ALIAS.get(base, base)


def _url(repo: str, season: str) -> str:
    archivo = LIGAS[repo][0]
    return f"{BASE}/{repo}/master/{season}/{archivo}"


def download_all(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for repo in LIGAS:
        for season in SEASONS:
            try:
                r = requests.get(_url(repo, season), timeout=30,
                                 headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                (dest_dir / f"{repo}_{season}.txt").write_text(r.text, encoding="utf-8")
            except Exception:
                continue


def _parse_file(text: str, comp: str, season: str) -> list:
    filas = []
    year_ini = int(season[:4])
    cur_year = year_ini
    cur_date = None
    stage = "league"
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("▪"):
            stage = "league" if "League" in s or "Matchday" in s else "ko"
            continue
        md = _DIA.match(s)
        if md:
            mes = _MESES.get(md.group(2))
            dia = int(md.group(3))
            yr = int(md.group(4)) if md.group(4) else (
                year_ini if mes >= 7 else year_ini + 1)  # temporada cruza el año
            cur_year = yr
            if mes:
                cur_date = pd.Timestamp(year=cur_year, month=mes, day=dia)
            continue
        if comp == "CL":
            # saltar eliminatorias (prórroga/penales/agregado): forma no comparable
            if "a.e.t" in s or "pen." in s or " agg" in s:
                continue
            pm = _PARTIDO_CL.match(s)
        else:
            pm = _PARTIDO.match(s)
        if pm and cur_date is not None:
            home = pm.group("home").strip()
            away = pm.group("away").strip()
            # descartar líneas de goleadores u otras sin nombres plausibles
            if len(home) < 2 or len(away) < 2 or "'" in home or "'" in away:
                continue
            home, away = normalize_team(home), normalize_team(away)
            filas.append({
                "date": cur_date, "home_team": home, "away_team": away,
                "home_score": int(pm.group("hs")), "away_score": int(pm.group("as")),
                "competition": comp, "stage": stage if comp == "CL" else "league",
                "neutral": False, "weight_importance": 1.0,
            })
    return filas


def load_training(data_dir: Path) -> pd.DataFrame:
    filas = []
    for repo, (_, comp) in LIGAS.items():
        for season in SEASONS:
            f = data_dir / f"{repo}_{season}.txt"
            if f.exists():
                filas += _parse_file(f.read_text(encoding="utf-8"), comp, season)
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
    return df.sort_values("date").reset_index(drop=True)


def recent_form(df: pd.DataFrame, team: str, n: int = 5) -> str:
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    rows = df[mask].sort_values("date", ascending=False).head(n)
    out = []
    for r in rows.itertuples():
        local = r.home_team == team
        rival = r.away_team if local else r.home_team
        gf, gc = (r.home_score, r.away_score) if local else (r.away_score, r.home_score)
        res = "G" if gf > gc else ("E" if gf == gc else "P")
        out.append(f"- {r.date.date()} {r.competition} vs {rival}: {gf}-{gc} ({res})")
    return "\n".join(out) if out else "- sin partidos recientes"
