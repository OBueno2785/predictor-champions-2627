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
# ligas de los demás clubes de la Champions: comp -> (repo, carpeta, código de archivo)
# (europe/<carpeta>/<temporada>_<código>.txt; Bélgica: belgium/<temporada>/be1.txt)
LIGAS_EXTRA = {
    "FRA": ("europe", "france", "fr1"), "POR": ("europe", "portugal", "pt1"),
    "NED": ("europe", "netherlands", "nl1"), "TUR": ("europe", "turkey", "tr1"),
    "GRE": ("europe", "greece", "gr1"), "NOR": ("europe", "norway", "no1"),
    "CZE": ("europe", "czech-republic", "cz1"), "UKR": ("europe", "ukraine", "ua1"),
    "SVK": ("europe", "slovakia", "sk1"), "AZE": ("europe", "azerbaijan", "az1"),
    "BEL": ("belgium", None, "be1"),
}
CALENDARIO = {"NOR"}  # temporada = año calendario ("2025"), no "2025-26"
# copas europeas (enlazan las ligas entre sí): archivo de champions-league -> comp
COPAS = {"clq.txt": "CLQ", "el.txt": "EL", "elq.txt": "ELQ", "conf.txt": "UECL", "confq.txt": "UECLQ"}
EUROPEAS = {"CL", *COPAS.values()}


def temporadas(comp: str) -> list:
    return [s[:4] for s in SEASONS] if comp in CALENDARIO else SEASONS


def _url_extra(comp: str, season: str) -> str:
    repo, carpeta, cod = LIGAS_EXTRA[comp]
    if repo == "belgium":
        return f"{BASE}/belgium/master/{season}/{cod}.txt"
    return f"{BASE}/europe/master/{carpeta}/{season}_{cod}.txt"


def _archivos() -> list:
    """(url, nombre local, comp, temporada) de todo lo que se descarga."""
    out = [(_url(repo, s), f"{repo}_{s}.txt", comp, s)
           for repo, (_, comp) in LIGAS.items() for s in SEASONS]
    out += [(_url_extra(comp, s), f"of_{comp}_{s}.txt", comp, s)
            for comp in LIGAS_EXTRA for s in temporadas(comp)]
    out += [(f"{BASE}/champions-league/master/{s}/{arch}", f"of_{comp}_{s}.txt", comp, s)
            for arch, comp in COPAS.items() for s in SEASONS]
    return out

_MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_DIA = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?")
# Liga doméstica: <local>  <hs>-<as> [ (ht) ]  <visita>
_PARTIDO = re.compile(
    r"^(?:\d{1,2}[:.]\d{2}\s+)?(?P<home>(?:\d\.\s+)?[^\d].*?)\s+(?P<hs>\d{1,2})-(?P<as>\d{1,2})"
    r"(?:\s*\(\d+-\d+\))?\s+(?P<away>(?:\d\.\s+)?[^\d(].*?)\s*$")
# Liga doméstica, formato nuevo (2024-25, 2026-27): <local>  v  <visita>  <hs>-<as> [ (ht) ]
_PARTIDO_V = re.compile(
    r"^(?:\d{1,2}[:.]\d{2}\s+)?(?P<home>.+?)\s+v\s+(?P<away>.+?)\s+"
    r"(?P<hs>\d{1,2})-(?P<as>\d{1,2})(?:\s*\(\d+-\d+\))?\s*$")
# líneas de goleadores: "(Kane 23', Musiala 45+2')"
_GOLEADOR = re.compile(r"\d'")
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
    # variantes entre temporadas/formatos de openfootball (la 2025-26 usa nombres
    # cortos y el resto largos): sin unificar, un club se ratea como dos equipos
    "Borussia M'gladbach": "Borussia Mönchengladbach",
    "Espanyol de Barcelona": "Espanyol", "Como 1907": "Como",
    "Parma Calcio 1913": "Parma", "St Pauli 1910": "St Pauli",
    "Heidenheim 1846": "Heidenheim",
    "Club Atlético de Madrid": "Atlético Madrid", "Real Betis Balompié": "Real Betis",
    "Real Sociedad de Fútbol": "Real Sociedad", "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Celta de Vigo": "Celta", "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "Bologna FC 1909": "Bologna", "SS Lazio": "Lazio", "ACF Fiorentina": "Fiorentina",
    "Genoa CFC": "Genoa", "Red Bull Salzburg": "Salzburg", "LASK Linz": "LASK",
    "Blau Weiß Linz": "Blau-Weiß Linz",
    # ligas nuevas y copas europeas (cada temporada de openfootball/europe usa otra
    # variante; revisado liga por liga el 5-nov-2026)
    "Racing Club de Lens": "Lens", "Lille": "Lille OSC", "Olympique Marseille": "Olympique de Marseille",
    "Stade Brestois 29": "Stade Brestois", "AJ Auxerre": "Auxerre", "Rennes": "Stade Rennais",
    "Stade Rennais FC 1901": "Stade Rennais", "Strasbourg Alsace": "Strasbourg",
    "Sporting": "Sporting CP", "Sporting Braga": "Braga", "Sporting Clube de Braga": "Braga",
    "AZ Alkmaar": "AZ", "Twente '65": "Twente", "Feyenoord": "Feyenoord Rotterdam",
    "Atiker Konyaspor": "Konyaspor", "Aytemiz Alanyaspor": "Alanyaspor",
    "Gençlerbirligi": "Gençlerbirliği", "Kasimpasa": "Kasımpaşa",
    "AEK Athen": "AEK", "AEK Athens": "AEK", "PAOK": "PAOK Saloniki", "Aris Thessaloniki": "Aris Saloniki",
    "Kifisia": "AE Kifisias", "OFI": "OFI Heraklion", "Volos": "Volos NFC",
    "PAE Olympiakos SFP": "Olympiakos Piraeus", "Royale Union Saint-Gilloise": "Union Saint-Gilloise",
    "1FC Slovácko": "Slovácko", "Bohemians 1905": "Bohemians Praha 1905", "Fastav Zlín": "Zlín",
    "Hradec Kralove": "Hradec Králové", "Mlada Boleslav": "Mladá Boleslav",
    "ŠK Slovan Bratislava": "Slovan Bratislava", "Sumgayit": "Sumgayıt", "Qarabağ Ağdam": "Qarabağ",
    "Lazio Roma": "Lazio", "KF Drita": "Drita Gjilan", "GNK Dinamo Zagreb": "Dinamo Zagreb",
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
    for url, nombre, _, _ in _archivos():
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404:
                continue
            r.raise_for_status()
            (dest_dir / nombre).write_text(r.text, encoding="utf-8")
        except Exception:
            continue


_ESPERADOS = re.compile(r"^#\s*Matches\s+(\d+)", re.M)


def esperados(text: str) -> int | None:
    """Partidos que anuncia la cabecera del archivo (# Matches N)."""
    m = _ESPERADOS.search(text)
    return int(m.group(1)) if m else None


def _parse_file(text: str, comp: str, season: str) -> list:
    filas = []
    year_ini = int(season[:4])
    calendario = "-" not in season
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
            if md.group(4):
                yr = int(md.group(4))
            elif calendario:
                yr = cur_year
            else:
                yr = year_ini if mes >= 7 else year_ini + 1  # temporada cruza el año
            cur_year = yr
            if mes:
                cur_date = pd.Timestamp(year=cur_year, month=mes, day=dia)
            continue
        if "[" in s:  # [awarded], [annulled]...: resultado de mesa, no comparable
            continue
        if "a.e.t" in s or "pen." in s:  # prórroga o penales: no comparable con 90'
            continue
        if comp in EUROPEAS:
            # saltar eliminatorias (prórroga/penales/agregado): forma no comparable
            if "a.e.t" in s or "pen." in s or " agg" in s:
                continue
            pm = _PARTIDO_CL.match(s)
        else:
            pm = _PARTIDO_V.match(s) or _PARTIDO.match(s)
        if pm and cur_date is not None:
            home = pm.group("home").strip()
            away = pm.group("away").strip()
            # descartar líneas de goleadores u otras sin nombres plausibles
            # (no basta con "'": hay clubes como "Borussia M'gladbach")
            if len(home) < 2 or len(away) < 2 or home.startswith("(") \
                    or " v " in home or " v " in away \
                    or _GOLEADOR.search(home) or _GOLEADOR.search(away):
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
    """Todo el historial. Si una liga-temporada de openfootball está incompleta y
    src.ingest.tsdb la completó (data/tsdb_<comp>_<temporada>.json), se usa esa."""
    import json
    filas = []
    for _, nombre, comp, season in _archivos():
        reemplazo = data_dir / f"tsdb_{comp}_{season}.json"
        if reemplazo.exists():
            for r in json.loads(reemplazo.read_text(encoding="utf-8")):
                filas.append({**r, "date": pd.Timestamp(r["date"]), "competition": comp,
                              "stage": "league", "neutral": False, "weight_importance": 1.0})
            continue
        f = data_dir / nombre
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
