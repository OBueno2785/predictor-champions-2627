# Predictor — Fase de Liga Champions League 2026-27

Réplica del mecanismo del Predictor Mundial 2026, adaptado al fútbol de clubes y
al **formato suizo** de la Champions (desde 2024-25): 36 equipos, una sola tabla,
8 partidos cada uno. Clasifican 1-8 a octavos directo, 9-24 a un playoff de
eliminación, y 25-36 quedan fuera.

## Qué se reutiliza (motor, agnóstico al deporte)

Copiado tal cual del Mundial:
- `src/model/dixon_coles.py` — Poisson bivariado con corrección DC, MLE con
  gradiente analítico, decaimiento temporal.
- `src/calibration/` — ECE, Spiegelhalter Z, RPS, temperature scaling.
- `src/blend.py` — mezcla mecánica modelo+mercado (pool lineal) + `blend_final`
  con override consciente del mercado por bajas confirmadas + `score_summary`.
- `src/agents/schemas.py`, `src/agents/debate.py` — debate multiagente vía
  `claude -p` headless (sin API key), Juez con deltas log-xG acotados.

## Qué es nuevo (específico de clubes)

- `src/ingest/openfootball.py` — resultados de clubes desde openfootball (GitHub
  raw). Reúne **ligas domésticas** (España, Inglaterra, Alemania, Italia) +
  **Champions**, que es la que **enlaza equipos de distintas ligas en una sola
  escala** (el reto del fútbol de clubes: sin partidos entre ligas, cada liga
  queda en su propia escala). Incluye:
  - Parser doble (formato liga vs formato CL con etiquetas de país "(ESP)").
  - **Normalización de nombres** (`normalize_team`): openfootball usa variantes
    para el mismo club ("Arsenal FC"/"Arsenal", "Real Madrid CF"/"C.F.");
    se unifican quitando tokens de tipo de club y con un mapa de alias.
  - Etiqueta de etapa (`stage`: league/ko) para separar fase de liga de
    eliminatorias.
- `src/league.py` — tabla en formato suizo + zonas de clasificación
  (1-8 / 9-24 / 25-36) + salida markdown.
- `src/predict.py` — entrena el modelo y predice cualquier cruce.

## Estado

**Fase 1 — HECHA**: ingesta de clubes + modelo con enlace entre ligas + tabla
suiza. Verificado:
- 3.296 partidos (2023-2026), 130 equipos. Ranking de fuerza sensato (Arsenal,
  PSG, City, Bayern, Real Madrid, Liverpool, Barcelona arriba).
- Predice cualquier cruce: p. ej. Real Madrid vs Man City → 39/25/36, xG 1.6-1.5.
- Reconstruye la tabla de fase de liga CL 25-26 (144 partidos, 36 equipos).

**Fase 2 — PENDIENTE (requiere el sorteo, ~fin agosto 2026)**: los participantes
y el fixture de la CL 26-27 no existen hasta el sorteo. Cuando estén:
1. Cargar el fixture oficial + cuotas (The Odds API tiene mercado de CL).
2. Cablear el pipeline completo ya copiado: calibración (T, goles), mezcla con
   mercado, debate multiagente (prompts adaptados a clubes e incentivos de la
   tabla suiza), override por bajas.
3. Scheduler T-60 + GitHub Actions (idéntico al Mundial).

**Fase 3 — PENDIENTE**: bracket eliminatorio (playoff 9-24 + octavos en adelante),
prórroga/penales, Monte Carlo para P(clasifica) y P(campeón).

## Limitaciones / mejoras conocidas

- **Ligas faltantes**: openfootball no tiene Francia, Portugal ni Países Bajos en
  este pool → PSG, Porto, Benfica, PSV, Ajax, Feyenoord se ratean **solo** por
  sus partidos de Champions (muestra fina, rating ruidoso). Mejora: añadir esas
  ligas (otra fuente) y/o anclar con **clubelo** (Elo cross-liga; su API estaba
  caída al construir esto) como prior de fuerza — el análogo del Elo de
  selecciones en el Mundial.
- Normalización de nombres es heurística; puede quedar algún residual
  ("Club Atlético de Madrid" vs "Atlético Madrid"). Ampliar `_ALIAS` según haga
  falta.

## Uso

```
python -m src.predict                         # ranking + sanity check
python -m src.predict "Real Madrid" "Bayern"  # predice un cruce (subcadena)
python -m src.league                          # reconstruye la tabla CL 25-26
```

## Calendario CL 2026-27

Sorteo de la fase de liga: ~fin de agosto 2026. Partidos: mediados de septiembre
2026 a fin de enero 2027. Playoffs y octavos: febrero-marzo 2027.
