# Pronósticos Champions League 2026-27 — Jornada 3

Marcador y probabilidades 1X2 del motor (Dixon-Coles) **calibrado y mezclado con el mercado**.

- **1X2 (ganador)**: *linear pool* 75% mercado / 25% modelo — mediana de las casas, sin margen.
- **Marcador (volumen de goles)**: matriz de goles con ritmos mezclados modelo+mercado a un
  peso **ajustado por datos** (`w_score = 0.60`), fiteado minimizando el error de
  marcador (log-loss del resultado exacto) sobre los partidos jugados, con contracción y tope
  de movimiento por muestra chica. Los ritmos del mercado salen del 1X2 + Over/Under (mediana
  de casas). El marcador mostrado es el más probable **dentro** del 1X2 dominante (consistente
  con el ganador).
- **Debate multiagente** (plantel/noticias + sentimiento + juez): ajusta el ritmo de gol ante
  bajas/alineaciones que el mercado aún no refleja (⚑ = override por información no precificada).

Hora de Lima (America/Lima).

### Jornada 3 · Día 1 — martes 20/10/2026

| Hora | Partido | Local | Empate | Visita | Marcador | Señal del debate |
|------|---------|:---:|:---:|:---:|:---:|---|
| 11:45 | Sabah FK – Borussia Dortmund | 10% | 16% | 74% | **0-2** | sin señal nueva |
| 11:45 | Fenerbahce – Slavia Praha | 65% | 20% | 15% | **2-0** | ⚑ override |
| 14:00 | Manchester City – AEK Athens | 84% | 10% | 5% | **3-0** | sin señal nueva |
| 14:00 | AS Roma – ŠK Slovan Bratislava | 82% | 12% | 6% | **3-0** | ⚑ override |
| 14:00 | VfB Stuttgart – Atlético Madrid | 30% | 24% | 45% | **1-2** | bajas confirmadas |
| 14:00 | Paris Saint Germain – Barcelona | 46% | 20% | 34% | **2-1** | ⚑ override |
| 14:00 | Napoli – Bodø/Glimt | 50% | 24% | 27% | **2-1** | bajas confirmadas |
| 14:00 | Liverpool – Villarreal | 66% | 19% | 15% | **2-1** | ⚑ override |
| 14:00 | Porto – PSV Eindhoven | 65% | 20% | 15% | **2-0** | ⚑ override |

> Nota: en el Día 1 el mercado de totales ya no estaba disponible al re-puntuar (los partidos
> salieron del feed), así que el ajuste de marcador usó solo el 1X2 del mercado.

### Jornada 3 · Día 2 — miércoles 21/10/2026

| Hora | Partido | Local | Empate | Visita | Marcador | Señal del debate |
|------|---------|:---:|:---:|:---:|:---:|---|
| 11:45 | Como – Manchester United | 53% | 22% | 25% | **2-1** | ⚑ override |
| 11:45 | Lille – Galatasaray | 56% | 20% | 23% | **2-1** | bajas confirmadas |
| 14:00 | Bayern Munich – Arsenal | 41% | 23% | 37% | **2-1** | bajas confirmadas |
| 14:00 | Aston Villa – Viking FK | 83% | 11% | 6% | **3-0** | ⚑ override |
| 14:00 | Club Brugge – RC Lens | 54% | 21% | 25% | **2-1** | ⚑ override |
| 14:00 | Real Betis – Feyenoord | 64% | 18% | 18% | **2-1** | bajas confirmadas |
| 14:00 | Inter Milan – Shakhtar Donetsk | 81% | 12% | 7% | **2-0** | bajas confirmadas |
| 14:00 | Sporting Lisbon – LASK | 84% | 10% | 6% | **2-0** | ⚑ override |
| 14:00 | Real Madrid – RB Leipzig | 81% | 11% | 8% | **2-1** | ⚑ override |
