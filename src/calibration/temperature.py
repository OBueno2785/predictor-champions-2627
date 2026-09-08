"""Temperature scaling sobre vectores de probabilidad 1X2.

p_i' = p_i^(1/T) / sum_j p_j^(1/T).  T > 1 suaviza (modelo sobreconfiado),
T < 1 agudiza (modelo subconfiado). Un solo parámetro: robusto con pocas
muestras, no altera el orden de las clases (preserva refinamiento).
"""
import numpy as np
from scipy.optimize import minimize_scalar


def apply(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0) ** (1.0 / T)
    return p / p.sum(axis=-1, keepdims=True)


def fit(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Minimiza log-loss en validación. Devuelve T óptimo."""
    def nll(T):
        p = apply(probs, T)
        return -np.log(np.clip(p[np.arange(len(outcomes)), outcomes], 1e-12, 1)).mean()

    res = minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded")
    return float(res.x)
