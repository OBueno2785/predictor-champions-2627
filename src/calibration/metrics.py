"""Métricas de evaluación y calibración para predicciones 1X2.

Codificación de resultado: 0 = gana local, 1 = empate, 2 = gana visita.
`probs` siempre es array (n, 3) en ese orden.
"""
import numpy as np

N_BINS = 10


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score (menor es mejor). Estándar para 1X2."""
    obs = np.zeros_like(probs)
    obs[np.arange(len(outcomes)), outcomes] = 1.0
    cdf_p = np.cumsum(probs, axis=1)[:, :2]
    cdf_o = np.cumsum(obs, axis=1)[:, :2]
    return float(((cdf_p - cdf_o) ** 2).sum(axis=1).mean() / 2)


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    obs = np.zeros_like(probs)
    obs[np.arange(len(outcomes)), outcomes] = 1.0
    return float(((probs - obs) ** 2).sum(axis=1).mean())


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, 1)
    return float(-np.log(p).mean())


def accuracy_1x2(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == outcomes).mean())


def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """ECE one-vs-rest. Umbral de alerta: > 0.05."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def spiegelhalter_z(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """|Z| > 1.96 => descalibrado (p < 0.05)."""
    num = np.sum((y_true - y_prob) * (1 - 2 * y_prob))
    denom = np.sqrt(np.sum((1 - 2 * y_prob) ** 2 * y_prob * (1 - y_prob)))
    return float(num / denom) if denom > 0 else 0.0


def calibration_report(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    """Métricas globales + calibración por clase (one-vs-rest)."""
    rep = {
        "n": len(outcomes),
        "rps": rps(probs, outcomes),
        "brier": brier(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "acc_1x2": accuracy_1x2(probs, outcomes),
    }
    for k, label in enumerate(["home", "draw", "away"]):
        y = (outcomes == k).astype(float)
        rep[f"ece_{label}"] = ece_binary(y, probs[:, k])
        rep[f"z_{label}"] = spiegelhalter_z(y, probs[:, k])
    rep["ece_macro"] = float(np.mean([rep["ece_home"], rep["ece_draw"], rep["ece_away"]]))
    return rep
