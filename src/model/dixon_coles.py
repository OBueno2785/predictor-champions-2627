"""Modelo Dixon-Coles: Poisson bivariado con corrección de marcadores bajos.

Ajuste por MLE con ponderación temporal exponencial y por importancia del
partido. Gradiente analítico (la malla de ~250 selecciones lo exige).
Identificabilidad vía penalización L2 leve sobre ataque/defensa.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

XI_DAILY = 0.00095     # decaimiento temporal: vida media ~2 años
L2_PENALTY = 1.0e-3
MAX_GOALS = 10
MIN_MATCHES = 8        # equipos con menos partidos en la ventana quedan fuera


def _tau_factors(lam, mu, rho):
    """Factores de corrección DC para los marcadores (0,0),(0,1),(1,0),(1,1)."""
    return {
        (0, 0): 1.0 - lam * mu * rho,
        (0, 1): 1.0 + lam * rho,
        (1, 0): 1.0 + mu * rho,
        (1, 1): 1.0 - rho,
    }


def matrix_from_rates(lam: float, mu: float, rho: float,
                      max_goals: int = MAX_GOALS) -> np.ndarray:
    """Matriz de marcadores Dixon-Coles a partir de los ritmos de gol y rho.
    Independiente del modelo entrenado (la usa src.reblend sin reentrenar)."""
    goals = np.arange(max_goals + 1)
    P = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    for (i, j), tau in _tau_factors(lam, mu, rho).items():
        P[i, j] *= max(tau, 1e-10)
    return P / P.sum()


@dataclass
class DixonColesModel:
    attack: dict
    defense: dict
    intercept: float
    home_adv: float
    rho: float
    fitted_at: str = ""
    n_matches: int = 0
    teams: list = field(default_factory=list)

    def rates(self, home: str, away: str, adv_side: int = 0,
              delta_home: float = 0.0, delta_away: float = 0.0):
        """adv_side: +1 ventaja para home, -1 para away, 0 neutral.
        delta_*: ajustes en log-xG provenientes del debate multiagente."""
        adv_h = self.home_adv if adv_side == 1 else 0.0
        adv_a = self.home_adv if adv_side == -1 else 0.0
        lam = np.exp(self.intercept + adv_h + self.attack[home] - self.defense[away] + delta_home)
        mu = np.exp(self.intercept + adv_a + self.attack[away] - self.defense[home] + delta_away)
        return lam, mu

    def score_matrix(self, home: str, away: str, adv_side: int = 0,
                     delta_home: float = 0.0, delta_away: float = 0.0,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.rates(home, away, adv_side, delta_home, delta_away)
        return matrix_from_rates(lam, mu, self.rho, max_goals)

    def outcome_probs(self, P: np.ndarray):
        p_home = np.tril(P, -1).sum()
        p_draw = np.trace(P)
        p_away = np.triu(P, 1).sum()
        return p_home, p_draw, p_away

    @staticmethod
    def top_scores(P: np.ndarray, n: int = 3):
        flat = [(P[i, j], i, j) for i in range(P.shape[0]) for j in range(P.shape[1])]
        flat.sort(reverse=True)
        return [(i, j, p) for p, i, j in flat[:n]]

    def strength_ranking(self) -> pd.DataFrame:
        rows = [{"team": t, "attack": self.attack[t], "defense": self.defense[t],
                 "strength": self.attack[t] + self.defense[t]} for t in self.teams]
        return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def fit(df: pd.DataFrame, ref_date=None, xi: float = XI_DAILY,
        l2: float = L2_PENALTY) -> DixonColesModel:
    """df: date, home_team, away_team, home_score, away_score, neutral, weight_importance."""
    # Filtra equipos con muestra mínima (selecciones no-FIFA con red desconectada
    # distorsionan el ranking). Dos pasadas para estabilizar.
    for _ in range(2):
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        ok = set(counts[counts >= MIN_MATCHES].index)
        df = df[df["home_team"].isin(ok) & df["away_team"].isin(ok)]

    ref_date = pd.Timestamp(ref_date) if ref_date else df["date"].max()
    days_ago = (ref_date - df["date"]).dt.days.clip(lower=0).to_numpy()
    w = df["weight_importance"].to_numpy() * np.exp(-xi * days_ago)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: k for k, t in enumerate(teams)}
    n = len(teams)
    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy(dtype=float)
    ag = df["away_score"].to_numpy(dtype=float)
    home_flag = (~df["neutral"]).to_numpy(dtype=float)

    def nll_grad(params):
        att, dfn = params[:n], params[n:2 * n]
        mu0, h = params[2 * n], params[2 * n + 1]
        log_lam = mu0 + h * home_flag + att[hi] - dfn[ai]
        log_mu = mu0 + att[ai] - dfn[hi]
        lam, mu = np.exp(log_lam), np.exp(log_mu)
        nll = -(w * (hg * log_lam - lam)).sum() - (w * (ag * log_mu - mu)).sum() \
            + l2 * (att @ att + dfn @ dfn)
        r_h = w * (hg - lam)
        r_a = w * (ag - mu)
        g_att = -(np.bincount(hi, r_h, n) + np.bincount(ai, r_a, n)) + 2 * l2 * att
        g_dfn = (np.bincount(ai, r_h, n) + np.bincount(hi, r_a, n)) + 2 * l2 * dfn
        g_mu0 = -(r_h.sum() + r_a.sum())
        g_h = -(r_h * home_flag).sum()
        return nll, np.concatenate([g_att, g_dfn, [g_mu0, g_h]])

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = np.log(max(df[["home_score", "away_score"]].to_numpy().mean(), 0.1))
    res = minimize(nll_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 8000, "maxfun": 8000})
    if not res.success:
        raise RuntimeError(f"MLE no convergió: {res.message}")

    att, dfn = res.x[:n], res.x[n:2 * n]
    mu0, h = res.x[2 * n], res.x[2 * n + 1]
    log_lam = mu0 + h * home_flag + att[hi] - dfn[ai]
    log_mu = mu0 + att[ai] - dfn[hi]
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    # rho por verosimilitud perfilada sobre marcadores bajos
    low = (hg <= 1) & (ag <= 1)
    lam_l, mu_l, hg_l, ag_l, w_l = lam[low], mu[low], hg[low], ag[low], w[low]

    def neg_ll_rho(rho):
        tau = np.ones_like(lam_l)
        m00 = (hg_l == 0) & (ag_l == 0)
        m01 = (hg_l == 0) & (ag_l == 1)
        m10 = (hg_l == 1) & (ag_l == 0)
        m11 = (hg_l == 1) & (ag_l == 1)
        tau[m00] = 1 - lam_l[m00] * mu_l[m00] * rho
        tau[m01] = 1 + lam_l[m01] * rho
        tau[m10] = 1 + mu_l[m10] * rho
        tau[m11] = 1 - rho
        if (tau <= 0).any():
            return np.inf
        return -(w_l * np.log(tau)).sum()

    rho = minimize_scalar(neg_ll_rho, bounds=(-0.25, 0.25), method="bounded").x

    return DixonColesModel(
        attack={t: att[idx[t]] for t in teams},
        defense={t: dfn[idx[t]] for t in teams},
        intercept=mu0, home_adv=h, rho=rho,
        fitted_at=str(ref_date.date()), n_matches=len(df), teams=teams,
    )
