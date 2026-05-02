"""Conditional logit estimation pipeline for residential location choice.

Estimates beta coefficients from observed HDB resale transactions using
McFadden's conditional logit model. The four utility attributes are:

    V_z = beta_1 * (price_z / income) + beta_2 * commute_z
        + beta_3 * facilities_z + beta_4 * amenity_z

where beta_1 and beta_2 should be negative (higher cost/commute is worse)
and beta_3 and beta_4 should be positive (better facilities/amenities
are preferred).

Literature fallback coefficients are calibrated from:
  - Phang & Wong (1997): Singapore housing demand elasticity
  - Lerman (1977): commute disutility in mode choice
  - Bayer et al. (2007): residential sorting on neighbourhood amenities
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from agent_urban_planning.research.data_base import normalize_town_name


# ------------------------------------------------------------------
# EstimationResult
# ------------------------------------------------------------------

@dataclass
class EstimationResult:
    """Holds estimated (or literature-fallback) conditional logit coefficients."""

    beta_price_income_ratio: float   # beta_1 -- should be negative
    beta_commute_minutes: float      # beta_2 -- should be negative
    beta_facilities_per_capita: float  # beta_3 -- should be positive
    beta_amenity: float              # beta_4 -- should be positive
    std_errors: dict[str, float]
    log_likelihood: float
    n_observations: int
    n_alternatives: int
    price_elasticity_eta: float
    estimated: bool                  # True if from data, False if literature fallback
    estimation_date: str
    data_source: str

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EstimationResult":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_file(self, path: str) -> None:
        """Write JSON to a file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, path: str) -> "EstimationResult":
        """Read from a JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())


# ------------------------------------------------------------------
# Income proxy from flat type
# ------------------------------------------------------------------

def flat_type_to_income_proxy(flat_type: str) -> float:
    """Map HDB flat type to approximate monthly household income (SGD).

    Based on HDB eligibility brackets and DOS Household Income Survey:
      - 1-2 ROOM: low-income rental/subsidized (below S$2,000/month)
      - 3 ROOM: young couples / lower-middle (S$4,500 median)
      - 4 ROOM: median household (S$7,500)
      - 5 ROOM / EXECUTIVE: upper-middle (S$11,000)
    """
    upper = flat_type.strip().upper()
    if upper in ("1 ROOM", "2 ROOM"):
        return 2000.0
    if upper == "3 ROOM":
        return 4500.0
    if upper == "4 ROOM":
        return 7500.0
    if upper in ("5 ROOM", "EXECUTIVE"):
        return 11000.0
    # Anything else (MULTI-GENERATION, etc.) -- national median
    return 5500.0


# ------------------------------------------------------------------
# Expected commute from employment shares
# ------------------------------------------------------------------

def compute_expected_commute(
    zone_name: str,
    employment_shares: dict[str, float],
    transport_network,
) -> float:
    """Compute employment-weighted expected commute from a residential zone.

    Args:
        zone_name: The residential zone to compute commute from.
        employment_shares: {zone: share} summing to ~1.0, representing
            the fraction of jobs in each zone.
        transport_network: Has get_best_route(from_zone, to_zone) method
            returning a route object with .time_minutes, or None.

    Returns:
        Expected commute in minutes. If no routes exist for any employment
        zone, returns 60.0 as a conservative default.
    """
    total_time = 0.0
    total_share = 0.0

    for emp_zone, share in employment_shares.items():
        if share <= 0:
            continue
        route = transport_network.get_best_route(zone_name, emp_zone)
        if route is not None:
            total_time += share * route.time_minutes
            total_share += share

    if total_share <= 0:
        return 60.0  # conservative default if no routes found
    # Normalize by actual reachable share
    return total_time / total_share


# ------------------------------------------------------------------
# Build estimation dataset
# ------------------------------------------------------------------

def build_estimation_dataset(
    transactions: list[dict],
    zone_characteristics: dict[str, dict],
    transport_network,
    employment_shares: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the feature matrix for conditional logit estimation.

    Args:
        transactions: List of dicts with keys: town, flat_type, resale_price.
        zone_characteristics: {zone_name: {housing_base_price, amenity_score,
            facilities_avg_quality}} for all zones in the choice set.
        transport_network: Network with get_best_route() method.
        employment_shares: {zone: share} of job distribution.

    Returns:
        X: array of shape [n_obs, n_zones, 4] -- feature matrix
        y: array of shape [n_obs] -- chosen zone index per observation
        n_zones: number of alternatives
    """
    zone_names = sorted(zone_characteristics.keys())
    n_zones = len(zone_names)
    zone_idx = {name: i for i, name in enumerate(zone_names)}

    # Pre-compute per-zone commutes (shared across all observations)
    zone_commutes = {}
    for z in zone_names:
        zone_commutes[z] = compute_expected_commute(z, employment_shares, transport_network)

    # Filter transactions to those whose town maps to a known zone
    valid_obs = []
    for txn in transactions:
        town_raw = txn["town"]
        chosen_zone = normalize_town_name(town_raw)
        if chosen_zone is None or chosen_zone not in zone_idx:
            continue

        income = flat_type_to_income_proxy(txn["flat_type"])
        resale_price = txn["resale_price"]

        # Convert resale price to monthly equivalent (25-year mortgage at 2.6% HDB rate)
        # monthly_payment = P * r / (1 - (1+r)^{-n})
        monthly_rate = 0.026 / 12
        n_payments = 25 * 12
        if monthly_rate > 0:
            monthly_payment = resale_price * monthly_rate / (1 - (1 + monthly_rate) ** (-n_payments))
        else:
            monthly_payment = resale_price / n_payments

        valid_obs.append((chosen_zone, income, monthly_payment))

    n_obs = len(valid_obs)
    if n_obs == 0:
        return np.zeros((0, n_zones, 4)), np.zeros(0, dtype=int), n_zones

    X = np.zeros((n_obs, n_zones, 4))
    y = np.zeros(n_obs, dtype=int)

    for i, (chosen_zone, income, _monthly_payment) in enumerate(valid_obs):
        y[i] = zone_idx[chosen_zone]

        for j, z in enumerate(zone_names):
            zc = zone_characteristics[z]
            price_z = zc["housing_base_price"]  # monthly equivalent

            X[i, j, 0] = price_z / income                   # price/income ratio
            X[i, j, 1] = zone_commutes[z]                   # expected commute (minutes)
            X[i, j, 2] = zc["facilities_avg_quality"]       # facilities density
            X[i, j, 3] = zc["amenity_score"]                # amenity score

    return X, y, n_zones


# ------------------------------------------------------------------
# Conditional logit log-likelihood and gradient
# ------------------------------------------------------------------

def clogit_log_likelihood(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Compute the negative log-likelihood of the conditional logit model.

    Args:
        beta: array of 4 coefficients [beta_1, beta_2, beta_3, beta_4].
        X: [n_obs, n_zones, 4] feature matrix.
        y: [n_obs] chosen zone index per observation.

    Returns:
        Negative log-likelihood (for minimization).

    Uses the logsumexp trick for numerical stability:
        LL = sum_i [ V_{i,chosen} - log(sum_z exp(V_{iz})) ]
    """
    n_obs = X.shape[0]
    # V[i, z] = X[i, z, :] @ beta -> shape [n_obs, n_zones]
    V = X @ beta  # broadcasting: [n_obs, n_zones, 4] @ [4] -> [n_obs, n_zones]

    # V for the chosen alternative
    V_chosen = V[np.arange(n_obs), y]  # [n_obs]

    # Log-sum-exp across alternatives for each observation
    log_denom = logsumexp(V, axis=1)  # [n_obs]

    # LL = sum(V_chosen - log_denom)
    ll = np.sum(V_chosen - log_denom)

    return -ll  # negative for minimization


def clogit_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the analytic gradient of the negative log-likelihood.

    Gradient of LL w.r.t. beta_k:
        dLL/dbeta_k = sum_i [ X_{i,chosen,k} - sum_z P(z|i) * X_{i,z,k} ]

    where P(z|i) = exp(V_{iz}) / sum_z' exp(V_{iz'})

    Returns the gradient of the NEGATIVE log-likelihood.
    """
    n_obs = X.shape[0]
    V = X @ beta  # [n_obs, n_zones]

    # Softmax probabilities P(z|i) for each observation
    # Numerically stable: subtract max per observation
    V_max = V.max(axis=1, keepdims=True)
    exp_V = np.exp(V - V_max)
    probs = exp_V / exp_V.sum(axis=1, keepdims=True)  # [n_obs, n_zones]

    # X for chosen alternatives: X[i, y[i], :] -> [n_obs, 4]
    X_chosen = X[np.arange(n_obs), y, :]  # [n_obs, 4]

    # Expected X under the model: sum_z P(z|i) * X[i,z,:] -> [n_obs, 4]
    # probs: [n_obs, n_zones], X: [n_obs, n_zones, 4]
    X_expected = np.einsum("ij,ijk->ik", probs, X)  # [n_obs, 4]

    # Gradient of LL: sum over observations
    grad_ll = np.sum(X_chosen - X_expected, axis=0)  # [4]

    return -grad_ll  # negative for minimization


# ------------------------------------------------------------------
# Full estimation pipeline
# ------------------------------------------------------------------

def estimate_choice_model(
    transactions: list[dict],
    zone_characteristics: dict[str, dict],
    transport_network,
    employment_shares: dict[str, float],
) -> EstimationResult:
    """Estimate the conditional logit model from HDB transaction data.

    Args:
        transactions: List of HDB resale transactions.
        zone_characteristics: Per-zone features dict.
        transport_network: Transport network with get_best_route().
        employment_shares: Job distribution across zones.

    Returns:
        EstimationResult with estimated coefficients.
    """
    X, y, n_zones = build_estimation_dataset(
        transactions, zone_characteristics, transport_network, employment_shares,
    )

    n_obs = X.shape[0]
    if n_obs == 0:
        raise ValueError("No valid observations for estimation.")

    # Initial guess: literature-informed starting values
    beta0 = np.array([-2.0, -0.015, 0.5, 0.8])

    result = minimize(
        clogit_log_likelihood,
        beta0,
        args=(X, y),
        method="L-BFGS-B",
        jac=clogit_gradient,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    beta_hat = result.x

    # Standard errors from inverse Hessian
    # L-BFGS-B returns hess_inv as an LbfgsInvHessProduct; convert to array
    try:
        hess_inv = result.hess_inv
        if hasattr(hess_inv, "todense"):
            hess_inv_arr = np.array(hess_inv.todense())
        else:
            # LbfgsInvHessProduct: multiply by identity to get full matrix
            n_params = len(beta_hat)
            hess_inv_arr = hess_inv @ np.eye(n_params)
        se = np.sqrt(np.maximum(np.diag(hess_inv_arr), 0.0))
    except Exception:
        se = np.zeros(len(beta_hat))

    # Price elasticity: eta = beta_1 * mean(price/income) * (1 - 1/n_zones)
    mean_price_income = X[:, :, 0].mean()
    eta = beta_hat[0] * mean_price_income * (1 - 1 / n_zones)

    param_names = [
        "beta_price_income_ratio",
        "beta_commute_minutes",
        "beta_facilities_per_capita",
        "beta_amenity",
    ]
    std_errors = {name: float(se[i]) for i, name in enumerate(param_names)}

    return EstimationResult(
        beta_price_income_ratio=float(beta_hat[0]),
        beta_commute_minutes=float(beta_hat[1]),
        beta_facilities_per_capita=float(beta_hat[2]),
        beta_amenity=float(beta_hat[3]),
        std_errors=std_errors,
        log_likelihood=float(-result.fun),
        n_observations=n_obs,
        n_alternatives=n_zones,
        price_elasticity_eta=float(eta),
        estimated=True,
        estimation_date=date.today().isoformat(),
        data_source="HDB resale transactions (data.gov.sg)",
    )


# ------------------------------------------------------------------
# Literature fallback
# ------------------------------------------------------------------

def literature_fallback() -> EstimationResult:
    """Return coefficients calibrated from published housing economics studies.

    This is the RECOMMENDED default for paper-quality simulations. Each
    coefficient traces to a published study, though the connection
    involves calibration judgments documented below.

    Sources and calibration methodology:

      β₁ = -2.0  (price/income ratio)
        Source: Phang, S.Y. & Wong, W.K. (1997), "Government Policies
        and Private Housing Prices in Singapore", Urban Studies 34(11).
        Method: Phang & Wong estimated Singapore HDB demand elasticity
        η ≈ -0.5. We back-calculate β₁ from η = β₁ × mean(price/income)
        × (1 - 1/n_zones) with mean(price/income) ≈ 0.26 for the
        Singapore scenario. This is the most grounded coefficient.

      β₂ = -0.015  (commute minutes)
        Source: Lerman, S.R. (1977), "Location, Housing, Auto Ownership,
        and Mode to Work", Transportation Research Record 610.
        Method: Lerman estimated commute disutility in US mode choice
        models. The -0.015/minute is a rough translation of his
        value-of-time estimates into our utility scale. NOT a Singapore
        estimate — cross-context adaptation.

      β₃ = 0.5  (facilities per capita)
        Source: Bayer, P., Ferreira, F. & McMillan, R. (2007), "A
        Unified Framework for Measuring Preferences for Schools and
        Neighborhoods", Journal of Political Economy 115(4).
        Method: β₃/β₁ is set to approximately match the services/price
        ratio in Bayer et al.'s US residential choice estimates.
        NOT a Singapore estimate — proportional scaling from US data.

      β₄ = 0.8  (amenity score)
        Source: Same Bayer et al. (2007).
        Method: β₄/β₁ is set to match their neighbourhood-quality/price
        ratio. NOT a Singapore estimate.

    Limitations:
      - Only β₁ is traceable to a Singapore-specific study (via η).
      - β₂, β₃, β₄ are cross-context adaptations from US/general studies.
      - Sensitivity analysis recommended: vary β₂–β₄ by ±50% to check
        robustness of simulation conclusions.
    """
    return EstimationResult(
        beta_price_income_ratio=-2.0,
        beta_commute_minutes=-0.015,
        beta_facilities_per_capita=0.5,
        beta_amenity=0.8,
        std_errors={
            "beta_price_income_ratio": 0.0,
            "beta_commute_minutes": 0.0,
            "beta_facilities_per_capita": 0.0,
            "beta_amenity": 0.0,
        },
        log_likelihood=0.0,
        n_observations=0,
        n_alternatives=27,
        price_elasticity_eta=-0.5,
        estimated=False,
        estimation_date=date.today().isoformat(),
        data_source="Literature calibration (Phang & Wong 1997, Lerman 1977, Bayer et al. 2007)",
    )
