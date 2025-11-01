from __future__ import annotations
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

@dataclass
class Domain:
    name: str
    LR: float
    T: float = 0.0
    V: float = 0.0
    E: float = 0.0

@dataclass
class Filters:
    F1_p: float; F1_alpha: float
    F2_p: float; F2_alpha: float
    F3_p: float; F3_alpha: float
    F4_p: float; F4_alpha: float
    p_F5: float; q_F5: float

@dataclass
class Controls:
    omega: List[float]

@dataclass
class Config:
    P_prior: float
    domains: Dict[str, Domain]
    filters: Filters
    controls: Controls
    rho: float = 0.0
    horizon_years: int = 10
    mc_samples: int = 1_000_000

# --- Core math --------------------------------------------------------------

def _prior_odds(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return p / (1 - p)

def _to_prob(odds: float) -> float:
    return odds / (1.0 + odds)

def _reception_multiplier(p: float, alpha: float) -> float:
    # Simple convex up-weight: lower reporting fraction -> higher multiplier
    p = min(max(p, 1e-9), 1.0)
    return 1.0 + alpha * (1.0 / p - 1.0)

def _verification_multiplier(p_F5: float, q_F5: float) -> float:
    # M_F5 = p_F5 / [p_F5 + q_F5 * (1 - p_F5)]
    p = min(max(p_F5, 1e-9), 1.0)
    q = min(max(q_F5, 1e-9), 1.0)
    return p / (p + q * (1.0 - p))

def _composite_omega(omega: List[float]) -> float:
    comp = 1.0
    for w in omega:
        comp *= float(w)
    return comp

def posterior_probability(cfg: Config) -> float:
    po = _prior_odds(cfg.P_prior)

    # reception filters (F1–F4)
    F1 = _reception_multiplier(cfg.filters.F1_p, cfg.filters.F1_alpha)
    F2 = _reception_multiplier(cfg.filters.F2_p, cfg.filters.F2_alpha)
    F3 = _reception_multiplier(cfg.filters.F3_p, cfg.filters.F3_alpha)
    F4 = _reception_multiplier(cfg.filters.F4_p, cfg.filters.F4_alpha)
    reception = F1 * F2 * F3 * F4

    # product of LRs across domains
    lr_prod = 1.0
    for d in cfg.domains.values():
        lr_prod *= max(d.LR, 1e-9)

    # operational controls
    omega = _composite_omega(cfg.controls.omega)

    # verification
    M_F5 = _verification_multiplier(cfg.filters.p_F5, cfg.filters.q_F5)

    odds = po * lr_prod * omega * reception * M_F5
    return _to_prob(odds)

# --- Forecasting ------------------------------------------------------------

def forecast_probability(cfg: Config, years: int) -> float:
    """Aggregate forecast using per-domain trend model and inclusion-exclusion."""
    per_domain = []
    for d in cfg.domains.values():
        base = max(min(d.LR / (1.0 + d.LR), 0.95), 1e-6)  # map LR>1 -> base prob proxy
        growth = 1.0 + d.T * d.V - d.E
        future = max(min(base * (growth ** years), 0.999999), 1e-9)
        per_domain.append(future)
    # Independence approximation (corr absorbed implicitly by omega/reception)
    p_none = 1.0
    for p in per_domain:
        p_none *= (1.0 - p)
    return 1.0 - p_none

# --- Monte Carlo with parameter uncertainty --------------------------------

def run_monte_carlo(cfg: Config, seed: Optional[int] = None, return_samples: bool = False
                   ) -> Tuple[float, Tuple[float, float], Optional[np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = cfg.mc_samples
    samples = np.empty(n, dtype=np.float64)

    # Uncertainty model: +/- 10% jitter on LRs, filters, and omegas; clamp to sensible ranges
    omega_arr = np.array(cfg.controls.omega, dtype=np.float64)

    for i in range(n):
        # jitter LRs
        lr_prod = 1.0
        for d in cfg.domains.values():
            jitter = rng.normal(1.0, 0.05)  # 5% sd
            lr = max(d.LR * jitter, 1e-6)
            lr_prod *= lr

        # jitter filters
        def jclip(x, lo, hi, sd=0.05):
            return float(np.clip(rng.normal(x, x*sd if x>0 else sd), lo, hi))

        F1 = _reception_multiplier(jclip(cfg.filters.F1_p, 1e-6, 1.0),
                                   jclip(cfg.filters.F1_alpha, 0.0, 1.0))
        F2 = _reception_multiplier(jclip(cfg.filters.F2_p, 1e-6, 1.0),
                                   jclip(cfg.filters.F2_alpha, 0.0, 1.0))
        F3 = _reception_multiplier(jclip(cfg.filters.F3_p, 1e-6, 1.0),
                                   jclip(cfg.filters.F3_alpha, 0.0, 1.0))
        F4 = _reception_multiplier(jclip(cfg.filters.F4_p, 1e-6, 1.0),
                                   jclip(cfg.filters.F4_alpha, 0.0, 1.0))
        reception = F1 * F2 * F3 * F4

        pF5 = jclip(cfg.filters.p_F5, 1e-6, 1.0)
        qF5 = jclip(cfg.filters.q_F5, 1e-6, 1.0)
        M_F5 = _verification_multiplier(pF5, qF5)

        omega = 1.0
        for w in omega_arr:
            omega *= jclip(w, 0.0, 1.0)

        odds = _prior_odds(cfg.P_prior) * lr_prod * omega * reception * M_F5
        samples[i] = _to_prob(odds)

    mean = float(np.mean(samples))
    lo, hi = float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
    return (mean, (lo, hi), samples if return_samples else None)

# --- Helpers to build Config from dict -------------------------------------

def config_from_dict(d: Dict) -> Config:
    # metadata ignored by engine; pass-through ok
    prior = d["prior"]["P_prior"]
    doms = {}
    for key, v in d["domains"].items():
        doms[key] = Domain(
            name=v.get("name", key),
            LR=float(v["LR"]),
            T=float(v.get("T", 0.0)),
            V=float(v.get("V", 0.0)),
            E=float(v.get("E", 0.0)),
        )
    f = d["filters"]
    verification = d.get("verification", {})
    filters = Filters(
        F1_p=float(f.get("F1", f).get("p") if isinstance(f.get("F1"), dict) else f.get("F1_p", 0.7)),
        F1_alpha=float(f.get("F1", f).get("alpha") if isinstance(f.get("F1"), dict) else f.get("F1_alpha", 0.5)),
        F2_p=float(f.get("F2", f).get("p") if isinstance(f.get("F2"), dict) else f.get("F2_p", 0.4)),
        F2_alpha=float(f.get("F2", f).get("alpha") if isinstance(f.get("F2"), dict) else f.get("F2_alpha", 0.7)),
        F3_p=float(f.get("F3", f).get("p") if isinstance(f.get("F3"), dict) else f.get("F3_p", 0.5)),
        F3_alpha=float(f.get("F3", f).get("alpha") if isinstance(f.get("F3"), dict) else f.get("F3_alpha", 0.6)),
        F4_p=float(f.get("F4", f).get("p") if isinstance(f.get("F4"), dict) else f.get("F4_p", 0.3)),
        F4_alpha=float(f.get("F4", f).get("alpha") if isinstance(f.get("F4"), dict) else f.get("F4_alpha", 0.8)),
        p_F5=float(verification.get("p_F5", d["verification"]["p_F5"] if "verification" in d else 0.10)),
        q_F5=float(verification.get("q_F5", d["verification"]["q_F5"] if "verification" in d else 0.20)),
    )
    controls = Controls(omega=[float(x) for x in d["controls"]["omega"]])
    rho = float(d.get("correlation", {}).get("rho", 0.0))
    horizon = int(d.get("metadata", {}).get("horizon_years", d.get("horizon_years", 10)))
    mc = int(d.get("metadata", {}).get("mc_samples", d.get("mc_samples", 1_000_000)))
    return Config(P_prior=prior, domains=doms, filters=filters, controls=controls, rho=rho,
                  horizon_years=horizon, mc_samples=mc)
