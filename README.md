# UABT — Unified Archetypal Bayesian Theory (v1.0)

Quantifies *Anomalus Phenomena Clusters (APCs)* via a transparent, reproducible Bayesian-style engine.

- Posterior odds:  Posterior_Odds = Prior_Odds × Π(LR_i) × Π(ω_j) × M_F5
- Posterior prob:  P_post = Posterior_Odds / (1 + Posterior_Odds)
- Forecast (per domain):  P_i(t+H) = P_i(past) × [1 + T_i × V_i − E_i]

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uabt.cli --config configs/uabt_params.json --seed 42 --mc 1000000
