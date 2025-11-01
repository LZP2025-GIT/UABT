from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .io_utils import load_params
from .engine import config_from_dict, posterior_probability, forecast_probability, run_monte_carlo

def main():
    ap = argparse.ArgumentParser(description="UABT v1.0 — Posterior and Forecast Calculator")
    ap.add_argument("--config", required=True, help="Path to params (.json/.yml/.csv)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for MC")
    ap.add_argument("--mc", type=int, default=None, help="Monte Carlo samples (override)")
    ap.add_argument("--forecast_years", type=int, default=10, help="Forecast horizon (years)")
    ap.add_argument("--dump_samples", action="store_true", help="Save posterior samples CSV")
    args = ap.parse_args()

    params = load_params(args.config)
    cfg = config_from_dict(params)
    if args.mc: cfg.mc_samples = args.mc

    Path("results").mkdir(exist_ok=True)

    # Point estimate (deterministic)
    p_post = posterior_probability(cfg)
    p_future = forecast_probability(cfg, years=args.forecast_years)

    # Monte Carlo for intervals
    mean, (lo, hi), samples = run_monte_carlo(cfg, seed=args.seed, return_samples=args.dump_samples)

    summary = {
        "posterior_point_estimate": p_post,
        "posterior_mc_mean": mean,
        "posterior_95cri": [lo, hi],
        "forecast_years": args.forecast_years,
        "forecast_aggregate_probability": p_future,
        "mc_samples": cfg.mc_samples,
        "seed": args.seed,
    }
    Path("results/uabt_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if samples is not None:
        df = pd.DataFrame({"posterior_samples": samples})
        df.to_csv("results/uabt_posterior_samples.csv", index=False)

if __name__ == "__main__":
    main()
