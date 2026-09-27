"""Evaluate the frozen v19 defensive absolute time-series momentum strategy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.cross_asset_budget import RiskBudgetConfig, run_risk_budget_backtest
from portfolio.optimizer import PortfolioCosts
from scripts.cross_asset_v17_universe import UNIVERSE, asset_sleeves, factor_loadings
from scripts.evaluate_cross_asset_v12 import TRAIN, VALIDATION, load_data
from scripts.evaluate_cross_asset_v13 import passes, segment_metrics
from scripts.evaluate_cross_asset_v15 import calibrate_absolute
from scripts.evaluate_equity_v7 import sha256

PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V19.md")
MOMENTUM_WEIGHTS = {63: 0.25, 126: 0.35, 252: 0.40}
SKIP_SESSIONS = 21
VOLATILITY_WINDOW = 60
MOVING_AVERAGE_WINDOW = 200
SIGNAL_THRESHOLD = 0.25
FACTOR_CAPS = pd.Series(
    {"equity": 0.50, "duration": 0.50, "credit": 0.30, "commodity": 0.50, "usd": 0.40}
)
SLEEVE_CAPS = pd.Series(
    {"equity": 0.60, "rates": 0.60, "credit": 0.40, "metals": 0.40,
     "commodity": 0.60, "usd": 0.30}
)
CONFIG = RiskBudgetConfig(
    rebalance_every=10,
    max_gross=1.0,
    max_name=0.10,
    max_turnover=0.25,
    max_participation=0.001,
    annual_volatility_cap=0.08,
    net_cap=0.60,
)
COSTS = PortfolioCosts(
    commission_bps=0,
    half_spread_bps=1,
    slippage_bps=1,
    impact_coefficient=0.10,
    borrow_annual=0.01,
    calendar_days=7,
)


def normalized_loadings() -> pd.DataFrame:
    """Return frozen coarse macro loadings on comparable scales."""
    frame = factor_loadings()
    return frame.div(frame.abs().max().replace(0, 1), axis=1)


def build_defensive_tsmom(data: dict) -> dict[str, pd.DataFrame]:
    """Build the frozen v19 signal and inverse-volatility sizing score without look-ahead."""
    close = data["close"]
    returns = data["returns"]
    eligibility = data["eligibility"]
    log_price = np.log(close)
    daily_volatility = returns.rolling(
        VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
    ).std().shift(1)

    components = {}
    for horizon, weight in MOMENTUM_WEIGHTS.items():
        move = log_price.shift(SKIP_SESSIONS).sub(log_price.shift(horizon))
        components[horizon] = move.div(
            daily_volatility * np.sqrt(horizon - SKIP_SESSIONS)
        ) * weight
    raw_signal = sum(components.values()).clip(-3, 3)

    moving_average = close.rolling(
        MOVING_AVERAGE_WINDOW, min_periods=MOVING_AVERAGE_WINDOW
    ).mean()
    confirmed = ((raw_signal.gt(0) & close.gt(moving_average))
                 | (raw_signal.lt(0) & close.lt(moving_average)))
    signal = raw_signal.where(confirmed, 0.0)
    signal = signal.where(signal.abs().ge(SIGNAL_THRESHOLD), 0.0).where(eligibility)

    annualized_volatility = daily_volatility * np.sqrt(252)
    sizing_score = signal.div(annualized_volatility.where(annualized_volatility.gt(0)))
    return {
        "raw_signal": raw_signal.where(eligibility),
        "moving_average": moving_average.where(eligibility),
        "signal": signal,
        "annualized_volatility": annualized_volatility.where(eligibility),
        "sizing_score": sizing_score.where(eligibility),
    }


def run_one(
    score: pd.DataFrame,
    data: dict,
    *,
    costs: PortfolioCosts = COSTS,
    signal_delay: int = 0,
):
    loadings = normalized_loadings()
    sleeves = asset_sleeves()
    delayed = score.shift(signal_delay) if signal_delay else score
    return run_risk_budget_backtest(
        delayed,
        data["returns"],
        data["adv"],
        data["eligibility"] & delayed.notna(),
        loadings,
        FACTOR_CAPS,
        sleeves,
        SLEEVE_CAPS,
        config=CONFIG,
        costs=costs,
    )


def risk_budgets_pass(metrics: dict) -> bool:
    return bool(
        metrics.get("status") == "COMPLETED"
        and all(
            metrics[key] <= 1.0001
            for key in (
                "maximum_net_budget_ratio",
                "maximum_factor_budget_ratio",
                "maximum_sleeve_budget_ratio",
            )
        )
    )


def progression_passes(
    train: dict,
    development: dict,
    double_cost: dict,
    delayed: dict,
) -> bool:
    """Apply the prespecified development progression gate, not a live-capital gate."""
    ordinary = all(
        metrics.get("status") == "COMPLETED"
        and metrics.get("sharpe", -np.inf) > 0
        and metrics.get("cagr", -np.inf) > 0
        and metrics.get("max_drawdown", -np.inf) >= -0.20
        for metrics in (train, development)
    )
    robustness = all(
        metrics.get("status") == "COMPLETED" and metrics.get("sharpe", -np.inf) > 0
        for metrics in (double_cost, delayed)
    )
    return bool(
        ordinary
        and robustness
        and development.get("annual_turnover", np.inf) <= 25
        and risk_budgets_pass(development)
    )


def write_report(output: Path, summary: dict, rows: list[dict]) -> None:
    metrics = pd.DataFrame(rows).set_index("segment")

    def value(segment: str, field: str, percent: bool = False) -> str:
        number = float(metrics.loc[segment, field])
        return f"{number:.2%}" if percent else f"{number:.3f}"

    report = f"""# v19 Defensive Time-Series Momentum

## Decision

**{summary['status']}**

This experiment tests one prespecified 45-ETF defensive time-series momentum strategy. It does not
select among parameter variants. The 2021–2024 locked test was not evaluated, and paper orders
remain disabled.

| Segment | Net Sharpe | Net CAGR | Gross Sharpe | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|---:|
| Train 2008–2016 | {value('train', 'sharpe')} | {value('train', 'cagr', True)} | {value('train', 'gross_sharpe')} | {value('train', 'max_drawdown', True)} | {value('train', 'annual_turnover')} |
| Development 2017–2020 | {value('development', 'sharpe')} | {value('development', 'cagr', True)} | {value('development', 'gross_sharpe')} | {value('development', 'max_drawdown', True)} | {value('development', 'annual_turnover')} |
| Development, doubled costs | {value('double_cost', 'sharpe')} | {value('double_cost', 'cagr', True)} | {value('double_cost', 'gross_sharpe')} | {value('double_cost', 'max_drawdown', True)} | {value('double_cost', 'annual_turnover')} |
| Development, one-session delay | {value('signal_delay_1', 'sharpe')} | {value('signal_delay_1', 'cagr', True)} | {value('signal_delay_1', 'gross_sharpe')} | {value('signal_delay_1', 'max_drawdown', True)} | {value('signal_delay_1', 'annual_turnover')} |

## Interpretation

- Development progression gate: **{'PASS' if summary['progression_gate_passed'] else 'FAIL'}**.
- Original strict promotion gate: **{'PASS' if summary['strict_promotion_gate_passed'] else 'FAIL'}**.
- Locked test evaluated: **No**.
- Orders allowed: **No**.
- Independently verified numerical-boundary solutions: **{summary['numerical_boundary_solutions']}**.

The development period is reused evidence because earlier versions examined it. Even a progression
pass is only permission to define a separate locked-test protocol; it is not an unbiased OOS claim
or authorization to paper trade.
"""
    (output / "REPORT.md").write_text(report)


def main(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    train_data = load_data(source, TRAIN[1], UNIVERSE)
    train_features = build_defensive_tsmom(train_data)
    calibration = calibrate_absolute(train_features["sizing_score"], train_data["returns"])
    if calibration["status"] != "ADMITTED":
        summary = {
            "status": "BLOCKED",
            "reason": calibration["reason"],
            "progression_gate_passed": False,
            "strict_promotion_gate_passed": False,
            "locked_test_evaluated": False,
            "orders_allowed": False,
        }
        (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
        return

    freeze = {
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": sha256(source),
        "universe": UNIVERSE,
        "momentum_weights": MOMENTUM_WEIGHTS,
        "skip_sessions": SKIP_SESSIONS,
        "volatility_window": VOLATILITY_WINDOW,
        "moving_average_window": MOVING_AVERAGE_WINDOW,
        "signal_threshold": SIGNAL_THRESHOLD,
        "calibration": calibration,
        "factor_caps": FACTOR_CAPS.to_dict(),
        "sleeve_caps": SLEEVE_CAPS.to_dict(),
        "portfolio": asdict(CONFIG),
        "costs": asdict(COSTS),
        "development_reused": True,
        "locked_test_loaded_for_selection": False,
    }
    (output / "frozen_run_inputs.json").write_text(json.dumps(freeze, indent=2) + "\n")

    development_data = load_data(source, VALIDATION[1], UNIVERSE)
    development_features = build_defensive_tsmom(development_data)
    appended_calibration = calibrate_absolute(
        development_features["sizing_score"], development_data["returns"]
    )
    if appended_calibration["digest"] != calibration["digest"] or not np.isclose(
        appended_calibration["slope"], calibration["slope"]
    ):
        raise RuntimeError("appending development rows changed frozen train calibration")
    score = development_features["sizing_score"] * calibration["slope"]

    baseline = run_one(score, development_data)
    doubled = run_one(score, development_data, costs=replace(COSTS, multiplier=2.0))
    delayed = run_one(score, development_data, signal_delay=1)
    for name, result in {
        "baseline": baseline,
        "double_cost": doubled,
        "signal_delay_1": delayed,
    }.items():
        result.daily.to_csv(output / f"{name}_daily.csv")
        result.rebalances.to_csv(output / f"{name}_rebalances.csv")

    train_metrics = segment_metrics(baseline, *TRAIN)
    development_metrics = segment_metrics(baseline, *VALIDATION)
    double_cost_metrics = segment_metrics(doubled, *VALIDATION)
    delayed_metrics = segment_metrics(delayed, *VALIDATION)
    run_metrics = (
        ("train", train_metrics, baseline, TRAIN),
        ("development", development_metrics, baseline, VALIDATION),
        ("double_cost", double_cost_metrics, doubled, VALIDATION),
        ("signal_delay_1", delayed_metrics, delayed, VALIDATION),
    )
    rows = [
        {
            "segment": name,
            **metrics,
            "numerical_boundary_solutions": int(
                (~result.rebalances.loc[start:end, "optimizer_converged"]).sum()
            ),
        }
        for name, metrics, result, (start, end) in run_metrics
    ]
    pd.DataFrame(rows).to_csv(output / "evaluation.csv", index=False)

    progression = progression_passes(
        train_metrics, development_metrics, double_cost_metrics, delayed_metrics
    )
    strict = passes(train_metrics, development_metrics, budgeted=True)
    summary = {
        "status": "DEVELOPMENT_ACCEPTED_TEST_LOCKED" if progression else "REJECTED",
        "reason": None if progression else "V19_FAILED_FROZEN_DEVELOPMENT_PROGRESSION_GATE",
        "progression_gate_passed": progression,
        "strict_promotion_gate_passed": strict,
        "locked_test_evaluated": False,
        "orders_allowed": False,
        "numerical_boundary_solutions": int(
            sum(row["numerical_boundary_solutions"] for row in rows)
        ),
        "calibration": calibration,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(output, summary, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("output/cross_asset_etfs_v17/etf_daily.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v19"))
    arguments = parser.parse_args()
    main(arguments.source, arguments.output)
