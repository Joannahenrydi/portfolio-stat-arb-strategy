"""Evaluate the frozen v19 defensive absolute time-series momentum strategy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.cross_asset import CrossAssetResult
from backtest.cross_asset_budget import RiskBudgetConfig, run_risk_budget_backtest
from portfolio.optimizer import PortfolioCosts
from scripts.cross_asset_v17_universe import UNIVERSE, asset_sleeves, factor_loadings
from scripts.evaluate_cross_asset_v12 import TRAIN, VALIDATION, load_data
from scripts.evaluate_cross_asset_v13 import passes, segment_metrics
from scripts.evaluate_cross_asset_v15 import calibrate_absolute
from scripts.evaluate_equity_v7 import sha256

PROTOCOL = Path("docs/MULTI_ASSET_PROTOCOL_V19.md")
FREEZE_REF = "v19-development-freeze-20260927"
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


def frozen_trade_cost_stress(
    baseline: CrossAssetResult,
    *,
    transaction_multiplier: float = 2.0,
    borrow_multiplier: float = 1.0,
) -> CrossAssetResult:
    """Reprice fixed baseline trades without allowing the optimizer to alter holdings."""
    multipliers = np.asarray([transaction_multiplier, borrow_multiplier], dtype=float)
    if not np.isfinite(multipliers).all() or (multipliers < 0).any():
        raise ValueError("cost multipliers must be finite and nonnegative")
    daily = baseline.daily.copy()
    daily["transaction_cost"] *= transaction_multiplier
    daily["borrow_cost"] *= borrow_multiplier
    daily["net_return"] = (
        daily["gross_return"] - daily["transaction_cost"] - daily["borrow_cost"]
    )
    return CrossAssetResult(
        daily=daily,
        weights=baseline.weights.copy(),
        rebalances=baseline.rebalances.copy(),
        status=baseline.status,
        reason=baseline.reason,
    )


def contribution_diagnostics(result: CrossAssetResult, data: dict) -> dict[str, pd.DataFrame]:
    """Build additive, cost-reconciled train/development contribution tables."""
    returns = data["returns"]
    weights = result.weights.reindex(index=returns.index, columns=returns.columns)
    contributions = weights * returns.fillna(0)
    daily = result.daily.reindex(returns.index)
    reconciliation_error = contributions.sum(axis=1).sub(daily["gross_return"]).abs().max()
    if not np.isfinite(reconciliation_error) or reconciliation_error > 1e-10:
        raise RuntimeError("security contributions do not reconcile to portfolio gross return")

    prior_close = data["close"]["SPY"].shift(1)
    prior_peak = data["close"]["SPY"].cummax().shift(1)
    drawdown = prior_close.div(prior_peak).sub(1)
    regime = pd.Series(
        np.select(
            [drawdown.lt(-0.10), drawdown.le(-0.05)],
            ["stress_below_10pct", "correction_5_to_10pct"],
            default="normal_above_5pct",
        ),
        index=returns.index,
        name="regime",
    )
    sleeves = asset_sleeves()
    yearly_rows, asset_rows, sleeve_rows, leg_rows, regime_rows, concentration_rows = (
        [], [], [], [], [], []
    )
    segments = {"train": TRAIN, "development": VALIDATION}
    for segment, (start, end) in segments.items():
        segment_daily = daily.loc[start:end]
        segment_weights = weights.loc[start:end]
        segment_contributions = contributions.loc[start:end]
        long_contribution = segment_contributions.where(segment_weights.gt(0), 0).sum(axis=1)
        short_contribution = segment_contributions.where(segment_weights.lt(0), 0).sum(axis=1)

        for year, index in segment_daily.groupby(segment_daily.index.year).groups.items():
            subset = segment_daily.loc[index]
            yearly_rows.append({
                "segment": segment,
                "year": int(year),
                "sessions": len(subset),
                "gross_return_contribution": float(subset["gross_return"].sum()),
                "long_return_contribution": float(long_contribution.loc[index].sum()),
                "short_return_contribution": float(short_contribution.loc[index].sum()),
                "transaction_cost": float(subset["transaction_cost"].sum()),
                "borrow_cost": float(subset["borrow_cost"].sum()),
                "net_return_arithmetic": float(subset["net_return"].sum()),
                "net_return_compounded": float((1 + subset["net_return"]).prod() - 1),
            })

        asset_total = segment_contributions.sum().sort_values(ascending=False)
        for symbol in returns.columns:
            asset_rows.append({
                "segment": segment,
                "symbol": symbol,
                "sleeve": sleeves[symbol],
                "gross_return_contribution": float(asset_total[symbol]),
                "long_return_contribution": float(
                    segment_contributions[symbol].where(segment_weights[symbol].gt(0), 0).sum()
                ),
                "short_return_contribution": float(
                    segment_contributions[symbol].where(segment_weights[symbol].lt(0), 0).sum()
                ),
                "average_absolute_weight": float(segment_weights[symbol].abs().mean()),
                "long_sessions": int(segment_weights[symbol].gt(0).sum()),
                "short_sessions": int(segment_weights[symbol].lt(0).sum()),
            })
        by_sleeve = segment_contributions.T.groupby(sleeves).sum().T
        long_by_sleeve = segment_contributions.where(segment_weights.gt(0), 0).T.groupby(
            sleeves
        ).sum().T
        short_by_sleeve = segment_contributions.where(segment_weights.lt(0), 0).T.groupby(
            sleeves
        ).sum().T
        for sleeve in sorted(sleeves.unique()):
            sleeve_rows.append({
                "segment": segment,
                "sleeve": sleeve,
                "gross_return_contribution": float(by_sleeve[sleeve].sum()),
                "long_return_contribution": float(long_by_sleeve[sleeve].sum()),
                "short_return_contribution": float(short_by_sleeve[sleeve].sum()),
                "average_gross_weight": float(
                    segment_weights.loc[:, sleeves.eq(sleeve)].abs().sum(axis=1).mean()
                ),
            })
        leg_rows.extend([
            {"segment": segment, "leg": "long",
             "gross_return_contribution": float(long_contribution.sum())},
            {"segment": segment, "leg": "short",
             "gross_return_contribution": float(short_contribution.sum())},
        ])
        segment_regime = regime.loc[start:end]
        for regime_name in sorted(segment_regime.unique()):
            mask = segment_regime.eq(regime_name)
            subset = segment_daily.loc[mask]
            regime_rows.append({
                "segment": segment,
                "regime": regime_name,
                "sessions": int(mask.sum()),
                "gross_return_contribution": float(subset["gross_return"].sum()),
                "long_return_contribution": float(long_contribution.loc[mask].sum()),
                "short_return_contribution": float(short_contribution.loc[mask].sum()),
                "transaction_cost": float(subset["transaction_cost"].sum()),
                "borrow_cost": float(subset["borrow_cost"].sum()),
                "net_return_arithmetic": float(subset["net_return"].sum()),
                "net_return_compounded": float((1 + subset["net_return"]).prod() - 1),
            })
        absolute = asset_total.abs().sort_values(ascending=False)
        top_five = absolute.head(5)
        absolute_total = float(absolute.sum())
        concentration_rows.append({
            "segment": segment,
            "top_5_absolute_contribution_share": (
                float(top_five.sum() / absolute_total) if absolute_total > 0 else np.nan
            ),
            "top_5_symbols": ",".join(top_five.index),
            "largest_positive_contributor": str(asset_total.index[0]),
            "largest_positive_contribution": float(asset_total.iloc[0]),
            "largest_negative_contributor": str(asset_total.index[-1]),
            "largest_negative_contribution": float(asset_total.iloc[-1]),
        })

    return {
        "yearly_attribution": pd.DataFrame(yearly_rows),
        "asset_attribution": pd.DataFrame(asset_rows),
        "sleeve_attribution": pd.DataFrame(sleeve_rows),
        "long_short_attribution": pd.DataFrame(leg_rows),
        "drawdown_regime_attribution": pd.DataFrame(regime_rows),
        "contribution_concentration": pd.DataFrame(concentration_rows),
    }


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


def write_report(
    output: Path,
    summary: dict,
    rows: list[dict],
    diagnostics: dict[str, pd.DataFrame],
) -> None:
    metrics = pd.DataFrame(rows).set_index("segment")

    def value(segment: str, field: str, percent: bool = False) -> str:
        number = float(metrics.loc[segment, field])
        return f"{number:.2%}" if percent else f"{number:.3f}"

    concentration = diagnostics["contribution_concentration"].set_index("segment")
    development_assets = diagnostics["asset_attribution"].query("segment == 'development'")
    development_assets = development_assets.sort_values(
        "gross_return_contribution", ascending=False
    )
    development_sleeves = diagnostics["sleeve_attribution"].query(
        "segment == 'development'"
    ).sort_values("gross_return_contribution", ascending=False)

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
| Development, frozen trades and 2x transaction cost | {value('frozen_trade_double_cost', 'sharpe')} | {value('frozen_trade_double_cost', 'cagr', True)} | {value('frozen_trade_double_cost', 'gross_sharpe')} | {value('frozen_trade_double_cost', 'max_drawdown', True)} | {value('frozen_trade_double_cost', 'annual_turnover')} |
| Development, reoptimized at doubled costs | {value('reoptimized_double_cost', 'sharpe')} | {value('reoptimized_double_cost', 'cagr', True)} | {value('reoptimized_double_cost', 'gross_sharpe')} | {value('reoptimized_double_cost', 'max_drawdown', True)} | {value('reoptimized_double_cost', 'annual_turnover')} |
| Development, one-session delay | {value('signal_delay_1', 'sharpe')} | {value('signal_delay_1', 'cagr', True)} | {value('signal_delay_1', 'gross_sharpe')} | {value('signal_delay_1', 'max_drawdown', True)} | {value('signal_delay_1', 'annual_turnover')} |

The frozen-trade stress leaves baseline holdings, turnover, gross return and borrow unchanged and
only doubles each realized transaction charge. The reoptimized stress is retained to show the
different portfolio the optimizer chooses when higher costs enter its objective.

## Contribution audit

- Development top-five absolute contribution share: **{concentration.loc['development', 'top_5_absolute_contribution_share']:.1%}** ({concentration.loc['development', 'top_5_symbols']}).
- Largest development contributor: **{development_assets.iloc[0]['symbol']}** ({development_assets.iloc[0]['gross_return_contribution']:.2%}).
- Largest development detractor: **{development_assets.iloc[-1]['symbol']}** ({development_assets.iloc[-1]['gross_return_contribution']:.2%}).
- Leading development sleeve: **{development_sleeves.iloc[0]['sleeve']}** ({development_sleeves.iloc[0]['gross_return_contribution']:.2%}).
- Development long-book contribution: **{summary['diagnostic_review']['development_long_contribution']:.2%}**; short-book contribution: **{summary['diagnostic_review']['development_short_contribution']:.2%}**.

Detailed additive contribution tables are stored in `yearly_attribution.csv`,
`asset_attribution.csv`, `sleeve_attribution.csv`, `long_short_attribution.csv`,
`drawdown_regime_attribution.csv` and `contribution_concentration.csv`. Costs remain at portfolio
level and are reconciled in the yearly and regime tables.

## Interpretation

- Development progression gate: **{'PASS' if summary['progression_gate_passed'] else 'FAIL'}**.
- Original strict promotion gate: **{'PASS' if summary['strict_promotion_gate_passed'] else 'FAIL'}**.
- Locked test evaluated: **No**.
- Orders allowed: **No**.
- Independently verified numerical-boundary solutions: **{summary['numerical_boundary_solutions']}**.
- Contribution diagnostic: **{summary['diagnostic_review']['status']}**.

The development period is reused evidence because earlier versions examined it. Even a progression
pass is only permission to define a separate locked-test protocol; it is not an unbiased OOS claim
or authorization to paper trade. The top-five concentration and weak short-book contribution must
be reviewed before a separate locked-test run is authorized.
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
        "freeze_ref": FREEZE_REF,
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
    frozen_doubled = frozen_trade_cost_stress(baseline, transaction_multiplier=2.0)
    reoptimized_doubled = run_one(score, development_data, costs=replace(COSTS, multiplier=2.0))
    delayed = run_one(score, development_data, signal_delay=1)
    for name, result in {
        "baseline": baseline,
        "frozen_trade_double_cost": frozen_doubled,
        "reoptimized_double_cost": reoptimized_doubled,
        "signal_delay_1": delayed,
    }.items():
        result.daily.to_csv(output / f"{name}_daily.csv")
        result.rebalances.to_csv(output / f"{name}_rebalances.csv")

    train_metrics = segment_metrics(baseline, *TRAIN)
    development_metrics = segment_metrics(baseline, *VALIDATION)
    frozen_double_cost_metrics = segment_metrics(frozen_doubled, *VALIDATION)
    reoptimized_double_cost_metrics = segment_metrics(reoptimized_doubled, *VALIDATION)
    delayed_metrics = segment_metrics(delayed, *VALIDATION)
    run_metrics = (
        ("train", train_metrics, baseline, TRAIN),
        ("development", development_metrics, baseline, VALIDATION),
        ("frozen_trade_double_cost", frozen_double_cost_metrics, frozen_doubled, VALIDATION),
        ("reoptimized_double_cost", reoptimized_double_cost_metrics,
         reoptimized_doubled, VALIDATION),
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
    diagnostics = contribution_diagnostics(baseline, development_data)
    for name, table in diagnostics.items():
        table.to_csv(output / f"{name}.csv", index=False)

    progression = progression_passes(
        train_metrics, development_metrics, frozen_double_cost_metrics, delayed_metrics
    )
    strict = passes(train_metrics, development_metrics, budgeted=True)
    concentration = diagnostics["contribution_concentration"].set_index("segment")
    leg_attribution = diagnostics["long_short_attribution"].set_index(["segment", "leg"])
    development_top_five = float(
        concentration.loc["development", "top_5_absolute_contribution_share"]
    )
    development_long = float(
        leg_attribution.loc[("development", "long"), "gross_return_contribution"]
    )
    development_short = float(
        leg_attribution.loc[("development", "short"), "gross_return_contribution"]
    )
    diagnostic_review = {
        "status": "REVIEW_REQUIRED_CONCENTRATED_LONG_BOOK"
        if development_top_five > 0.50 or abs(development_short) < 0.20 * abs(development_long)
        else "PASS",
        "development_top_5_absolute_contribution_share": development_top_five,
        "development_top_5_symbols": str(
            concentration.loc["development", "top_5_symbols"]
        ).split(","),
        "development_long_contribution": development_long,
        "development_short_contribution": development_short,
    }
    summary = {
        "status": "DEVELOPMENT_ACCEPTED_TEST_LOCKED" if progression else "REJECTED",
        "reason": None if progression else "V19_FAILED_FROZEN_DEVELOPMENT_PROGRESSION_GATE",
        "progression_gate_passed": progression,
        "strict_promotion_gate_passed": strict,
        "locked_test_evaluated": False,
        "orders_allowed": False,
        "freeze_ref": FREEZE_REF,
        "diagnostic_review": diagnostic_review,
        "numerical_boundary_solutions": int(
            sum(row["numerical_boundary_solutions"] for row in rows)
        ),
        "calibration": calibration,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(output, summary, rows, diagnostics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("output/cross_asset_etfs_v17/etf_daily.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/cross_asset_v19"))
    arguments = parser.parse_args()
    main(arguments.source, arguments.output)
