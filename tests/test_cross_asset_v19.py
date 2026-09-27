import numpy as np
import pandas as pd

from backtest.cross_asset import CrossAssetResult
from scripts.cross_asset_v17_universe import UNIVERSE
from scripts.evaluate_cross_asset_v19 import (
    SIGNAL_THRESHOLD,
    build_defensive_tsmom,
    contribution_diagnostics,
    frozen_trade_cost_stress,
    progression_passes,
)


def synthetic_data(periods=340, assets=("UP", "DOWN", "REVERSAL", "WEAK")):
    index = pd.bdate_range("2010-01-04", periods=periods)
    slopes = {"UP": 0.002, "DOWN": -0.002, "REVERSAL": 0.002, "WEAK": 0.00005}
    close = pd.DataFrame(index=index, columns=assets, dtype=float)
    for name in assets:
        log_path = np.arange(periods) * slopes[name]
        if name == "REVERSAL":
            log_path[-40:] -= np.arange(40) * 0.008
        elif name == "WEAK":
            log_path = np.random.default_rng(6).normal(0, 0.01, periods).cumsum()
        close[name] = np.exp(log_path) * 100
    returns = close.pct_change(fill_method=None)
    eligibility = pd.DataFrame(True, index=index, columns=assets)
    return {"close": close, "returns": returns, "eligibility": eligibility}


def test_v19_confirmation_and_dead_zone_are_directional():
    features = build_defensive_tsmom(synthetic_data())
    last = features["signal"].iloc[-1]
    assert last["UP"] > SIGNAL_THRESHOLD
    assert last["DOWN"] < -SIGNAL_THRESHOLD
    assert last["REVERSAL"] == 0
    assert last["WEAK"] == 0


def test_v19_sizing_is_inverse_volatility_and_finite():
    data = synthetic_data(assets=("UP", "DOWN"))
    features = build_defensive_tsmom(data)
    usable = features["signal"].notna() & features["signal"].ne(0)
    reconstructed = features["sizing_score"] * features["annualized_volatility"]
    assert np.allclose(
        reconstructed.where(usable).stack().to_numpy(),
        features["signal"].where(usable).stack().to_numpy(),
    )
    assert np.isfinite(features["sizing_score"].where(usable).stack()).all()


def test_v19_features_do_not_change_when_future_prices_change():
    data = synthetic_data()
    cutoff = data["close"].index[-21]
    original = build_defensive_tsmom(data)["sizing_score"].loc[:cutoff]
    changed = {key: value.copy() for key, value in data.items()}
    changed["close"].loc[changed["close"].index > cutoff] *= 7
    changed["returns"] = changed["close"].pct_change(fill_method=None)
    revised = build_defensive_tsmom(changed)["sizing_score"].loc[:cutoff]
    pd.testing.assert_frame_equal(original, revised)


def test_progression_gate_requires_positive_robustness_and_risk_compliance():
    base = {
        "status": "COMPLETED",
        "sharpe": 0.4,
        "cagr": 0.03,
        "max_drawdown": -0.12,
        "annual_turnover": 4.0,
        "maximum_net_budget_ratio": 0.9,
        "maximum_factor_budget_ratio": 0.8,
        "maximum_sleeve_budget_ratio": 1.0,
    }
    assert progression_passes(base, base, base, base)
    failed_cost = {**base, "sharpe": -0.01}
    assert not progression_passes(base, base, failed_cost, base)
    breached = {**base, "maximum_factor_budget_ratio": 1.01}
    assert not progression_passes(base, breached, base, base)


def test_frozen_trade_cost_stress_preserves_portfolio_and_only_reprices_costs():
    index = pd.bdate_range("2019-01-02", periods=3)
    daily = pd.DataFrame({
        "gross_return": [0.01, -0.02, 0.03],
        "transaction_cost": [0.001, 0.002, 0.0],
        "borrow_cost": [0.0001, 0.0002, 0.0003],
        "net_return": [0.0089, -0.0222, 0.0297],
        "turnover": [0.2, 0.3, 0.0],
    }, index=index)
    weights = pd.DataFrame({"A": [0.5, 0.4, 0.3], "B": [-0.5, -0.4, -0.3]}, index=index)
    rebalances = pd.DataFrame({"turnover": [0.2]}, index=index[:1])
    baseline = CrossAssetResult(daily, weights, rebalances, "COMPLETED", "OK")
    stressed = frozen_trade_cost_stress(baseline)
    pd.testing.assert_frame_equal(stressed.weights, baseline.weights)
    pd.testing.assert_frame_equal(stressed.rebalances, baseline.rebalances)
    pd.testing.assert_series_equal(stressed.daily.turnover, baseline.daily.turnover)
    pd.testing.assert_series_equal(stressed.daily.gross_return, baseline.daily.gross_return)
    pd.testing.assert_series_equal(stressed.daily.borrow_cost, baseline.daily.borrow_cost)
    assert np.allclose(stressed.daily.transaction_cost, baseline.daily.transaction_cost * 2)
    assert np.allclose(
        stressed.daily.net_return,
        baseline.daily.gross_return - 2 * baseline.daily.transaction_cost
        - baseline.daily.borrow_cost,
    )


def test_contribution_diagnostics_reconcile_and_use_lagged_drawdown_regimes():
    index = pd.bdate_range("2008-01-02", periods=6)
    returns = pd.DataFrame({"SPY": [0, 0.01, -0.2, 0.01, 0.02, 0.01],
                            "QQQ": [0, 0.02, -0.1, 0.02, 0.01, 0.02]}, index=index)
    close = 100 * (1 + returns).cumprod()
    weights = pd.DataFrame({"SPY": [0.5] * 6, "QQQ": [-0.5] * 6}, index=index)
    gross = (weights * returns).sum(axis=1)
    daily = pd.DataFrame({
        "gross_return": gross,
        "transaction_cost": 0.0,
        "borrow_cost": 0.0,
        "net_return": gross,
        "turnover": 0.0,
    }, index=index)
    result = CrossAssetResult(daily, weights, pd.DataFrame(index=index[:1]), "COMPLETED", "OK")
    # The production audit requires the frozen 45-ETF metadata, so embed the two active series in it.
    expanded_returns = returns.reindex(columns=UNIVERSE, fill_value=0.0)
    expanded_close = close.reindex(columns=UNIVERSE).ffill().fillna(100.0)
    expanded_weights = weights.reindex(columns=UNIVERSE, fill_value=0.0)
    result = CrossAssetResult(
        daily,
        expanded_weights,
        pd.DataFrame(index=index[:1]),
        "COMPLETED",
        "OK",
    )
    diagnostics = contribution_diagnostics(
        result, {"returns": expanded_returns, "close": expanded_close}
    )
    asset_total = diagnostics["asset_attribution"].gross_return_contribution.sum()
    assert np.isclose(asset_total, daily.gross_return.sum())
    regimes = diagnostics["drawdown_regime_attribution"].regime.tolist()
    assert "stress_below_10pct" in regimes
