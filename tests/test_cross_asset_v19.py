import numpy as np
import pandas as pd

from scripts.evaluate_cross_asset_v19 import (
    SIGNAL_THRESHOLD,
    build_defensive_tsmom,
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
