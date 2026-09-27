"""Risk-budgeted cross-asset portfolio backtest with soft macro exposure caps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from backtest.cross_asset import CrossAssetResult
from portfolio.optimizer import PortfolioCosts, estimate_costs


@dataclass(frozen=True)
class RiskBudgetConfig:
    rebalance_every: int = 5
    covariance_window: int = 126
    covariance_shrinkage: float = 0.50
    max_gross: float = 1.0
    max_name: float = 0.15
    max_turnover: float = 0.25
    max_participation: float = 0.001
    annual_volatility_cap: float = 0.10
    net_cap: float = 0.35
    nav: float = 100_000.0
    tolerance: float = 1e-7
    max_iterations: int = 500


def _is_feasible_candidate(result, bounds, linear, volatility_constraint, tolerance):
    """Independently validate a finite SLSQP candidate when convergence is not reported."""
    if result.x is None or not np.isfinite(result.x).all():
        return False
    lower_violation = float(np.max(bounds.lb - result.x))
    upper_violation = float(np.max(result.x - bounds.ub))
    linear_violation = float(np.max(linear.A @ result.x - linear.ub))
    volatility_violation = -float(volatility_constraint(result.x))
    return max(
        lower_violation,
        upper_violation,
        linear_violation,
        volatility_violation,
    ) <= tolerance


def run_risk_budget_backtest(
    expected_return: pd.DataFrame,
    returns: pd.DataFrame,
    adv: pd.DataFrame,
    eligibility: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    factor_caps: pd.Series,
    asset_sleeves: pd.Series,
    sleeve_caps: pd.Series,
    *,
    risk_scaler: pd.Series | None = None,
    config: RiskBudgetConfig | None = None,
    costs: PortfolioCosts | None = None,
) -> CrossAssetResult:
    config = config or RiskBudgetConfig()
    costs = costs or PortfolioCosts(
        commission_bps=0, half_spread_bps=1, slippage_bps=1,
        impact_coefficient=0.10, borrow_annual=0.01, calendar_days=7,
    )
    columns = returns.columns
    if not all(frame.index.equals(returns.index) and frame.columns.equals(columns)
               for frame in (expected_return, adv, eligibility)):
        raise ValueError("wide inputs must align")
    if not factor_loadings.index.equals(columns) or not asset_sleeves.index.equals(columns):
        raise ValueError("risk metadata must align")
    if risk_scaler is None:
        risk_scaler = pd.Series(1.0, index=returns.index)
    if not risk_scaler.index.equals(returns.index) or risk_scaler.isna().any():
        raise ValueError("risk scaler must be finite and align to sessions")
    if not np.isfinite(risk_scaler).all() or risk_scaler.le(0).any() or risk_scaler.gt(1).any():
        raise ValueError("risk scaler must be in (0, 1]")
    active = pd.Series(0.0, index=columns)
    pending_cost = pending_turnover = 0.0
    records, weights, rebalances = [], [], []
    last_session = None
    for offset, session in enumerate(returns.index):
        realized = returns.loc[session]
        if (active.ne(0) & realized.isna()).any():
            return CrossAssetResult(pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                                    pd.DataFrame(rebalances), "INVALID", "MISSING_HELD_RETURN")
        gross_return = float((active * realized.fillna(0)).sum())
        days = 1 if last_session is None else max(1, (session - last_session).days)
        borrow = float(active.clip(upper=0).abs().sum()) * costs.borrow_annual * days / 365 * costs.multiplier
        net_return = gross_return - pending_cost - borrow
        records.append({"session": session, "gross_return": gross_return,
                        "transaction_cost": pending_cost, "borrow_cost": borrow,
                        "net_return": net_return, "gross": float(active.abs().sum()),
                        "net": float(active.sum()), "turnover": pending_turnover})
        weights.append(active.rename(session))
        denominator = 1 + net_return
        if denominator <= 0:
            return CrossAssetResult(pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                                    pd.DataFrame(rebalances), "INVALID", "NONPOSITIVE_NAV")
        active = active.mul(1 + realized.fillna(0)).div(denominator)
        pending_cost = pending_turnover = 0.0
        if offset % config.rebalance_every == 0 and offset >= config.covariance_window:
            scale = float(risk_scaler.loc[session])
            usable = eligibility.loc[session] & expected_return.loc[session].notna() & adv.loc[session].gt(0)
            names = columns[usable]
            if len(names) < 8:
                last_session = session
                continue
            history = returns.loc[:session, names].tail(config.covariance_window)
            covariance = history.cov().to_numpy(dtype=float)
            covariance = ((1 - config.covariance_shrinkage) * covariance
                          + config.covariance_shrinkage * np.diag(np.diag(covariance)))
            alpha = expected_return.loc[session, names].to_numpy(dtype=float)
            previous = active.loc[names].to_numpy(dtype=float)
            fixed = active.drop(names)
            fixed_gross = float(fixed.abs().sum())
            fixed_net = float(fixed.sum())
            factors = factor_loadings.loc[names]
            fixed_factor_exposure = factor_loadings.drop(index=names).T @ fixed
            sleeves = asset_sleeves.loc[names]
            fixed_sleeve_gross = fixed.abs().groupby(asset_sleeves.drop(index=names)).sum()
            liquidity = adv.loc[session, names].to_numpy(dtype=float)
            capacity = liquidity * config.max_participation / config.nav
            base_cost = (costs.commission_bps + costs.half_spread_bps + costs.slippage_bps) / 10_000
            weekly_borrow = costs.borrow_annual * 7 / 365 * costs.multiplier
            n = len(names)
            # x = [weights, absolute holdings, absolute trades]. Auxiliary variables make
            # every cost/exposure constraint linear; only the volatility cap is quadratic.
            objective_vector = np.r_[
                -alpha - weekly_borrow / 2,
                np.full(n, weekly_borrow / 2),
                np.full(n, base_cost * costs.multiplier),
            ]
            rows, upper = [], []

            def add(row, limit, output_rows=rows, output_upper=upper):
                output_rows.append(row)
                output_upper.append(limit)

            eye, zeros = np.eye(n), np.zeros((n, n))
            for row, limit in zip(np.hstack([eye, -eye, zeros]), np.zeros(n)):
                add(row, limit)  # w <= |w|
            for row, limit in zip(np.hstack([-eye, -eye, zeros]), np.zeros(n)):
                add(row, limit)  # -w <= |w|
            for row, limit in zip(np.hstack([eye, zeros, -eye]), previous):
                add(row, limit)  # w - previous <= |trade|
            for row, limit in zip(np.hstack([-eye, zeros, -eye]), -previous):
                add(row, limit)  # previous - w <= |trade|
            scaled_gross = config.max_gross * scale
            available_gross = scaled_gross - fixed_gross
            scaled_name = config.max_name * scale
            required_reduction = [max(0.0, np.abs(previous).sum() - max(available_gross, 0))]
            required_reduction.append(
                max(0.0, abs(previous.sum() + fixed_net) - config.net_cap * scale)
            )
            required_reduction.append(float(np.maximum(np.abs(previous) - scaled_name, 0).sum()))
            for factor in factors.columns:
                vector = factors[factor].to_numpy(dtype=float)
                total_exposure = vector @ previous + float(fixed_factor_exposure[factor])
                excess = max(0.0, abs(total_exposure) - float(factor_caps[factor]) * scale)
                required_reduction.append(excess / max(float(np.abs(vector).max()), 1e-12))
            for sleeve, cap in sleeve_caps.items():
                mask = sleeves.eq(sleeve).to_numpy()
                required_reduction.append(
                    max(0.0, np.abs(previous[mask]).sum()
                        + float(fixed_sleeve_gross.get(sleeve, 0)) - float(cap) * scale)
                )
            # A risk-cap cut overrides the ordinary turnover cap only by the minimum
            # amount needed to restore feasibility; the resulting trades are still costed.
            turnover_limit = max(config.max_turnover, max(required_reduction) + config.tolerance)
            add(np.r_[np.zeros(n), np.ones(n), np.zeros(n)], available_gross)
            add(np.r_[np.zeros(2 * n), np.ones(n)], turnover_limit)
            add(np.r_[np.ones(n), np.zeros(2 * n)], config.net_cap * scale - fixed_net)
            add(np.r_[-np.ones(n), np.zeros(2 * n)], config.net_cap * scale + fixed_net)
            for factor in factors.columns:
                vector = factors[factor].to_numpy(dtype=float)
                cap = float(factor_caps[factor]) * scale
                fixed_exposure = float(fixed_factor_exposure[factor])
                add(np.r_[vector, np.zeros(2 * n)], cap - fixed_exposure)
                add(np.r_[-vector, np.zeros(2 * n)], cap + fixed_exposure)
            for sleeve, cap in sleeve_caps.items():
                mask = sleeves.eq(sleeve).to_numpy(dtype=float)
                remaining = float(cap) * scale - float(fixed_sleeve_gross.get(sleeve, 0))
                add(np.r_[np.zeros(n), mask, np.zeros(n)], remaining)
            linear_matrix = np.vstack(rows)
            linear_upper = np.asarray(upper)
            linear = LinearConstraint(linear_matrix, -np.inf, linear_upper)

            def objective(x, vector=objective_vector):
                return float(10_000 * vector @ x)

            def objective_jac(_x, vector=objective_vector):
                return 10_000 * vector

            variance_cap = config.annual_volatility_cap ** 2 / 252

            def volatility_constraint(x, dimension=n, cov=covariance, cap=variance_cap):
                weight = x[:dimension]
                return cap - float(weight @ cov @ weight)

            def volatility_jac(x, dimension=n, cov=covariance):
                return np.r_[-2 * cov @ x[:dimension], np.zeros(2 * dimension)]

            initial_weight = np.clip(
                previous, -config.max_name * scale, config.max_name * scale
            )
            initial_trade = np.abs(initial_weight - previous)
            initial = np.r_[initial_weight, np.abs(initial_weight), initial_trade]
            bounds = Bounds(
                np.r_[np.full(n, -config.max_name * scale), np.zeros(2 * n)],
                np.r_[np.full(n, config.max_name * scale),
                      np.full(n, config.max_name * scale), capacity],
            )
            result = minimize(
                objective, initial, jac=objective_jac, method="SLSQP", bounds=bounds,
                constraints=[linear, {"type": "ineq", "fun": volatility_constraint,
                                      "jac": volatility_jac}],
                options={"maxiter": config.max_iterations, "ftol": 1e-9, "disp": False},
            )
            independently_feasible = _is_feasible_candidate(
                result, bounds, linear, volatility_constraint, config.tolerance
            )
            if not result.success and not independently_feasible:
                return CrossAssetResult(pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                                        pd.DataFrame(rebalances), "INVALID",
                                        f"RISK_BUDGET_OPTIMIZER_FAILED_{result.status}_{result.message}")
            target = active.copy()
            target.loc[names] = result.x[:n]
            trade = target - active
            daily_vol = pd.Series(0.0, index=columns)
            daily_vol.loc[names] = np.sqrt(np.maximum(np.diag(covariance), 0))
            projected = estimate_costs(active.to_numpy(), target.to_numpy(), daily_vol.to_numpy(),
                                       adv.loc[session].fillna(0).to_numpy(), config.nav, costs)
            pending_cost = projected["transaction"]
            pending_turnover = float(trade.abs().sum())
            active = target
            factor_exposure = factor_loadings.T @ target
            sleeve_gross = target.abs().groupby(asset_sleeves).sum()
            rebalances.append({"session": session, "eligible_assets": len(names),
                               "gross": float(target.abs().sum()), "net": float(target.sum()),
                               "turnover": pending_turnover,
                               "turnover_limit": turnover_limit,
                               "forecast_volatility": float(
                                   np.sqrt(result.x[:n] @ covariance @ result.x[:n] * 252)
                               ),
                               "optimizer_converged": bool(result.success),
                               "optimizer_status": int(result.status),
                               "risk_scaler": scale,
                               "net_budget_ratio": float(abs(target.sum()) / (config.net_cap * scale)),
                               "maximum_factor_budget_ratio": float(
                                   (factor_exposure.abs() / (factor_caps * scale)).max()
                               ),
                               "maximum_sleeve_budget_ratio": float(
                                   (sleeve_gross / (sleeve_caps * scale)).max()
                               ),
                               **{f"exposure_{key}": float(value) for key, value in factor_exposure.items()}})
        last_session = session
    return CrossAssetResult(pd.DataFrame(records).set_index("session"), pd.DataFrame(weights),
                            pd.DataFrame(rebalances).set_index("session"), "COMPLETED", "OK")
