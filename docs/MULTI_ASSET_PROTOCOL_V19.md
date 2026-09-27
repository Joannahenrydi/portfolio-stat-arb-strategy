# v19 — Defensive Time-Series Momentum

Frozen on 2026-09-27 before v19 execution. The objective is to test one prespecified defensive
extension of the profitable v15 T01 time-series trend baseline. This is not a parameter search.
The 2017–2020 development period has been examined by earlier research and is therefore a reused
development audit. The 2021–2024 test remains locked and must not be loaded by this experiment.

## Fixed research windows

- Train: 2008-01-02 through 2016-12-30.
- Reused development audit: 2017-01-03 through 2020-12-31.
- Locked test: 2021-01-04 through 2024-12-31.

The source is the frozen 45-ETF v17 research snapshot. It is third-party adjusted history and is
not an exchange-grade point-in-time archive. Any historically accepted candidate still requires
Alpaca/SIP replication and a prospective paper record before orders are permitted.

## Prespecified strategy

For each eligible ETF, compute volatility-normalized absolute momentum over 3, 6 and 12 months,
skipping the most recent 21 sessions. Combine the three components using fixed weights of 25%,
35% and 40%. No cross-sectional ranking, demeaning, relative-value signal or machine learning is
allowed.

Apply two defensive rules using information available at the decision close:

1. A long signal is retained only when adjusted close is above its trailing 200-session moving
   average; a short signal is retained only when adjusted close is below that average.
2. Set the signal to zero when its absolute value is below 0.25.

Convert the retained signal to a sizing score by dividing by trailing 60-session annualized
volatility. Estimate one zero-intercept calibration slope on train only. The slope must be positive;
otherwise the experiment is blocked. Recalculation with appended development data must reproduce
the train calibration exactly.

## Frozen portfolio and costs

- Rebalance every 10 sessions.
- Annual portfolio volatility target/cap: 8%.
- Gross cap: 100%; name cap: 10%; net cap: 60%; turnover cap: 25% per rebalance.
- ADV participation cap: 0.10% at a USD 100,000 research NAV.
- Loose v17 macro factor and sleeve budgets; exact factor neutrality is prohibited.
- Costs: 1 bp half-spread, 1 bp slippage, square-root impact coefficient 0.10, no commission and
  a frozen 1% annual short-borrow proxy.

## Gates and decisions

The sole v19 candidate is not compared with alternative parameters. It passes the development
progression gate only if both train and development complete, have positive net CAGR and Sharpe,
maximum drawdown is at least -20%, development annual turnover is at most 25, every recorded risk
budget is satisfied, and development Sharpe remains positive under a frozen-trade doubled
transaction-cost stress and with a one-session signal delay. This gate means only
`DEVELOPMENT_ACCEPTED_TEST_LOCKED`; it is not live acceptance.

The frozen-trade stress must reuse the baseline weights, trades, gross returns, turnover and borrow
charges exactly. It replaces each realized transaction charge with twice that charge and recomputes
net return as `gross_return - 2 * transaction_cost - borrow_cost`. Reoptimizing after increasing the
cost multiplier answers a different question and is reported only as a secondary diagnostic.

The post-run concentration audit is descriptive and cannot change the progression decision. It
reports additive return contribution by calendar year, ETF, sleeve, long/short leg and lagged SPY
drawdown regime. Regimes use information available before each return: normal is drawdown above
-5%, correction is -5% through -10%, and stress is below -10%. Top-five concentration is the five
largest absolute ETF contributions divided by total absolute ETF contribution.

The original strict promotion gate is reported separately: train Sharpe above 0.70, development
Sharpe above 0.50, development CAGR above 5%, train and development drawdown at least -15%, and
development annual turnover at most 25. Failure is retained as a valid research result.

This run must write `locked_test_evaluated: false`, must not create a locked-test performance file,
and can never enable paper orders. A separate, explicitly authorized protocol and run are required
to unlock 2021–2024 after this development decision is frozen.

## Methodology amendment

On 2026-09-27, before any locked-test access, the doubled-cost definition was corrected because the
first implementation allowed the optimizer to choose different trades after observing the higher
cost assumption. The baseline strategy, signal parameters, portfolio constraints, research windows
and gates were not changed. Contribution diagnostics were added at the same time. The amendment is
recorded in Git history and all regenerated artifacts identify the corrected frozen-trade stress.
