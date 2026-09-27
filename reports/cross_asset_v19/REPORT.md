# v19 Defensive Time-Series Momentum

## Decision

**DEVELOPMENT_ACCEPTED_TEST_LOCKED**

This experiment tests one prespecified 45-ETF defensive time-series momentum strategy. It does not
select among parameter variants. The 2021–2024 locked test was not evaluated, and paper orders
remain disabled.

| Segment | Net Sharpe | Net CAGR | Gross Sharpe | Max drawdown | Annual turnover |
|---|---:|---:|---:|---:|---:|
| Train 2008–2016 | 1.210 | 2.45% | 1.238 | -5.20% | 0.201 |
| Development 2017–2020 | 0.927 | 3.60% | 0.946 | -9.53% | 0.611 |
| Development, frozen trades and 2x transaction cost | 0.923 | 3.59% | 0.946 | -9.53% | 0.611 |
| Development, reoptimized at doubled costs | 0.931 | 2.58% | 0.943 | -7.66% | 0.183 |
| Development, one-session delay | 0.984 | 3.20% | 1.000 | -7.92% | 0.581 |

The frozen-trade stress leaves baseline holdings, turnover, gross return and borrow unchanged and
only doubles each realized transaction charge. The reoptimized stress is retained to show the
different portfolio the optimizer chooses when higher costs enter its objective.

## Contribution audit

- Development top-five absolute contribution share: **75.5%** (SPY,LQD,EMB,HYG,MUB).
- Largest development contributor: **SPY** (4.38%).
- Largest development detractor: **FXE** (-0.52%).
- Leading development sleeve: **credit** (7.74%).
- Development long-book contribution: **15.02%**; short-book contribution: **-0.27%**.

Detailed additive contribution tables are stored in `yearly_attribution.csv`,
`asset_attribution.csv`, `sleeve_attribution.csv`, `long_short_attribution.csv`,
`drawdown_regime_attribution.csv` and `contribution_concentration.csv`. Costs remain at portfolio
level and are reconciled in the yearly and regime tables.

## Interpretation

- Development progression gate: **PASS**.
- Original strict promotion gate: **FAIL**.
- Locked test evaluated: **No**.
- Orders allowed: **No**.
- Independently verified numerical-boundary solutions: **1**.
- Contribution diagnostic: **REVIEW_REQUIRED_CONCENTRATED_LONG_BOOK**.

The development period is reused evidence because earlier versions examined it. Even a progression
pass is only permission to define a separate locked-test protocol; it is not an unbiased OOS claim
or authorization to paper trade. The top-five concentration and weak short-book contribution must
be reviewed before a separate locked-test run is authorized.
