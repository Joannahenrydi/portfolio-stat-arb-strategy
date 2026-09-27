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
| Development, doubled costs | 0.931 | 2.58% | 0.943 | -7.66% | 0.183 |
| Development, one-session delay | 0.984 | 3.20% | 1.000 | -7.92% | 0.581 |

## Interpretation

- Development progression gate: **PASS**.
- Original strict promotion gate: **FAIL**.
- Locked test evaluated: **No**.
- Orders allowed: **No**.
- Independently verified numerical-boundary solutions: **1**.

The development period is reused evidence because earlier versions examined it. Even a progression
pass is only permission to define a separate locked-test protocol; it is not an unbiased OOS claim
or authorization to paper trade.
