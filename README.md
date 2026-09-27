# Cross-Asset Statistical Arbitrage & Trend Research Platform

Research platform for cross-asset trend, equity statistical arbitrage, cost-aware portfolio
construction, walk-forward validation and fail-closed paper execution.

> **Current decision: DEVELOPMENT ACCEPTED; LOCKED TEST NOT EVALUATED.** The prespecified v19
> defensive time-series momentum candidate passed its reused-development progression gate. It did
> not pass the original strict promotion gate, its contribution audit requires concentration and
> long-book review, and paper orders remain disabled.

## Latest evidence

The project began with U.S. equity residual alpha and evolved into a 45-ETF multi-asset platform
covering equities, rates, credit, metals, commodities and currencies. The main research finding is
that portfolio construction is no longer the binding constraint: price-only alpha loses stability
after 2017.

| Frozen experiment | Train net Sharpe | Development net Sharpe | Net CAGR | Max drawdown | Decision |
|---|---:|---:|---:|---:|---|
| v13 exact-neutral trend | -0.444 | -0.715 | -0.74% development | -4.28% development | Reject |
| v13 risk-budgeted trend | 0.726 | **0.638** | **5.16% development** | -16.43% development | Reject: drawdown |
| v15 time-series trend | 0.688 | 0.547 | 4.33% development | **-14.68% development** | Reject: return gates |
| v16 50/50 trend ensemble | **0.865** | 0.431 | 8.47% train / 3.35% development | -14.25% train / -15.16% development | Reject: instability |
| v18 expanded-universe winner | **0.823** | 0.212 | 7.25% train / 1.41% development | **-13.64% train** / -12.93% development | Reject: instability |
| v19 defensive TSMOM | **1.210** | **0.927** | 2.45% train / **3.60% development** | **-5.20% train** / **-9.53% development** | Development pass; test locked |

v13 isolated a construction error in v12: forcing equity, duration, credit, commodity and USD
exposure to exactly zero removed the macro trend the strategy was meant to earn. Replacing exact
neutrality with explicit risk budgets raised development net Sharpe from -0.715 to 0.638. Later
versions added time-series momentum, train-only ensemble selection, 45-ETF universe expansion and
risk-target selection. These improved training performance and drawdown control, but the first
development evaluation remained too weak for promotion.

v19 retained only absolute time-series momentum, added a fixed 200-day trend confirmation and
dead zone, used inverse-volatility signal sizing, reduced rebalancing to every ten sessions and
kept macro exposures inside risk budgets instead of neutralizing them. It remained profitable at
doubled costs and with a one-session signal delay. The 2017–2020 period is reused evidence, so the
result is permission to define a separate locked-test protocol rather than an unbiased OOS claim.
The contribution audit found that the top five ETFs generated 75.5% of absolute development
contribution and that almost all positive contribution came from the long book, led by credit and
equity exposure. This does not retroactively change the frozen progression gate, but it prevents a
diversified-alpha interpretation. The 2021–2024 test has not been evaluated.

- [v13–v18 consolidated report](reports/cross_asset_v13_v18/REPORT.md)
- [Development NAV and drawdown](reports/cross_asset_v13_v18/development_nav_drawdown.png)
- [Research ladder](reports/cross_asset_v13_v18/research_ladder.png)
- [v18 frozen selection result](reports/cross_asset_v18/development_audit.csv)
- [v19 defensive TSMOM report](reports/cross_asset_v19/REPORT.md)
- [v19 frozen protocol](docs/MULTI_ASSET_PROTOCOL_V19.md)

## Research capabilities

- Point-in-time-aware data gates and explicit data-quality labels.
- Cross-sectional residual, adaptive Kalman, price/volume and cross-asset trend alpha families.
- Exact equity beta/sector neutrality and risk-budgeted macro portfolio construction.
- Covariance risk, gross/net/name/factor/sleeve, turnover and ADV participation constraints.
- Commission, spread, slippage, square-root impact and borrow-cost accounting.
- Frozen train/development/test protocols, delay/cost/universe kill tests and locked-test controls.
- Alpaca paper execution that emits no orders unless every promotion gate passes.

## Equity research foundation

The original Alpaca equity panel contains SIP raw/all-adjusted daily bars for 503 current candidates
and SPY, a liquid top-400 universe and a 356-name continuous-history research cohort. Those results
remain labeled `research_snapshot_only`: Alpaca's current asset master cannot certify historical
membership or remove survivor bias. The code retains strict bitemporal PIT gates for vendor data
that can provide historical security-master records.

- [Strategy execution report](reports/equity_final/REPORT.md)
- [Data audit](reports/equity_v2/data_quality/DATA_STATUS.md)
- [Backtest results](reports/equity_v2/backtest/evaluation.json)
- [Kill tests](reports/equity_v2/robustness/kill_tests.csv)
- [Promotion decision](reports/equity_final/promotion_decision.json)
- [Cost-aware optimized target](reports/equity_final/optimized_portfolio/optimized_target.json)
- [Portfolio diagnostics](reports/equity_final/diagnostics/README.md)
- [v5 Conditional Market-Neutral report](reports/equity_v5/REPORT.md)
- [v7 Residual Short-Horizon Alpha Discovery report](reports/equity_v7/REPORT.md)
- [v8 research-objective sleeve allocation report](reports/equity_v8/REPORT.md)
- [v9 adaptive Kalman, orthogonal alpha and cost-confidence report](reports/equity_v9/REPORT.md)
- [v10 expanded-universe and tail-concentration report](reports/equity_v10/REPORT.md)
- [v11 Train-only OHLCV Alpha Discovery report](reports/equity_v11/REPORT.md)
- [v12 ETF cross-asset relative-value report](reports/cross_asset_v12/REPORT.md)
- [v13–v18 risk-budget, trend-blend and universe-expansion report](reports/cross_asset_v13_v18/REPORT.md)

## Repository structure

- `data/`: bitemporal universes, effective/available-time checks and historical-identity gates.
- `features/`: residual reversal, Kalman innovation, volume/volatility alpha and train calibration.
- `models/`: stable interfaces for calibration and covariance models.
- `portfolio/`: beta/sector/style neutralization and cost-aware portfolio optimization.
- `execution/`: commission, spread, slippage, impact and borrow costs.
- `risk/`: fail-closed promotion rules.
- `backtest/`: holding drift, expanding walk-forward evaluation and PnL attribution.
- `live/`: Alpaca-paper targets and order plans; rejected strategies emit no orders.
- `reports/`: data, backtest, robustness and rejection records.
- `tests/`: look-ahead, PIT, neutrality, cost, 400-stock optimization and paper-gate tests.

## Installation and execution

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[alpaca,data,dev]'
pytest -q
ruff check data features models portfolio execution risk backtest live scripts tests
```

Data collection:

```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."
python scripts/collect_week1_alpaca.py \
  --output output/week1-$(date +%Y%m%d) \
  --feed sip --start 2017-01-01 --end 2026-09-19
```

Reproduce the frozen equity research and shadow output:

```bash
python scripts/audit_equity_snapshot.py
python scripts/evaluate_equity_v2.py
python scripts/robustness_equity_v2.py
python scripts/build_optimized_target.py
PYTHONPATH=. python scripts/evaluate_equity_v9.py
PYTHONPATH=. python scripts/diagnose_equity_v9.py
PYTHONPATH=. python scripts/plot_v9_results.py
python -m live.generate_shadow
```

Reproduce the frozen cross-asset research:

```bash
python -m scripts.evaluate_cross_asset_v13
python -m scripts.evaluate_cross_asset_v14
python -m scripts.evaluate_cross_asset_v15
python -m scripts.evaluate_cross_asset_v16
python -m scripts.collect_cross_asset_etfs_v17 \
  --output output/cross_asset_etfs_v17 --start 2007-01-01 --end 2025-01-01
python -m scripts.evaluate_cross_asset_v17
python -m scripts.evaluate_cross_asset_v18
python -m scripts.evaluate_cross_asset_v19
```

A two-to-three-month prospective paper record must accumulate on future trading days; historical
backtests cannot substitute for it.
