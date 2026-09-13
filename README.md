# AI-Driven Algorithmic Trading — SMC + XGBoost System (Portfolio Edition)

A code sample from a larger, private algorithmic-trading project for the **Volatility 75 (1s) Index**: a Smart Money Concepts (SMC) market-structure trading system with a single machine-learning component, a purged/embargoed walk-forward XGBoost classifier, layered on top of it, plus the surrounding engineering: a realistic backtesting engine, a training harness, and a live FastAPI/WebSocket trading dashboard.

**This is a deliberately partial repository.** The proprietary core, the deterministic market-structure parsing engine, the multi-timeframe fusion logic, and the labeling scheme (the actual trading rules) has been withheld and is not included here. What's included is the surrounding production infrastructure: the backtester, the training pipeline, the live dashboard, and the validation tooling, so you can evaluate the engineering directly. **Full source access is available under contract/NDA** see [Full system access](#full-system-access) below.

## What the full system does (for context)

The complete pipeline (not all of it in this repo) is organized as eight stages: historical data ingestion → deterministic SMC structure parsing (swing highs/lows, strong/weak nodes, Breaks of Structure, Changes of Character, liquidity sweeps, Points of Interest) → causal multi-timeframe fusion with regime/pullback-quality features → quality-filtered triple-barrier labeling → XGBoost training under purged walk-forward cross-validation → realistic backtesting → live/demo execution via a FastAPI dashboard that re-invokes the identical offline parsing code on every closed candle, specifically to minimize the gap between backtested and live behavior.

## What it achieved

- A parity-preservation architecture, measured directly: replaying a fixed window of decisions through the backtester and through the live engine's decision log gave **93% trade-for-trade fidelity** between the two, with both residual discrepancies traced to fixable configuration bugs rather than logic errors.
- A four-part adversarial validation gauntlet entry-fill sensitivity, higher-timeframe merge-alignment verification, shuffled-label/cross-asset statistical controls, and live/backtest parity replay applied to the project's own favorable initial results, not only reported after the fact.
- Structural setups shown to be stable across two disjoint multi-year historical periods (85.2% vs. 84.8% win rate under an idealized fill assumption), independent of the open question below about whether that translates into a realistic tradeable edge.

## Known issues and limitations (reported honestly)

- An initial backtest looked attractive (profit factor 2.30), but collapses to **0.70** once entries are simulated with a realistic "wait for the bar to close" fill assumption. Under shuffled-label and cross-asset controls, the model's directional signal sits at or fractionally below the 0.50 chance level (**AUC ≈ 0.48–0.52**). The core trading edge is not established, this project's own conclusion is that its main contribution is the validation methodology and parity-preserving architecture, not a proven edge.
- Two unresolved parity discrepancies behind the 93% live/backtest fidelity figure, both traced to configuration/orchestration issues rather than structural-logic bugs.
- The live engine trades latency-efficiency for parity strength (full lookback re-parsed from scratch via subprocess on every closed candle); this hasn't been load-tested at larger scale.
- Specific defects found and fixed during development: a look-ahead artifact in trade-outcome resolution, a higher-timeframe merge that could leak up to 4 hours of future information into a lower-timeframe bar, a silently-defeated cumulative touch-counter filter, an SL-compression/TP-calculation ordering bug, a missing minimum-stop-distance check, and a Windows CSV-path quoting bug.

## What's in this repository

```
.
├── ai_backtester_be5.py        # Realistic order-fill backtesting engine
├── xgboost_trainer_dynamic.py  # Purged walk-forward training harness
├── preprocessing.py            # Parity-preserving feature-encoding pattern
├── dashboard_appz4.py          # Live/demo FastAPI + WebSocket trading dashboard
├── tools/                      # Validation & diagnostic scripts (see below)
├── requirements.txt
└── .gitignore
```

**Not included:** the market-structure parser, the multi-timeframe fusion/regime-feature builder, the labeling script, historical data, and trained model files. `dashboard_appz4.py` references these (it calls them as subprocesses and reads their output columns) so it will not run end-to-end as-is in this repository it's included to show the live-execution architecture: the FastAPI app shape, the asyncio trading loop, WebSocket telemetry, and the layered risk-management protocol (session/cooldown gates, position sizing, break-even/trailing management, a rolling circuit breaker).

The dashboard's login now reads `DASHBOARD_USERNAME` / `DASHBOARD_PASSWORD` from environment variables rather than a hardcoded credential.

## What each included file demonstrates

- **`ai_backtester_be5.py`** : realistic execution modeling: spread/slippage, limit-order activation scanning with an explicit fix for a same-bar look-ahead defect, position sizing with a minimum-stop-distance guard, break-even/trailing management, and a rolling circuit breaker. Also includes the project's own parity-verification tooling (`--anti-repaint`, merge-alignment checking) — the instrumentation used to *find* the defects listed above, not just avoid them.
- **`xgboost_trainer_dynamic.py`** : `sklearn.TimeSeriesSplit` with an explicit purge/embargo gap, per-fold precision-first threshold discovery, class-imbalance-aware sample weighting, and a metadata sidecar design that lets the exact same encoding be reproduced at inference time.
- **`preprocessing.py`** : the shared module imported unmodified by training, backtesting, and the live dashboard, with strict (no-fallback) categorical encoding that raises rather than silently mis-scoring an unseen value. This is the single mechanism most responsible for keeping training, backtesting, and live inference from silently diverging.
- **`dashboard_appz4.py`** : a FastAPI app with a long-running `asyncio` trading loop, a single-source-of-truth in-process state dict, WebSocket telemetry broadcast to multiple clients, and a REST control surface (start/stop, circuit-breaker reset, pending-order cancellation, a diagnostic pipeline replay endpoint, log export).
- **`tools/`** : the validation scripts used to hunt down real defects during development: look-ahead-bias checks (`check_lookahead.py`, `check_lookahead2.py`), an out-of-sample robustness harness (`oos_split.py`, `oos_experiment.py`, `oos_confirmed.py`), a labeler-vs-backtester outcome comparator (`diagnose_gap.py`), a feature-importance extraction tool (`get_top_features.py`), and a MetaTrader5 historical-data fetch utility (`get_vix75_history.py`).

## Requirements

- Python 3.10 or 3.11
- `pip install -r requirements.txt`
- `MetaTrader5` (Windows-only) is only needed to run the live dashboard or the data-fetch utility, the backtester and trainer are pure pandas/NumPy/XGBoost and run on any OS, though both expect input CSVs shaped by the (withheld) parsing/fusion/labeling stages.

## Full system access

The complete pipeline including the structural-parsing engine, multi-timeframe fusion, labeling logic, exact CLI reproduction commands, and the trained production model is maintained in a private repository. I'm glad to walk through it live, share read access under an NDA, or discuss it in the context of a specific contract. Reach out if you'd like to see it in full.
