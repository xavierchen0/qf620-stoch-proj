# Liberation Day Tariffs Project Overview

QF620 Stochastic Modelling in Finance examines how the 2-Apr-2025 “Liberation Day Tariffs” shock (“Trump Slump”) rippled through SPX options. This repository hosts the data preparation notebooks and analysis scripts underpinning that investigation.

---

## 1. Motivation
- **Event context**: Tariff headlines on 2-Apr triggered a rapid SPX selloff and panic selling fueled by retaliatory rhetoric. Understanding the derivatives-market response helps test stress-behavior hypotheses.
- **Core question**: How did SPX options, implied volatility, and the risk-neutral distribution behave before, during, and after the shock across both regular (RTH) and extended (ETH) trading hours?

---

## 2. Data Processing Roadmap
1. **Collection & Preprocessing**
   - Identify RTH (09:30–16:00 ET) vs ETH (overnight sessions) to align timestamps.
   - Gather SPX spot, VIX index, ES futures quotes, and SPX option sheets for calls/puts across expiries.
2. **Implied Vol Surfaces**
   - Use mid-prices to build surfaces for multiple maturities before/during/after the event.
3. **Volatility Time Series**
   - Track selected strikes (ATM, OTM calls/puts) to observe IV dynamics.

Outputs: clean asset dataframe (SPX, VIX, ES) plus call/put tables ready for IV or hedging analysis.

---

## 3. Analytical Framework
1. **Implied Volatility Smile** – Visualize shifts in smiles/surfaces around the event.
2. **Stochastic Volatility (e.g., SABR)** – Fit/compare to observed dynamics.
3. **Risk-Neutral PDF (Breeden-Litzenberger)** – Extract density pre/post event for tail behavior insight.

---

## 4. Hypotheses
1. **Volatility Spike** – IVs jumped materially on 2-Apr, signaling heightened uncertainty.
2. **Left-Skewed PDF** – Risk-neutral distribution developed a fatter left tail post shock.
3. **Delta-Hedging Stress** – Market makers running delta-neutral books faced larger hedging errors.
4. **Custom Ideas** – Extend with additional phenomena (liquidity, skew-of-skew, etc.).

---

## 5. Deliverables
- ≤10-page report including:
  1. Executive summary of findings.
  2. Data/methodology section (assumptions, preprocessing).
  3. Visuals for IV surfaces, volatility index, and/or risk-neutral PDFs over time.
  4. Comparative pre/post analysis of market dynamics.
  5. Conclusion addressing hypotheses.
- **Deadline**: Wednesday, 19-Nov-2025 at noon.

---

## 6. Repository Structure
- `proj.ju.py` – Jupytext notebook cleaning ETH/RTH datasets, producing consolidated assets and option frames, plus SPX/VIX visualizations.
- `main.py` and other notebooks – Additional modeling or exploratory scripts (see file headers for details).
- `ETH/`, `RTH/` – Raw CSV dumps split by session (assets, call/put bid/ask quotes).

---

## 7. Getting Started
1. Install dependencies (via `uv` / `pip` / `conda`): `pandas`, `matplotlib`, `numpy`, etc.
2. Run `proj.ju.py` to create the cleaned dataframes and plots.
3. Proceed with IV calibration, risk-neutral density estimation, and hypothesis testing using the cleaned outputs.

For questions or contributions, open an issue or submit a PR with reproducible steps.***
