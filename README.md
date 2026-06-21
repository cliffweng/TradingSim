# TradingSim

**TradingSim** is a Python-based trading simulation suite built with Streamlit. It provides two interactive web apps for backtesting trading strategies and analyzing sector rotation — no coding required.

Built by **Cliff Weng**.

---

## Apps

### 1. Trading Simulation App (`trading_sim.py`)

A single-ticker backtesting engine that fetches real historical data via Yahoo Finance and lets you test 10 different trading strategies with full trade logging, tax-aware P&L, and visual charting.

**Strategies available:**
- Moving Average Crossover — buy when short MA crosses above long MA
- RSI (Relative Strength Index) — buy oversold, sell overbought
- Bollinger Bands Breakout — buy when price touches lower band, sell at upper band
- Price Momentum — follow positive/negative price momentum
- MACD — trade MACD / signal line crossovers
- Mean Reversion — buy when z-score is far below the mean
- Donchian Channel — buy on breakout above the upper channel
- Stochastic Oscillator — trade %K/%D crossovers in overbought/oversold zones
- ATR Trailing Stop — trail price using Average True Range
- Dual Thrust — trade breakouts from range-based levels

**Features:**
- Real-time data fetching from Yahoo Finance (cached via `@lru_cache`)
- Tax-aware mode with configurable capital gains tax rate
- Side-by-side comparison vs Buy & Hold
- Annual return breakdown (pre-tax and after-tax)
- Trade blotter with P&L tracking, downloadable as CSV
- Interactive Plotly charts with buy/sell signal markers
- Parameter sliders for every strategy

### 2. [Sector Rotation Strategy App]((README_ROTATION.md)) (`sector_rotation.py`)

A multi-ETF sector rotation backtester that allocates across 7 sector ETFs and rebalances periodically based on 5 different rotation signals.

**Sector ETFs tracked:**
- XLK (Technology), XLV (Healthcare), XLF (Financials), XLY (Consumer Discretionary), XLP (Consumer Staples), XLE (Energy), XLU (Utilities)

**Strategies available:**
- Momentum (Top N) — pick the best-performing sectors
- RSI (Oversold Sectors) — buy sectors that are oversold
- Mean Reversion — buy sectors that have underperformed
- Relative Strength (vs S&P 500) — pick sectors outperforming the broader market
- Risk-Adjusted Momentum (Sharpe) — pick sectors with the best risk-adjusted returns

**Features:**
- Real data from Yahoo Finance, with fallback to simulated random-walk data
- Rebalancing at monthly, quarterly, or yearly frequency
- Benchmark comparison vs S&P 500
- ETF allocation stack plot showing portfolio composition over time
- Tax-aware mode with short/long-term capital gains tracking
- Tax efficiency tips for real-world execution
- Trade log with P&L per transaction
- Results download as CSV

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/yourusername/TradingSim.git
cd TradingSim
pip install -r requirements.txt
```

### Run the Apps

```bash
# Trading Simulation App
streamlit run trading_sim.py

# Sector Rotation Strategy App
streamlit run sector_rotation.py
```

---

## Project Structure

```
TradingSim/
├── trading_sim.py              # Single-ticker backtesting app
├── sector_rotation.py          # Multi-ETF sector rotation app
├── strategies.py               # Trading strategy implementations (10 strategies)
├── financial_analyst_team.py   # Financial analyst team module
├── requirements.txt            # Python dependencies
├── AGENTS.md                   # Agent coding guidelines
├── README.md                   # This file
├── yf_cache/                   # Cached Yahoo Finance data
└── tests/
    └── test_analysts.py        # Tests for analyst module
```

## License

MIT License
