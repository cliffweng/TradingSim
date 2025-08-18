# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

This is a Python-based trading simulation application built with Streamlit. To run the applications:

```bash
# Main trading simulator
streamlit run trading_sim.py

# Sector rotation strategy
streamlit run sector_rotation.py
```

## Project Architecture

### Core Components

1. **trading_sim.py** - Main Streamlit application for single-stock trading strategies
   - Interactive web interface for backtesting trading strategies
   - Supports multiple strategy types with configurable parameters
   - Includes trade blotter with P&L tracking and cash management
   - Visualizes price data, signals, and performance metrics

2. **strategies.py** - Strategy implementations using abstract base class pattern
   - `TradingStrategy` - Abstract base class defining the interface
   - `MACrossoverStrategy` - Moving average crossover strategy
   - `RSIStrategy` - RSI-based mean reversion strategy  
   - `BollingerBandsStrategy` - Bollinger bands breakout strategy
   - `PriceMomentumStrategy` - Price momentum strategy
   - All strategies return standardized DataFrames with 'signal' and 'positions' columns

3. **sector_rotation.py** - Sector rotation strategy using ETFs
   - Momentum-based sector rotation with configurable lookback periods
   - Multi-asset portfolio management with dynamic rebalancing
   - ETF allocation visualization and transaction tracking
   - Comparison against S&P 500 benchmark

### Data Management

- **yfinance** integration for fetching historical price data
- **Local caching** in `yf_cache/` directory using pickle files to avoid repeated API calls
- Cache files named by date range (e.g., `etfdata_2020-01-01_2025-07-01.pkl`)
- Fallback to simulated data if real data fetch fails

### Key Design Patterns

1. **Strategy Pattern** - All trading strategies inherit from `TradingStrategy` base class
2. **Data Caching** - Uses `@st.cache_data` decorator and pickle files for performance
3. **Modular Architecture** - Clear separation between strategy logic, data fetching, and UI
4. **Trade Management** - Comprehensive trade tracking with cash management and P&L calculation

### Trading Logic

- **Signal Generation** - Each strategy generates buy/sell signals based on technical indicators
- **Position Management** - Tracks position states (0=no position, 1=long, -1=short)
- **Cash Management** - Calculates shares based on available cash and position sizing
- **Performance Metrics** - Calculates returns, CAGR, and comparison to buy-and-hold

### Visualization Stack

- **Streamlit** - Web interface and user controls
- **Plotly** - Interactive charts for price data, signals, and portfolio performance
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical calculations for indicators

## Development Notes

- The application uses real-time data fetching but caches results for performance
- Strategy parameters are configurable through Streamlit sidebar widgets
- All price data is adjusted for splits/dividends using yfinance's auto_adjust feature
- Error handling includes fallback to simulated data if API calls fail
- Trade blotter includes detailed transaction history with P&L tracking