# Sector Rotation Strategy App

## Overview
This Streamlit app demonstrates a momentum-based sector rotation strategy using popular sector ETFs. The strategy dynamically allocates capital among the top-performing sectors, rebalancing monthly based on recent momentum. Users can configure the lookback period and the number of ETFs to hold, and allocations are weighted by the strength of each ETF's momentum.

## Design & Implementation
- **Frontend:** Built with Streamlit for interactive web-based visualization and parameter selection.
- **Data:** Uses yfinance to fetch historical price data for a set of sector ETFs. Data is cached locally for efficiency.
- **Strategy Logic:**
  - At each rebalance date (monthly), calculate the total return for each ETF over the user-defined lookback period (in months).
  - Select the top N ETFs by momentum (user-configurable).
  - Allocate portfolio weights to these ETFs in proportion to their positive momentum (normalized). If all momenta are non-positive, weights are distributed equally.
  - Portfolio value is updated daily based on the weighted returns of the current holdings.
  - Buy/sell transactions are detected and displayed based on changes in ETF weights.
- **Visualization:**
  - Portfolio value over time
  - Stack plot of ETF allocation percentages
  - Monthly returns of all ETFs
  - Table of all buy/sell transactions

## Theory
- **Sector Rotation:** The strategy is based on the idea that different sectors outperform at different stages of the economic cycle. By rotating into the strongest sectors (momentum), the strategy aims to capture superior returns and reduce drawdowns.
- **Momentum:** Empirical research shows that assets/sectors with strong recent performance tend to continue outperforming in the short to medium term. This app uses total return over a configurable lookback window as the momentum metric.
- **Diversification:** By holding the top N sectors, the strategy balances concentration (to capture outperformance) and diversification (to reduce risk).
- **Dynamic Weights:** Allocating based on the strength of momentum allows the strategy to overweight the strongest trends, rather than using equal weights.

## Usage
1. **Select Date Range:** Choose the backtest period.
2. **Configure Strategy:**
   - Set the lookback period (in months) for momentum calculation.
   - Set the number of ETFs to hold at each rebalance.
3. **Run & Analyze:**
   - View portfolio performance, allocation, and trades.
   - Download results for further analysis.

## Future Improvements
- **Custom ETF Selection:** Allow users to select their own ETFs or asset universe.
- **Alternative Momentum Metrics:** Support for risk-adjusted momentum, volatility filters, or dual momentum (absolute/relative).
- **Transaction Costs & Slippage:** Incorporate trading costs for more realistic backtesting.
- **Tax Impact Simulation:** Model after-tax returns based on user tax rates and account types.
- **Advanced Rebalancing:** Support for threshold or volatility-based rebalancing, not just fixed intervals.
- **Performance Metrics:** Add Sharpe ratio, max drawdown, rolling returns, and other analytics.
- **Email/Alert Integration:** Notify users when a rebalance is due or a trade is triggered.
- **Multi-Asset Support:** Extend to include bonds, commodities, or international ETFs.

## References
- Moskowitz, T.J., Ooi, Y.H., & Pedersen, L.H. (2012). "Time Series Momentum." Journal of Financial Economics.
- Antonacci, G. (2014). "Dual Momentum Investing."
- [Investopedia: Sector Rotation](https://www.investopedia.com/terms/s/sectorrotation.asp)

---
*Created with Streamlit, yfinance, and Plotly. For educational and research purposes only.*
