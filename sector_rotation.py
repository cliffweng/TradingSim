import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import os
import pickle

# Streamlit app configuration
st.set_page_config(page_title="Sector Rotation Strategy", layout="wide")

# Title and description
st.title("Sector Rotation Strategy with ETFs")
st.write("""
This app demonstrates various sector rotation strategies using popular sector ETFs.
Each strategy rebalances periodically to rotate between sectors based on different signals.
Note: For tax efficiency, execute trades in tax-advantaged accounts (e.g., IRA) or hold assets
for over a year to qualify for long-term capital gains rates. When selling at a loss, swap to
ETFs tracking different indices to avoid wash-sale rule issues.
""")

# Define sector ETFs
etfs = {
    "XLK": "Technology",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities"
}

# Strategy selection
st.sidebar.header("Strategy Selection")
strategy_type = st.sidebar.selectbox(
    "Sector Rotation Strategy",
    [
        "Momentum (Top N)",
        "RSI (Oversold Sectors)",
        "Mean Reversion",
        "Relative Strength (vs S&P 500)",
        "Risk-Adjusted Momentum (Sharpe)"
    ],
    help="Choose the sector rotation signal. Each strategy selects which ETFs to hold at each rebalance date based on different criteria."
)

# Dynamic parameters based on strategy
st.sidebar.header("Strategy Parameters")

if strategy_type == "Momentum (Top N)":
    lookback_months = st.sidebar.slider("Lookback Period (months)", min_value=1, max_value=12, value=6,
        help="Number of months of historical returns used to rank sector ETF performance")
    top_n = st.sidebar.slider("Number of ETFs to Hold", min_value=1, max_value=len(etfs), value=3,
        help="Number of top-performing ETFs to hold in the portfolio at each rebalance")
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["ME", "QE", "YE"], index=0,
        help="How often the portfolio is rebalanced: Monthly (ME), Quarterly (QE), or Yearly (YE)")
elif strategy_type == "RSI (Oversold Sectors)":
    rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14,
        help="Lookback period for RSI calculation. Lower values make the indicator more sensitive.")
    oversold_threshold = st.sidebar.slider("Oversold Threshold", min_value=10, max_value=40, value=30,
        help="RSI level below which a sector is considered oversold and a candidate for buying")
    top_n = st.sidebar.slider("Max ETFs to Hold", min_value=1, max_value=len(etfs), value=3,
        help="Maximum number of oversold ETFs to hold simultaneously")
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["ME", "QE", "YE"], index=0,
        help="How often the portfolio is rebalanced: Monthly (ME), Quarterly (QE), or Yearly (YE)")
elif strategy_type == "Mean Reversion":
    lookback_months = st.sidebar.slider("Lookback Period (months)", min_value=1, max_value=12, value=3,
        help="Number of months of returns used to calculate z-scores and identify underperforming sectors")
    z_threshold = st.sidebar.slider("Z-Score Threshold", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        help="Z-score threshold for identifying oversold sectors. Sectors below -threshold are bought.")
    top_n = st.sidebar.slider("Max ETFs to Hold", min_value=1, max_value=len(etfs), value=3,
        help="Maximum number of oversold ETFs to hold simultaneously")
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["ME", "QE", "YE"], index=0,
        help="How often the portfolio is rebalanced: Monthly (ME), Quarterly (QE), or Yearly (YE)")
elif strategy_type == "Relative Strength (vs S&P 500)":
    lookback_months = st.sidebar.slider("Lookback Period (months)", min_value=1, max_value=12, value=6,
        help="Number of months used to calculate each ETF's excess return over the S&P 500")
    top_n = st.sidebar.slider("Number of ETFs to Hold", min_value=1, max_value=len(etfs), value=3,
        help="Number of ETFs with the highest relative strength to hold")
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["ME", "QE", "YE"], index=0,
        help="How often the portfolio is rebalanced: Monthly (ME), Quarterly (QE), or Yearly (YE)")
else:  # Risk-Adjusted Momentum (Sharpe)
    lookback_months = st.sidebar.slider("Lookback Period (months)", min_value=1, max_value=12, value=6,
        help="Number of months used to calculate Sharpe ratios for each sector ETF")
    top_n = st.sidebar.slider("Number of ETFs to Hold", min_value=1, max_value=len(etfs), value=3,
        help="Number of ETFs with the highest Sharpe ratios to hold")
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["ME", "QE", "YE"], index=0,
        help="How often the portfolio is rebalanced: Monthly (ME), Quarterly (QE), or Yearly (YE)")

# Tax considerations
st.sidebar.header("Tax Settings")
enable_tax = st.sidebar.checkbox("Enable Tax-Aware Mode", value=True,
    help="When enabled, capital gains tax is calculated on profitable trades. Tax is tracked per transaction and reflected in after-tax returns.")
if enable_tax:
    tax_rate = st.sidebar.slider("Capital Gains Tax Rate (%)", min_value=0, max_value=50, value=37,
        help="Tax rate applied to profitable trades. Short-term gains are taxed at ordinary income rates (up to 37% in the US).") / 100
    st.sidebar.write("Note: Short-term gains use income tax rate; long-term gains use preferential rate.")

st.sidebar.divider()
st.sidebar.caption("Built by wengc — [GitHub](https://github.com/wengc)")

# User input for date range
st.subheader("Select Date Range")
st.caption("Choose the historical period for backtesting the sector rotation strategy across all ETFs.")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2020, 1, 1),
        help="Beginning of the backtest period")
with col2:
    end_date = st.date_input("End Date", value=datetime(2025, 7, 1),
        help="End of the backtest period")

# Function to fetch or simulate data
@st.cache_data
def get_etf_data(tickers, start, end):
    cache_dir = 'yf_cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"etfdata_{start}_{end}.pkl")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
        else:
            raw_data = yf.download(list(tickers.keys()), start=start, end=end, progress=False, auto_adjust=True)
            if raw_data.empty:
                raise ValueError("No data returned from yfinance.")
            # Handle MultiIndex columns (multiple tickers)
            if isinstance(raw_data.columns, pd.MultiIndex):
                col0 = raw_data.columns.get_level_values(0)
                if 'Adj Close' in col0:
                    data = raw_data['Adj Close']
                elif 'Close' in col0:
                    data = raw_data['Close']
                else:
                    raise ValueError("Neither 'Adj Close' nor 'Close' found in yfinance MultiIndex data.")
            # Handle flat columns (single ticker)
            elif 'Adj Close' in raw_data.columns:
                data = raw_data[['Adj Close']].copy()
                data.columns = [list(tickers.keys())[0]]
            elif 'Close' in raw_data.columns:
                data = raw_data[['Close']].copy()
                data.columns = [list(tickers.keys())[0]]
            elif isinstance(raw_data, pd.DataFrame) and len(raw_data.columns) == 1:
                data = raw_data.copy()
                data.columns = [list(tickers.keys())[0]]
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' found in yfinance data and cannot infer price column.")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        return data
    except Exception as e:
        st.warning(f"Real data fetch failed. Using simulated data. Error: {e}")
        dates = pd.date_range(start=start, end=end, freq='D')
        data = pd.DataFrame(index=dates)
        np.random.seed(42)
        for ticker in tickers:
            # Simulate prices with random walk
            prices = np.cumprod(1 + np.random.normal(0.0002, 0.01, len(dates)))
            data[ticker] = prices * 100  # Scale to realistic price levels
        return data

# Fetch or simulate data
_etf_cache_file = os.path.join('yf_cache', f"etfdata_{start_date}_{end_date}.pkl")
if os.path.exists(_etf_cache_file):
    st.sidebar.success(f"ETF data: loaded from cache")
    st.toast(f"ETF data loaded from local cache", icon="💾")
    data = get_etf_data(etfs, start_date, end_date)
else:
    st.sidebar.info("ETF data: fetching from yfinance...")
    with st.spinner(f"Fetching ETF pricing data from yfinance ({', '.join(etfs.keys())})..."):
        data = get_etf_data(etfs, start_date, end_date)
    st.sidebar.success("ETF data: fetched from yfinance")
    st.toast("ETF pricing data retrieved from yfinance", icon="✅")

# Fetch S&P 500 data for comparison
@st.cache_data
def get_sp500_data(start, end):
    cache_dir = 'yf_cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"sp500_{start}_{end}.pkl")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                sp500 = pickle.load(f)
        else:
            sp500_raw = yf.download('^GSPC', start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(sp500_raw.columns, pd.MultiIndex):
                if 'Adj Close' in sp500_raw.columns.get_level_values(0):
                    sp500 = sp500_raw['Adj Close']
                elif 'Close' in sp500_raw.columns.get_level_values(0):
                    sp500 = sp500_raw['Close']
                else:
                    raise ValueError('No Close/Adj Close in S&P 500 data')
            elif 'Adj Close' in sp500_raw.columns:
                sp500 = sp500_raw['Adj Close']
            elif 'Close' in sp500_raw.columns:
                sp500 = sp500_raw['Close']
            else:
                raise ValueError('No Close/Adj Close in S&P 500 data')
            with open(cache_file, 'wb') as f:
                pickle.dump(sp500, f)
        if isinstance(sp500, pd.DataFrame):
            sp500 = sp500.squeeze()
        return sp500
    except Exception as e:
        st.warning(f"S&P 500 data fetch failed: {e}")
        return pd.Series(index=pd.date_range(start=start, end=end, freq='D'), dtype=float)

_sp500_cache_file = os.path.join('yf_cache', f"sp500_{start_date}_{end_date}.pkl")
if os.path.exists(_sp500_cache_file):
    st.sidebar.success("S&P 500 data: loaded from cache")
    st.toast("S&P 500 benchmark data loaded from local cache", icon="💾")
    sp500 = get_sp500_data(start_date, end_date)
else:
    st.sidebar.info("S&P 500 data: fetching from yfinance...")
    with st.spinner("Fetching S&P 500 benchmark data from yfinance (^GSPC)..."):
        sp500 = get_sp500_data(start_date, end_date)
    st.sidebar.success("S&P 500 data: fetched from yfinance")
    st.toast("S&P 500 benchmark data retrieved from yfinance", icon="✅")

# Calculate monthly returns for visualization
monthly_returns = data.resample('ME').last().pct_change().dropna()


def momentum_strategy(data, lookback_months, rebalance_freq, top_n):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio_Value'] + list(data.columns))
    portfolio = portfolio.astype({'Portfolio_Value': float, **{col: float for col in data.columns}})
    portfolio.iloc[0, 0] = 10000.0
    returns = data.pct_change().fillna(0)
    rebalance_dates = set(pd.date_range(data.index[0], data.index[-1], freq=rebalance_freq))
    current_weights = pd.Series(0.0, index=data.columns)
    for i, date in enumerate(data.index[1:], 1):
        if date in rebalance_dates:
            lookback_start = date - pd.offsets.MonthEnd(lookback_months)
            if lookback_start in data.index:
                lookback_data = data.loc[lookback_start:date].pct_change().sum()
                top_etfs = lookback_data.sort_values(ascending=False).head(top_n)
                momenta = top_etfs.clip(lower=0)
                if momenta.sum() > 0:
                    weights = momenta / momenta.sum()
                else:
                    weights = pd.Series(1.0 / top_n, index=top_etfs.index)
                current_weights = pd.Series(0.0, index=data.columns)
                current_weights[top_etfs.index] = weights.values.astype(float)
        portfolio.iloc[i, 1:] = current_weights.values
        prev_value = portfolio.iloc[i-1, 0]
        daily_return = (returns.loc[date] * current_weights).sum()
        portfolio.iloc[i, 0] = prev_value * (1 + daily_return)
    for col in data.columns:
        portfolio[col] = portfolio[col].ffill()
    portfolio['Portfolio_Value'] = portfolio['Portfolio_Value'].ffill()
    return portfolio


def rsi_strategy(data, rsi_period, rebalance_freq, oversold_threshold, top_n):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio_Value'] + list(data.columns))
    portfolio = portfolio.astype({'Portfolio_Value': float, **{col: float for col in data.columns}})
    portfolio.iloc[0, 0] = 10000.0
    returns = data.pct_change().fillna(0)
    rebalance_dates = set(pd.date_range(data.index[0], data.index[-1], freq=rebalance_freq))
    current_weights = pd.Series(0.0, index=data.columns)
    for i, date in enumerate(data.index[1:], 1):
        if date in rebalance_dates:
            rsi_vals = pd.Series(index=data.columns, dtype=float)
            for col in data.columns:
                col_data = data[col]
                delta = col_data.diff()
                gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / (loss + 1e-8)
                rsi_vals[col] = 100 - (100 / (1 + rs.iloc[-1]))
            oversold_etfs = rsi_vals[rsi_vals < oversold_threshold].sort_values()
            if len(oversold_etfs) >= 1:
                selected = oversold_etfs.head(top_n)
            else:
                selected = rsi_vals.sort_values().head(top_n)
            if len(selected) > 0:
                weights = pd.Series(1.0 / len(selected), index=selected.index)
                current_weights = pd.Series(0.0, index=data.columns)
                current_weights[selected.index] = weights.values.astype(float)
            else:
                current_weights = pd.Series(0.0, index=data.columns)
        portfolio.iloc[i, 1:] = current_weights.values
        prev_value = portfolio.iloc[i-1, 0]
        daily_return = (returns.loc[date] * current_weights).sum()
        portfolio.iloc[i, 0] = prev_value * (1 + daily_return)
    for col in data.columns:
        portfolio[col] = portfolio[col].ffill()
    portfolio['Portfolio_Value'] = portfolio['Portfolio_Value'].ffill()
    return portfolio


def mean_reversion_strategy(data, lookback_months, z_threshold, rebalance_freq, top_n):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio_Value'] + list(data.columns))
    portfolio = portfolio.astype({'Portfolio_Value': float, **{col: float for col in data.columns}})
    portfolio.iloc[0, 0] = 10000.0
    returns = data.pct_change().fillna(0)
    rebalance_dates = set(pd.date_range(data.index[0], data.index[-1], freq=rebalance_freq))
    current_weights = pd.Series(0.0, index=data.columns)
    for i, date in enumerate(data.index[1:], 1):
        if date in rebalance_dates:
            lookback_start = date - pd.offsets.MonthEnd(lookback_months)
            if lookback_start in data.index:
                lookback_data = data.loc[lookback_start:date]
                cumulative = lookback_data.pct_change().sum()
                rolling_mean = cumulative.mean()
                rolling_std = cumulative.std()
                z_scores = (cumulative - rolling_mean) / rolling_std
                oversold = z_scores[z_scores < -z_threshold].sort_values()
                selected = oversold.head(top_n)
                if len(selected) > 0:
                    weights = pd.Series(1.0 / len(selected), index=selected.index)
                    current_weights = pd.Series(0.0, index=data.columns)
                    current_weights[selected.index] = weights.values.astype(float)
                else:
                    current_weights = pd.Series(0.0, index=data.columns)
        portfolio.iloc[i, 1:] = current_weights.values
        prev_value = portfolio.iloc[i-1, 0]
        daily_return = (returns.loc[date] * current_weights).sum()
        portfolio.iloc[i, 0] = prev_value * (1 + daily_return)
    for col in data.columns:
        portfolio[col] = portfolio[col].ffill()
    portfolio['Portfolio_Value'] = portfolio['Portfolio_Value'].ffill()
    return portfolio


def relative_strength_strategy(data, sp500_series, lookback_months, rebalance_freq, top_n):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio_Value'] + list(data.columns))
    portfolio = portfolio.astype({'Portfolio_Value': float, **{col: float for col in data.columns}})
    portfolio.iloc[0, 0] = 10000.0
    returns = data.pct_change().fillna(0)
    sp500_reindexed = sp500_series.reindex(data.index, method='ffill')
    sp500_returns = sp500_reindexed.pct_change().fillna(0)
    rebalance_dates = set(pd.date_range(data.index[0], data.index[-1], freq=rebalance_freq))
    current_weights = pd.Series(0.0, index=data.columns)
    for i, date in enumerate(data.index[1:], 1):
        if date in rebalance_dates:
            lookback_start = date - pd.offsets.MonthEnd(lookback_months)
            if lookback_start in data.index:
                etf_returns = data.loc[lookback_start:date].pct_change().sum()
                sp500_return = sp500_reindexed.loc[lookback_start:date].pct_change().sum()
                relative_strength = etf_returns - sp500_return
                top_etfs = relative_strength.sort_values(ascending=False).head(top_n)
                momenta = top_etfs.clip(lower=0)
                if momenta.sum() > 0:
                    weights = momenta / momenta.sum()
                else:
                    weights = pd.Series(1.0 / top_n, index=top_etfs.index)
                current_weights = pd.Series(0.0, index=data.columns)
                current_weights[top_etfs.index] = weights.values.astype(float)
        portfolio.iloc[i, 1:] = current_weights.values
        prev_value = portfolio.iloc[i-1, 0]
        daily_return = (returns.loc[date] * current_weights).sum()
        portfolio.iloc[i, 0] = prev_value * (1 + daily_return)
    for col in data.columns:
        portfolio[col] = portfolio[col].ffill()
    portfolio['Portfolio_Value'] = portfolio['Portfolio_Value'].ffill()
    return portfolio


def sharpe_momentum_strategy(data, lookback_months, rebalance_freq, top_n):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio_Value'] + list(data.columns))
    portfolio = portfolio.astype({'Portfolio_Value': float, **{col: float for col in data.columns}})
    portfolio.iloc[0, 0] = 10000.0
    returns = data.pct_change().fillna(0)
    rebalance_dates = set(pd.date_range(data.index[0], data.index[-1], freq=rebalance_freq))
    current_weights = pd.Series(0.0, index=data.columns)
    for i, date in enumerate(data.index[1:], 1):
        if date in rebalance_dates:
            lookback_start = date - pd.offsets.MonthEnd(lookback_months)
            if lookback_start in data.index:
                lookback_returns = data.loc[lookback_start:date].pct_change().dropna()
                if len(lookback_returns) > 1:
                    cumulative_returns = lookback_returns.sum()
                    rolling_std = lookback_returns.std()
                    sharpe_ratios = cumulative_returns / (rolling_std + 1e-8)
                    top_etfs = sharpe_ratios.sort_values(ascending=False).head(top_n)
                    selected = top_etfs[top_etfs > 0]
                    if len(selected) > 0:
                        weights = selected / selected.sum()
                    else:
                        weights = pd.Series(1.0 / top_n, index=top_etfs.index)
                    current_weights = pd.Series(0.0, index=data.columns)
                    current_weights[selected.index if len(selected) > 0 else top_etfs.index] = weights.values.astype(float)
        portfolio.iloc[i, 1:] = current_weights.values
        prev_value = portfolio.iloc[i-1, 0]
        daily_return = (returns.loc[date] * current_weights).sum()
        portfolio.iloc[i, 0] = prev_value * (1 + daily_return)
    for col in data.columns:
        portfolio[col] = portfolio[col].ffill()
    portfolio['Portfolio_Value'] = portfolio['Portfolio_Value'].ffill()
    return portfolio


# Run selected strategy
if strategy_type == "Momentum (Top N)":
    portfolio = momentum_strategy(data, lookback_months, rebalance_freq, top_n)
elif strategy_type == "RSI (Oversold Sectors)":
    portfolio = rsi_strategy(data, rsi_period, rebalance_freq, oversold_threshold, top_n)
elif strategy_type == "Mean Reversion":
    portfolio = mean_reversion_strategy(data, lookback_months, z_threshold, rebalance_freq, top_n)
elif strategy_type == "Relative Strength (vs S&P 500)":
    portfolio = relative_strength_strategy(data, sp500, lookback_months, rebalance_freq, top_n)
else:  # Risk-Adjusted Momentum (Sharpe)
    portfolio = sharpe_momentum_strategy(data, lookback_months, rebalance_freq, top_n)

# Calculate ETF allocation percentages for stack plot
allocation = portfolio[data.columns]

# Calculate S&P 500 portfolio value (normalized to $10,000 at start)
sp500 = get_sp500_data(start_date, end_date)
sp500 = sp500.reindex(portfolio.index, method='ffill')
sp500_returns = sp500.pct_change().fillna(0)
# Ensure sp500 is a 1D Series (not DataFrame or 2D array)
if isinstance(sp500, pd.DataFrame):
    sp500 = sp500.squeeze(axis=1)
sp500_value = (1 + sp500_returns).cumprod() * 10000

# Visualize results
if enable_tax:
    st.subheader("Portfolio Performance vs S&P 500 (After-Tax)")
else:
    st.subheader("Portfolio Performance vs S&P 500")

# Calculate after-tax portfolio values (will be added to plot_data if tax enabled)
after_tax_portfolio = portfolio['Portfolio_Value'].copy()

# This will be used after trades_df is defined below
def create_plot_data(portfolio_val, sp500_val, trades_df, enable_tax):
    plot_data = pd.DataFrame({
        'S&P 500': sp500_val,
        'Strategy': portfolio_val
    })
    after_tax = portfolio_val.copy()
    if enable_tax and not trades_df.empty:
        trades_sorted = trades_df.sort_values('Date').reset_index(drop=True)
        for _, trade in trades_sorted.iterrows():
            if trade['Action'] == 'Sell' and trade['Tax'] is not None and trade['Tax'] > 0:
                after_tax.loc[trade['Date']:] = after_tax.loc[trade['Date']:] - trade['Tax']
        plot_data['Strategy (Pre-Tax)'] = portfolio_val
        plot_data['Strategy (After-Tax)'] = after_tax
        plot_data = plot_data.drop(columns=['Strategy'])
    return plot_data

# Create display names for ETFs
etfs_display = {ticker: f"{ticker} - {name}" for ticker, name in etfs.items()}

st.subheader("ETF Allocation Stack Plot")
# Rename columns for display
allocation_display = allocation.rename(columns=etfs_display)
fig_stack = px.area(
    allocation_display,
    x=allocation_display.index,
    y=allocation_display.columns,
    title='ETF Allocation Percentage Over Time',
    labels={'value': 'Allocation Percentage', 'variable': 'ETF'}
)
fig_stack.update_yaxes(range=[0, 1])
st.plotly_chart(fig_stack, use_container_width=True)

st.subheader("Monthly Returns of ETFs")
# Rename columns for display
monthly_returns_display = monthly_returns.rename(columns=etfs_display)
fig3 = px.line(monthly_returns_display, y=monthly_returns_display.columns, title='Monthly Returns by ETF')
st.plotly_chart(fig3, use_container_width=True)

# Identify buy and sell signals for multi-ETF strategy, including amount, shares, and PnL
trades = []
prev_weights = pd.Series(0, index=allocation.columns)
prev_portfolio_value = portfolio.iloc[0]['Portfolio_Value']
# Track open positions for PnL calculation
open_positions = {etf: {'amount': 0, 'shares': 0, 'buy_price': 0, 'buy_date': None} for etf in allocation.columns}
for i, (date, row) in enumerate(allocation.iterrows()):
    curr_weights = row
    curr_portfolio_value = portfolio.iloc[i]['Portfolio_Value']
    prices = data.loc[date] if date in data.index else None
    for etf in allocation.columns:
        # Buy: weight goes from 0 to >0
        if prev_weights[etf] == 0 and curr_weights[etf] > 0:
            amount = curr_portfolio_value * curr_weights[etf]
            shares = amount / prices[etf] if prices is not None and prices[etf] > 0 else float('nan')
            trades.append({'Date': date, 'Action': 'Buy', 'ETF': etf, 'Amount': amount, 'Shares': shares, 'PnL': None, 'Tax': None, 'AfterTaxPnL': None})
            open_positions[etf] = {'amount': amount, 'shares': shares, 'buy_price': prices[etf] if prices is not None else float('nan'), 'buy_date': date}
        # Sell: weight goes from >0 to 0
        if prev_weights[etf] > 0 and curr_weights[etf] == 0:
            amount = curr_portfolio_value * prev_weights[etf]
            shares = amount / prices[etf] if prices is not None and prices[etf] > 0 else float('nan')
            # Calculate PnL if we have a buy record
            buy_info = open_positions.get(etf, None)
            if buy_info and buy_info['shares'] > 0 and prices is not None and prices[etf] > 0:
                pnl = (prices[etf] - buy_info['buy_price']) * buy_info['shares']
                holding_days = (date - buy_info['buy_date']).days if buy_info['buy_date'] else 0
                is_long_term = holding_days > 365
                # For simplicity, use top tax rate for both (user can adjust)
                tax = pnl * tax_rate if pnl > 0 and enable_tax else 0
                after_tax_pnl = pnl - tax if enable_tax else pnl
                trades.append({'Date': date, 'Action': 'Sell', 'ETF': etf, 'Amount': amount, 'Shares': shares, 'PnL': pnl, 'Tax': tax if enable_tax else None, 'AfterTaxPnL': after_tax_pnl if enable_tax else None})
            else:
                trades.append({'Date': date, 'Action': 'Sell', 'ETF': etf, 'Amount': amount, 'Shares': shares, 'PnL': None, 'Tax': None, 'AfterTaxPnL': None})
            open_positions[etf] = {'amount': 0, 'shares': 0, 'buy_price': 0, 'buy_date': None}
    prev_weights = curr_weights
    prev_portfolio_value = curr_portfolio_value
trades_df = pd.DataFrame(trades)

# Now create plot data after trades_df is defined
plot_data = create_plot_data(portfolio['Portfolio_Value'], sp500_value, trades_df, enable_tax)

fig1 = px.line(
    plot_data,
    x=plot_data.index,
    y=plot_data.columns,
    title='Portfolio Value Over Time (vs S&P 500)'
)
st.plotly_chart(fig1, use_container_width=True)

# Tax efficiency note
st.subheader("Tax Efficiency Tips")
st.write("""
- **Use Tax-Advantaged Accounts**: Execute this strategy in an IRA or 401(k) to avoid capital gains taxes.
- **Long-Term Holding**: Hold ETFs for over a year to qualify for lower long-term capital gains rates (0%, 15%, or 20% in the U.S.).
- **Wash-Sale Avoidance**: When selling an ETF at a loss, swap to an ETF tracking a different index (e.g., XLK to VGT) to avoid the wash-sale rule.
- **Tax-Loss Harvesting**: Sell underperforming ETFs to offset gains elsewhere in your portfolio.
- **Consult a Tax Advisor**: Tax rules vary by jurisdiction. Verify strategies with a professional.
""")

# Option to download portfolio data
st.subheader("Download Results")
csv = portfolio.to_csv().encode('utf-8')
st.download_button("Download Portfolio Data", csv, "portfolio_data.csv", "text/csv")

# Display trades
st.subheader("Buy and Sell Transactions")
if not trades_df.empty:
    cols = ['Date', 'Action', 'ETF', 'Amount', 'Shares', 'PnL']
    if enable_tax:
        cols.extend(['Tax', 'AfterTaxPnL'])
    st.dataframe(trades_df[cols].sort_values('Date').reset_index(drop=True))
else:
    st.write("No buy and sell transactions detected.")