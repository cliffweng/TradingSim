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
This app demonstrates a momentum-based sector rotation strategy using popular sector ETFs.
The strategy rebalances quarterly, investing in the ETF with the highest 6-month return.
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

# User input for date range
st.subheader("Select Date Range")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime(2025, 7, 1))

# Sidebar for user-configurable parameters
st.sidebar.header("Strategy Parameters")
lookback_months = st.sidebar.slider("Lookback Period (months)", min_value=1, max_value=12, value=6)
top_n = st.sidebar.slider("Number of ETFs to Hold", min_value=1, max_value=len(etfs), value=3)

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
data = get_etf_data(etfs, start_date, end_date)

# Calculate monthly returns for visualization
monthly_returns = data.resample('ME').last().pct_change().dropna()

# Momentum-based sector rotation strategy
def sector_rotation_strategy(data, lookback_months=6, rebalance_freq='ME', top_n=3):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio_Value'] + list(data.columns))
    # Ensure all columns are float dtype
    portfolio = portfolio.astype({'Portfolio_Value': float, **{col: float for col in data.columns}})
    portfolio.iloc[0, 0] = 10000.0  # Start with $10,000 as float
    returns = data.pct_change().fillna(0)
    rebalance_dates = set(pd.date_range(data.index[0], data.index[-1], freq=rebalance_freq))
    current_weights = pd.Series(0.0, index=data.columns)  # float dtype
    for i, date in enumerate(data.index[1:], 1):
        # Rebalance if it's a rebalance date
        if date in rebalance_dates:
            lookback_start = date - pd.offsets.MonthEnd(lookback_months)
            if lookback_start in data.index:
                lookback_data = data.loc[lookback_start:date].pct_change().sum()
                top_etfs = lookback_data.sort_values(ascending=False).head(top_n)
                # Momentum-based weights: proportional to positive momentum, normalized
                momenta = top_etfs.clip(lower=0)
                if momenta.sum() > 0:
                    weights = momenta / momenta.sum()
                else:
                    weights = pd.Series(1.0 / top_n, index=top_etfs.index)
                current_weights = pd.Series(0.0, index=data.columns)
                current_weights[top_etfs.index] = weights.values.astype(float)
        # Carry forward weights if not a rebalance date
        portfolio.iloc[i, 1:] = current_weights.values
        # Update portfolio value based on weighted return
        prev_value = portfolio.iloc[i-1, 0]
        daily_return = (returns.loc[date] * current_weights).sum()
        portfolio.iloc[i, 0] = prev_value * (1 + daily_return)
    # Forward fill weights for any missing values
    for col in data.columns:
        inferred = portfolio[col].infer_objects(copy=False)
        portfolio[col] = inferred.ffill()
    inferred_val = portfolio['Portfolio_Value'].infer_objects(copy=False)
    portfolio['Portfolio_Value'] = inferred_val.ffill()
    return portfolio

# Run strategy with user parameters
portfolio = sector_rotation_strategy(data, lookback_months=lookback_months, rebalance_freq='ME', top_n=top_n)

# Calculate ETF allocation percentages for stack plot
allocation = portfolio[data.columns]

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
        # Always return as Series (not DataFrame)
        if isinstance(sp500, pd.DataFrame):
            sp500 = sp500.squeeze()
        return sp500
    except Exception as e:
        st.warning(f"S&P 500 data fetch failed: {e}")
        return pd.Series(index=pd.date_range(start=start, end=end, freq='D'), dtype=float)

sp500 = get_sp500_data(start_date, end_date)
# Calculate S&P 500 portfolio value (normalized to $10,000 at start)
sp500 = sp500.reindex(portfolio.index, method='ffill')
sp500_returns = sp500.pct_change().fillna(0)
# Ensure sp500 is a 1D Series (not DataFrame or 2D array)
if isinstance(sp500, pd.DataFrame):
    sp500 = sp500.squeeze(axis=1)
sp500_value = (1 + sp500_returns).cumprod() * 10000

# Visualize results
st.subheader("Portfolio Performance vs S&P 500")
fig1 = px.line(
    pd.DataFrame({
        'Strategy': portfolio['Portfolio_Value'],
        'S&P 500': sp500_value
    }),
    x=portfolio.index,
    y=['Strategy', 'S&P 500'],
    title='Portfolio Value Over Time (vs S&P 500)'
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("ETF Allocation Stack Plot")
fig_stack = px.area(
    allocation,
    x=allocation.index,
    y=allocation.columns,
    title='ETF Allocation Percentage Over Time',
    labels={'value': 'Allocation Percentage', 'variable': 'ETF'}
)
fig_stack.update_yaxes(range=[0, 1])
st.plotly_chart(fig_stack, use_container_width=True)

st.subheader("Monthly Returns of ETFs")
fig3 = px.line(monthly_returns, x=monthly_returns.index, y=monthly_returns.columns, title='Monthly Returns by ETF')
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
            trades.append({'Date': date, 'Action': 'Buy', 'ETF': etf, 'Amount': amount, 'Shares': shares, 'PnL': None})
            open_positions[etf] = {'amount': amount, 'shares': shares, 'buy_price': prices[etf] if prices is not None else float('nan'), 'buy_date': date}
        # Sell: weight goes from >0 to 0
        if prev_weights[etf] > 0 and curr_weights[etf] == 0:
            amount = curr_portfolio_value * prev_weights[etf]
            shares = amount / prices[etf] if prices is not None and prices[etf] > 0 else float('nan')
            # Calculate PnL if we have a buy record
            buy_info = open_positions.get(etf, None)
            if buy_info and buy_info['shares'] > 0 and prices is not None and prices[etf] > 0:
                pnl = (prices[etf] - buy_info['buy_price']) * buy_info['shares']
            else:
                pnl = None
            trades.append({'Date': date, 'Action': 'Sell', 'ETF': etf, 'Amount': amount, 'Shares': shares, 'PnL': pnl})
            open_positions[etf] = {'amount': 0, 'shares': 0, 'buy_price': 0, 'buy_date': None}
    prev_weights = curr_weights
    prev_portfolio_value = curr_portfolio_value
trades_df = pd.DataFrame(trades)

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
    st.dataframe(trades_df[['Date', 'Action', 'ETF', 'Amount', 'Shares', 'PnL']].sort_values('Date').reset_index(drop=True))
else:
    st.write("No buy or sell transactions detected.")