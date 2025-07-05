import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from functools import lru_cache
from strategies import TradingStrategy, MACrossoverStrategy, RSIStrategy, BollingerBandsStrategy

# Cache historical data
@lru_cache(maxsize=32)
def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

# Streamlit app
def main():
    st.title("Trading Simulation App")

    # Sidebar for user inputs
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())

    # Strategy selection
    strategy_choice = st.sidebar.selectbox(
        "Trading Strategy",
        ["Moving Average Crossover", "RSI", "Bollinger Bands Breakout"]
    )

    # Strategy-specific parameters
    if strategy_choice == "Moving Average Crossover":
        short_window = st.sidebar.slider("Short MA Window", min_value=5, max_value=50, value=20)
        long_window = st.sidebar.slider("Long MA Window", min_value=20, max_value=200, value=50)
    elif strategy_choice == "RSI":
        rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=50, value=14)
        overbought = st.sidebar.slider("Overbought Threshold", min_value=50, max_value=90, value=70)
        oversold = st.sidebar.slider("Oversold Threshold", min_value=10, max_value=50, value=30)
    else:  # Bollinger Bands Breakout
        bb_window = st.sidebar.slider("BB Window", min_value=5, max_value=50, value=20)
        bb_num_std = st.sidebar.slider("BB Num Std Dev", min_value=1, max_value=4, value=2)

    # Fetch data
    if ticker and start_date and end_date:
        try:
            with st.spinner("Fetching historical data..."):
                data = fetch_historical_data(ticker, start_date.strftime("%Y-%m-%d"), 
                                           end_date.strftime("%Y-%m-%d"))
            
            if data.empty:
                st.error("No data available for the selected ticker and date range.")
                return

            # Initialize strategy
            if strategy_choice == "Moving Average Crossover":
                strategy = MACrossoverStrategy(short_window=short_window, long_window=long_window)
            elif strategy_choice == "RSI":
                strategy = RSIStrategy(rsi_period=rsi_period, overbought=overbought, oversold=oversold)
            else:
                strategy = BollingerBandsStrategy(window=bb_window, num_std=bb_num_std)
            signals = strategy.generate_signals(data)

            # Create Plotly chart
            fig = go.Figure()

            # Add price line
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                   name='Price', line=dict(color='blue')))
            
            # Add strategy-specific lines
            if strategy_choice == "Moving Average Crossover":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['short_mavg'], 
                                       name=f'Short MA ({short_window})', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['long_mavg'], 
                                       name=f'Long MA ({long_window})', line=dict(color='green')))
            elif strategy_choice == "RSI":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['rsi'], 
                                       name='RSI', line=dict(color='purple')))
                fig.add_hline(y=overbought, line_dash="dash", line_color="red", name='Overbought')
                fig.add_hline(y=oversold, line_dash="dash", line_color="green", name='Oversold')
            else:
                fig.add_trace(go.Scatter(x=signals.index, y=signals['upper_band'], 
                                       name='Upper Band', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['lower_band'], 
                                       name='Lower Band', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['price'], 
                                       name='Price', line=dict(color='blue')))

            # Add buy/sell signals
            buy_signals = signals[signals['positions'] == 1]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['price'], 
                                   name='Buy Signal', mode='markers', 
                                   marker=dict(symbol='triangle-up', size=10, color='green')))
            sell_signals = signals[signals['positions'] == -1]
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['price'], 
                                   name='Sell Signal', mode='markers', 
                                   marker=dict(symbol='triangle-down', size=10, color='red')))

            # Update layout
            fig.update_layout(
                title=f"{ticker} Stock Price with {strategy_choice} Signals",
                xaxis_title="Date",
                yaxis_title="Price (USD) / RSI" if strategy_choice == "RSI" else "Price (USD)",
                template="plotly_white",
                height=600
            )

            # Display chart
            st.plotly_chart(fig, use_container_width=True)

            # Display performance metrics
            st.subheader("Performance Metrics")
            returns = signals['price'].pct_change()
            strategy_returns = returns * signals['signal'].shift(1)
            total_trades = len(buy_signals) + len(sell_signals)
            cum_returns = (1 + strategy_returns).cumprod().iloc[-1] - 1
            st.write(f"Total Trades: {total_trades}")
            st.write(f"Cumulative Returns: {cum_returns:.2%}")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    main()