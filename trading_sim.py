import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import List
from strategies import TradingStrategy, MACrossoverStrategy, RSIStrategy, BollingerBandsStrategy, PriceMomentumStrategy

st.set_page_config(layout="wide")
@dataclass
class TradeRecord:
    date: datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    price: float
    shares: float = 1000  # Default position size
    pnl: float = 0.0
    strategy: str = ""
    
def calculate_trade_metrics(signals: pd.DataFrame, price_data: pd.DataFrame, strategy_name: str, ticker: str, starting_cash: float) -> List[TradeRecord]:
    trades = []
    position = 0
    entry_price = 0
    cash = starting_cash
    shares_held = 0
    
    for date, row in signals.iterrows():
        if row['positions'] == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = row['price']
            # Buy as many shares as possible with available cash
            shares_to_buy = int(cash // entry_price)
            cost = shares_to_buy * entry_price
            cash -= cost
            shares_held = shares_to_buy
            trades.append(TradeRecord(
                date=date,
                ticker=ticker,
                action='BUY',
                price=entry_price,
                shares=shares_to_buy,
                strategy=strategy_name
            ))
        elif row['positions'] == -1 and position == 1:  # Sell signal
            position = 0
            exit_price = row['price']
            proceeds = shares_held * exit_price
            pnl = (exit_price - entry_price) * shares_held
            cash += proceeds
            trades.append(TradeRecord(
                date=date,
                ticker=ticker,
                action='SELL',
                price=exit_price,
                shares=shares_held,
                pnl=pnl,
                strategy=strategy_name
            ))
            shares_held = 0
    
    return trades, cash

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
    # Add starting cash to sidebar
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=1000))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    starting_cash = st.sidebar.number_input("Starting Cash ($)", min_value=1000, max_value=10000000, value=100000, step=1000)

    # Strategy selection
    strategy_choice = st.sidebar.selectbox(
        "Trading Strategy",
        [
            "Moving Average Crossover",
            "RSI",
            "Bollinger Bands Breakout",
            "Price Momentum"
        ]
    )

    # Strategy-specific parameters
    if strategy_choice == "Moving Average Crossover":
        short_window = st.sidebar.slider("Short MA Window", min_value=5, max_value=50, value=20)
        long_window = st.sidebar.slider("Long MA Window", min_value=20, max_value=200, value=50)
    elif strategy_choice == "RSI":
        rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=50, value=14)
        overbought = st.sidebar.slider("Overbought Threshold", min_value=50, max_value=90, value=70)
        oversold = st.sidebar.slider("Oversold Threshold", min_value=10, max_value=50, value=30)
    elif strategy_choice == "Bollinger Bands Breakout":
        bb_window = st.sidebar.slider("BB Window", min_value=5, max_value=50, value=20)
        bb_num_std = st.sidebar.slider("BB Num Std Dev", min_value=1, max_value=4, value=2)
    else:  # Price Momentum
        momentum_window = st.sidebar.slider("Momentum Window", min_value=2, max_value=50, value=10)

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
            elif strategy_choice == "Bollinger Bands Breakout":
                strategy = BollingerBandsStrategy(window=bb_window, num_std=bb_num_std)
            else:
                strategy = PriceMomentumStrategy(momentum_window=momentum_window)
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
            elif strategy_choice == "Bollinger Bands Breakout":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['upper_band'], 
                                       name='Upper Band', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['lower_band'], 
                                       name='Lower Band', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['price'], 
                                       name='Price', line=dict(color='blue')))
            elif strategy_choice == "Price Momentum":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['momentum'], 
                                       name='Momentum', line=dict(color='orange')))

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
                yaxis_title="Price (USD) / RSI / Momentum" if strategy_choice in ["RSI", "Price Momentum"] else "Price (USD)",
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
            st.write(f"Cumulative Returns (Strategy): {cum_returns:.2%}")

            # Buy and Hold comparison
            buy_price = data['Close'].iloc[0]
            sell_price = data['Close'].iloc[-1]
            shares_bh = int(starting_cash // buy_price)
            cash_bh = starting_cash - (shares_bh * buy_price)
            final_bh_value = shares_bh * sell_price + cash_bh
            cum_bh_return = (final_bh_value - starting_cash) / starting_cash
            st.write(f"Cumulative Returns (Buy & Hold): {cum_bh_return:.2%}")

            # Annual returns for strategy
            strat_portfolio = (1 + strategy_returns).cumprod() * starting_cash
            strat_portfolio.index = pd.to_datetime(strat_portfolio.index)
            strat_annual = strat_portfolio.resample('Y').last().pct_change().dropna()
            # Annual returns for buy & hold
            bh_portfolio = (data['Close'] / buy_price) * starting_cash
            bh_portfolio.index = pd.to_datetime(bh_portfolio.index)
            bh_annual = bh_portfolio.resample('Y').last().pct_change().dropna()

            annual_df = pd.DataFrame({
                'Strategy Annual Return': strat_annual,
                'Buy & Hold Annual Return': bh_annual
            })
            annual_df = annual_df.applymap(lambda x: f"{x:.2%}")
            st.write("Annual Returns:")
            st.dataframe(annual_df)
            
            # Calculate and display trade blotter
            st.subheader("Trade Blotter")
            trades, final_cash = calculate_trade_metrics(signals, data, strategy_choice, ticker, starting_cash)
            if trades:
                trades_df = pd.DataFrame([vars(trade) for trade in trades])
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df = trades_df.sort_values('date')
                
                # Format the trade blotter
                trades_df['price'] = trades_df['price'].round(2)
                trades_df['pnl'] = trades_df['pnl'].round(2)
                trades_df['value'] = (trades_df['price'] * trades_df['shares']).round(2)
                
                # Calculate cumulative P&L
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum().round(2)
                
                # Display formatted trade blotter
                st.dataframe(
                    trades_df[[
                        'date', 'ticker', 'action', 'price', 'shares', 'value',
                        'pnl', 'cumulative_pnl', 'strategy'
                    ]].style.format({
                        'price': '${:.2f}',
                        'value': '${:,.2f}',
                        'pnl': '${:,.2f}',
                        'cumulative_pnl': '${:,.2f}'
                    }).background_gradient(subset=['pnl', 'cumulative_pnl'], cmap='RdYlGn')
                )
                
                # Show final cash balance
                st.write(f"Final Cash Balance: ${final_cash:,.2f}")
                
                # Add download button for trade blotter
                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Trade Blotter",
                    csv,
                    f"{ticker}_{strategy_choice}_trades.csv",
                    "text/csv",
                    key='download-trades'
                )
            else:
                st.write("No trades generated during this period.")

            # Annualized return (CAGR) for strategy
            n_years = (strat_portfolio.index[-1] - strat_portfolio.index[0]).days / 365.25
            strat_final = strat_portfolio.iloc[-1]
            strat_cagr = (strat_final / starting_cash) ** (1 / n_years) - 1
            # Annualized return (CAGR) for buy & hold
            bh_final = bh_portfolio.iloc[-1]
            bh_cagr = (bh_final / starting_cash) ** (1 / n_years) - 1
            st.write(f"Annualized Return (Strategy): {strat_cagr:.2%}")
            st.write(f"Annualized Return (Buy & Hold): {bh_cagr:.2%}")

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

if __name__ == "__main__":
    main()