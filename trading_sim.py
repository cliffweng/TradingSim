import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import List
from strategies import TradingStrategy, MACrossoverStrategy, RSIStrategy, BollingerBandsStrategy, PriceMomentumStrategy, MACDStrategy, MeanReversionStrategy, DonchianChannelStrategy, StochasticOscillatorStrategy, ATRStrategy, DualThrustStrategy

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
    reason: str = ""
    tax: float = 0.0
    after_tax_pnl: float = 0.0
    
def calculate_trade_metrics(signals: pd.DataFrame, price_data: pd.DataFrame, strategy_name: str, ticker: str, starting_cash: float, enable_tax: bool = False, tax_rate: float = 0.0) -> List[TradeRecord]:
    trades = []
    position = 0
    entry_price = 0
    entry_reason = ""
    cash = starting_cash
    shares_held = 0
    
    for date, row in signals.iterrows():
        signal_row = signals.loc[date]
        if row['positions'] == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = row['price']
            shares_to_buy = int(cash // entry_price)
            cost = shares_to_buy * entry_price
            cash -= cost
            shares_held = shares_to_buy
            reason = get_trade_reason("BUY", signal_row, strategy_name)
            trades.append(TradeRecord(
                date=date,
                ticker=ticker,
                action='BUY',
                price=entry_price,
                shares=shares_to_buy,
                strategy=strategy_name,
                reason=reason
            ))
        elif row['positions'] == -1 and position == 1:  # Sell signal
            position = 0
            exit_price = row['price']
            proceeds = shares_held * exit_price
            pnl = (exit_price - entry_price) * shares_held
            reason = get_trade_reason("SELL", signal_row, strategy_name)
            # Calculate tax on gains if enabled
            tax = pnl * tax_rate if enable_tax and pnl > 0 else 0.0
            after_tax_pnl = pnl - tax if enable_tax else pnl
            # Subtract tax from proceeds
            cash += (proceeds - tax)
            trades.append(TradeRecord(
                date=date,
                ticker=ticker,
                action='SELL',
                price=exit_price,
                shares=shares_held,
                pnl=pnl,
                strategy=strategy_name,
                reason=reason,
                tax=tax,
                after_tax_pnl=after_tax_pnl
            ))
            shares_held = 0
    
    return trades, cash


def get_trade_reason(action: str, signal_row, strategy_name: str) -> str:
    if strategy_name == "Moving Average Crossover":
        return f"Short MA crossed above Long MA" if action == "BUY" else f"Short MA crossed below Long MA"
    elif strategy_name == "RSI":
        rsi = signal_row.get('rsi', 0)
        return f"RSI oversold ({rsi:.1f})" if action == "BUY" else f"RSI overbought ({rsi:.1f})"
    elif strategy_name == "Bollinger Bands Breakout":
        return f"Price below lower band" if action == "BUY" else f"Price above upper band"
    elif strategy_name == "Price Momentum":
        mom = signal_row.get('momentum', 0)
        return f"Positive momentum ({mom:.2f})" if action == "BUY" else f"Negative momentum ({mom:.2f})"
    elif strategy_name == "MACD":
        macd = signal_row.get('macd', 0)
        signal_line = signal_row.get('signal_line', 0)
        return f"MACD above signal line ({macd:.2f} vs {signal_line:.2f})" if action == "BUY" else f"MACD below signal line ({macd:.2f} vs {signal_line:.2f})"
    elif strategy_name == "Mean Reversion":
        z = signal_row.get('z_score', 0)
        return f"Z-score oversold ({z:.2f})" if action == "BUY" else f"Z-score overbought ({z:.2f})"
    elif strategy_name == "Donchian Channel":
        return f"Price broke above upper channel" if action == "BUY" else f"Price broke below lower channel"
    elif strategy_name == "Stochastic Oscillator":
        k = signal_row.get('k', 0)
        d = signal_row.get('d', 0)
        return f"%K above %D in oversold ({k:.1f} vs {d:.1f})" if action == "BUY" else f"%K below %D in overbought ({k:.1f} vs {d:.1f})"
    elif strategy_name == "ATR Trailing Stop":
        return f"Price above ATR trailing stop" if action == "BUY" else f"Price below ATR trailing stop"
    elif strategy_name == "Dual Thrust":
        return f"Price broke above upper level" if action == "BUY" else f"Price broke below lower level"
    return f"{action} signal"

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
            "Price Momentum",
            "MACD",
            "Mean Reversion",
            "Donchian Channel",
            "Stochastic Oscillator",
            "ATR Trailing Stop",
            "Dual Thrust"
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
    elif strategy_choice == "Price Momentum":
        momentum_window = st.sidebar.slider("Momentum Window", min_value=2, max_value=50, value=10)
    elif strategy_choice == "MACD":
        fast_period = st.sidebar.slider("Fast EMA Period", min_value=5, max_value=30, value=12)
        slow_period = st.sidebar.slider("Slow EMA Period", min_value=15, max_value=50, value=26)
        signal_period = st.sidebar.slider("Signal Line Period", min_value=5, max_value=20, value=9)
    elif strategy_choice == "Mean Reversion":
        mr_window = st.sidebar.slider("Mean Reversion Window", min_value=5, max_value=50, value=20)
        z_threshold = st.sidebar.slider("Z-Score Threshold", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
    elif strategy_choice == "Donchian Channel":
        channel_period = st.sidebar.slider("Channel Period", min_value=5, max_value=50, value=20)
    elif strategy_choice == "Stochastic Oscillator":
        stoch_k = st.sidebar.slider("%K Period", min_value=5, max_value=30, value=14)
        stoch_d = st.sidebar.slider("%D Period", min_value=2, max_value=10, value=3)
        stoch_overbought = st.sidebar.slider("Stoch Overbought", min_value=50, max_value=90, value=80)
        stoch_oversold = st.sidebar.slider("Stoch Oversold", min_value=10, max_value=50, value=20)
    elif strategy_choice == "ATR Trailing Stop":
        atr_period = st.sidebar.slider("ATR Period", min_value=5, max_value=30, value=14)
        atr_mult = st.sidebar.slider("ATR Multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    else:  # Dual Thrust
        dt_lookback = st.sidebar.slider("Lookback Period", min_value=5, max_value=50, value=20)
        dt_k = st.sidebar.slider("K Value", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    # Tax settings
    st.sidebar.header("Tax Settings")
    enable_tax = st.sidebar.checkbox("Enable Tax-Aware Mode", value=True)
    tax_rate = 0.0  # default
    if enable_tax:
        tax_rate = st.sidebar.slider("Capital Gains Tax Rate (%)", min_value=0, max_value=50, value=37) / 100

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
            elif strategy_choice == "Price Momentum":
                strategy = PriceMomentumStrategy(momentum_window=momentum_window)
            elif strategy_choice == "MACD":
                strategy = MACDStrategy(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
            elif strategy_choice == "Mean Reversion":
                strategy = MeanReversionStrategy(window=mr_window, z_threshold=z_threshold)
            elif strategy_choice == "Donchian Channel":
                strategy = DonchianChannelStrategy(channel_period=channel_period)
            elif strategy_choice == "Stochastic Oscillator":
                strategy = StochasticOscillatorStrategy(k_period=stoch_k, d_period=stoch_d, overbought=stoch_overbought, oversold=stoch_oversold)
            elif strategy_choice == "ATR Trailing Stop":
                strategy = ATRStrategy(atr_period=atr_period, atr_multiplier=atr_mult)
            else:  # Dual Thrust
                strategy = DualThrustStrategy(lookback_period=dt_lookback, k_value=dt_k)
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
            elif strategy_choice == "MACD":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['macd'], 
                                       name='MACD', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['signal_line'], 
                                       name='Signal Line', line=dict(color='orange')))
            elif strategy_choice == "Mean Reversion":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['z_score'], 
                                       name='Z-Score', line=dict(color='purple')))
                fig.add_hline(y=z_threshold, line_dash="dash", line_color="red", name='Upper Threshold')
                fig.add_hline(y=-z_threshold, line_dash="dash", line_color="green", name='Lower Threshold')
            elif strategy_choice == "Donchian Channel":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['upper_channel'], 
                                       name='Upper Channel', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['lower_channel'], 
                                       name='Lower Channel', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['middle_channel'], 
                                       name='Middle Channel', line=dict(color='gray', dash='dot')))
            elif strategy_choice == "Stochastic Oscillator":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['k'], 
                                       name='%K', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['d'], 
                                       name='%D', line=dict(color='orange')))
                fig.add_hline(y=stoch_overbought, line_dash="dash", line_color="red", name='Overbought')
                fig.add_hline(y=stoch_oversold, line_dash="dash", line_color="green", name='Oversold')
            elif strategy_choice == "ATR Trailing Stop":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['trailing_stop'], 
                                       name='Trailing Stop', line=dict(color='red', dash='dash')))
            elif strategy_choice == "Dual Thrust":
                fig.add_trace(go.Scatter(x=signals.index, y=signals['upper_level'], 
                                       name='Upper Level', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=signals.index, y=signals['lower_level'], 
                                       name='Lower Level', line=dict(color='green', dash='dash')))

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
                yaxis_title="Price (USD) / Indicator" if strategy_choice in ["RSI", "Price Momentum", "MACD", "Mean Reversion"] else "Price (USD)",
                template="plotly_white",
                height=600
            )

            # Display chart
            st.plotly_chart(fig, use_container_width=True)

            # Display performance metrics
            st.subheader("Performance Metrics")
            returns = signals['price'].pct_change()
            strategy_returns = returns * signals['signal'].shift(1)
            cum_returns = (1 + strategy_returns).cumprod().iloc[-1] - 1
            
            total_trades = len(buy_signals) + len(sell_signals)
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

            # Calculate trades first (needed for tax calculations)
            trades, final_cash = calculate_trade_metrics(signals, data, strategy_choice, ticker, starting_cash, enable_tax, tax_rate)
            
            # Annual returns for strategy
            strat_portfolio = (1 + strategy_returns).cumprod() * starting_cash
            strat_portfolio.index = pd.to_datetime(strat_portfolio.index)
            
            # Build after-tax portfolio if enabled
            strat_portfolio_after_tax = strat_portfolio.copy()
            if enable_tax and trades:
                for trade in trades:
                    if trade.action == 'SELL' and trade.tax > 0:
                        trade_date = pd.Timestamp(trade.date)
                        # Find the first date index >= trade_date
                        later_dates = strat_portfolio_after_tax.index[strat_portfolio_after_tax.index >= trade_date]
                        if len(later_dates) > 0:
                            start_date = later_dates[0]
                            strat_portfolio_after_tax.loc[start_date:] -= trade.tax
            
            strat_annual = strat_portfolio.resample('YE').last().pct_change().dropna()
            
            # Annual returns for buy & hold
            bh_portfolio = (data['Close'] / buy_price) * starting_cash
            bh_portfolio.index = pd.to_datetime(bh_portfolio.index)
            bh_annual = bh_portfolio.resample('YE').last().pct_change().dropna()

            annual_df = pd.DataFrame({
                'Strategy Annual Return': strat_annual,
                'Buy & Hold Annual Return': bh_annual
            })
            
            if enable_tax and trades:
                strat_annual_after_tax = strat_portfolio_after_tax.resample('YE').last().pct_change().dropna()
                annual_df['Strategy After-Tax Return'] = strat_annual_after_tax
            
            annual_df = annual_df.map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            st.write("Annual Returns:")
            st.dataframe(annual_df)
            
            # Calculate and display trade blotter
            st.subheader("Trade Blotter")
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
                cols = ['date', 'ticker', 'action', 'price', 'shares', 'value', 'pnl', 'cumulative_pnl', 'strategy', 'reason']
                if enable_tax:
                    trades_df['tax'] = trades_df['tax'].round(2)
                    trades_df['after_tax_pnl'] = trades_df['after_tax_pnl'].round(2)
                    cols.extend(['tax', 'after_tax_pnl'])
                st.dataframe(
                    trades_df[cols].style.format({
                        'price': '${:.2f}',
                        'value': '${:,.2f}',
                        'pnl': '${:,.2f}',
                        'cumulative_pnl': '${:,.2f}',
                        'tax': '${:,.2f}',
                        'after_tax_pnl': '${:,.2f}'
                    }).background_gradient(subset=['pnl', 'cumulative_pnl'], cmap='RdYlGn')
                )
                
                # Show final cash balance
                st.write(f"Final Cash Balance: ${final_cash:,.2f}")
                
                # Show after-tax performance if enabled
                if enable_tax and trades:
                    total_tax = sum(t.tax for t in trades if t.tax > 0)
                    # Use the after-tax portfolio value
                    after_tax_final = strat_portfolio_after_tax.iloc[-1]
                    cum_after_tax = (after_tax_final - starting_cash) / starting_cash
                    st.write(f"Total Tax Paid: ${total_tax:,.2f}")
                    st.write(f"After-Tax Cumulative Returns: {cum_after_tax:.2%}")
                
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