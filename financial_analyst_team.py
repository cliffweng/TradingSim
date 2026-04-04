import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
from strategies import MACrossoverStrategy, RSIStrategy

# --- Analyst Classes ---

class DataFetcher:
    """Fetches financial data using yfinance."""
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def get_financials(self) -> Dict[str, pd.DataFrame]:
        """Retrieves balance sheet, income statement, and cash flow."""
        return {
            "balance_sheet": self.stock.balance_sheet,
            "income_stmt": self.stock.income_stmt,
            "cash_flow": self.stock.cashflow,
            "info": self.stock.info
        }

    def get_history(self, period="2y") -> pd.DataFrame:
        """Retrieves historical price data."""
        return self.stock.history(period=period)

class FundamentalAnalyst:
    """Analyzes financial statements and ratios."""
    def analyze(self, financials: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        info = financials.get("info", {})
        income_stmt = financials.get("income_stmt")
        balance_sheet = financials.get("balance_sheet")
        
        analysis = {
            "ratios": {},
            "trends": {},
            "health_score": 0
        }

        # Key Ratios from 'info' (most reliable for current snapshot)
        analysis["ratios"]["PE_Ratio"] = info.get("trailingPE")
        analysis["ratios"]["Forward_PE"] = info.get("forwardPE")
        analysis["ratios"]["PEG_Ratio"] = info.get("pegRatio")
        analysis["ratios"]["Profit_Margin"] = info.get("profitMargins")
        analysis["ratios"]["ROE"] = info.get("returnOnEquity")
        analysis["ratios"]["Debt_to_Equity"] = info.get("debtToEquity")

        # Trend Analysis (using historical statements if available)
        if income_stmt is not None and not income_stmt.empty:
            try:
                # Revenue Growth (YoY)
                revenues = income_stmt.loc["Total Revenue"]
                if len(revenues) >= 2:
                    growth = (revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1]
                    analysis["trends"]["Revenue_Growth_YoY"] = growth
                
                # Net Income Growth
                net_income = income_stmt.loc["Net Income"]
                if len(net_income) >= 2:
                    ni_growth = (net_income.iloc[0] - net_income.iloc[1]) / net_income.iloc[1]
                    analysis["trends"]["Net_Income_Growth_YoY"] = ni_growth
            except KeyError:
                pass # Some keys might vary

        # Simple Health Score Calculation
        score = 0
        if analysis["ratios"].get("Profit_Margin", 0) > 0.10: score += 1
        if analysis["ratios"].get("ROE", 0) > 0.15: score += 1
        if analysis["trends"].get("Revenue_Growth_YoY", 0) > 0.05: score += 1
        if analysis["ratios"].get("Debt_to_Equity", 100) < 100: score += 1 # debtToEquity is usually %
        
        analysis["health_score"] = score
        return analysis

class TechnicalAnalyst:
    """Analyzes price action and technical indicators."""
    def analyze(self, history: pd.DataFrame) -> Dict[str, Any]:
        if history.empty:
            return {"signal": "NEUTRAL", "details": "No data"}

        # Use existing strategies for analysis
        mac = MACrossoverStrategy(short_window=20, long_window=50)
        rsi_strat = RSIStrategy(rsi_period=14)
        
        mac_signals = mac.generate_signals(history)
        rsi_signals = rsi_strat.generate_signals(history)
        
        latest_mac = mac_signals.iloc[-1]
        latest_rsi = rsi_signals.iloc[-1]
        
        signal_score = 0
        details = []

        # MACD/MA Logic
        if latest_mac['positions'] == 1:
            signal_score += 1
            details.append("Golden Cross (Bullish)")
        elif latest_mac['positions'] == -1:
            signal_score -= 1
            details.append("Death Cross (Bearish)")
            
        # RSI Logic
        current_rsi = latest_rsi['rsi']
        details.append(f"RSI is {current_rsi:.2f}")
        if current_rsi < 30:
            signal_score += 1
            details.append("RSI Oversold (Bullish)")
        elif current_rsi > 70:
            signal_score -= 1
            details.append("RSI Overbought (Bearish)")
            
        # Price Trend
        current_price = history['Close'].iloc[-1]
        ma50 = latest_mac['long_mavg'] # reusing 50 day from MAC strategy
        if current_price > ma50:
            signal_score += 1
            details.append("Price above 50-day MA")
        else:
            signal_score -= 1
            details.append("Price below 50-day MA")

        final_signal = "NEUTRAL"
        if signal_score >= 2: final_signal = "BUY"
        elif signal_score <= -2: final_signal = "SELL"
        
        return {
            "signal": final_signal,
            "score": signal_score,
            "details": details,
            "latest_price": current_price
        }

class ChiefStrategist:
    """Aggregates reports and forms a final recommendation."""
    def generate_report(self, ticker: str, fund_analysis: Dict, tech_analysis: Dict) -> str:
        recommendation = "HOLD"
        
        # Weighing logic
        tech_score = tech_analysis.get("score", 0)
        fund_score = fund_analysis.get("health_score", 0)
        
        # Fundamental score is 0-4, Technical is roughly -3 to 3
        # Let's combine them
        total_score = tech_score + (fund_score - 2) # centering fundamental around 0
        
        if total_score >= 2: recommendation = "STRONG BUY"
        elif total_score >= 1: recommendation = "BUY"
        elif total_score <= -2: recommendation = "STRONG SELL"
        elif total_score <= -1: recommendation = "SELL"
        
        report = f"""
### Executive Summary for {ticker}
**Final Recommendation: {recommendation}**

#### Fundamental Analysis (Health Score: {fund_score}/4)
- **Profitability**: Profit Margin: {fund_analysis['ratios'].get('Profit_Margin', 'N/A'):.2%}, ROE: {fund_analysis['ratios'].get('ROE', 'N/A'):.2%}
- **Valuation**: P/E: {fund_analysis['ratios'].get('PE_Ratio', 'N/A')}, Forward P/E: {fund_analysis['ratios'].get('Forward_PE', 'N/A')}
- **Growth**: Revenue Growth: {fund_analysis['trends'].get('Revenue_Growth_YoY', 0):.2%}

#### Technical Analysis (Signal: {tech_analysis['signal']})
- **Price Action**: Current Price: ${tech_analysis.get('latest_price', 0):.2f}
- **Key Indicators**: {', '.join(tech_analysis['details'])}
        """
        return report

# --- Main App ---

def main():
    st.set_page_config(page_title="Financial Analyst Team", layout="wide")
    
    st.title("🤖 Financial Analyst Team")
    st.markdown("Assemble your AI team to research any company.")

    # Sidebar
    st.sidebar.header("Mission Parameters")
    ticker = st.sidebar.text_input("Target Ticker", value="NVDA").upper()
    
    if st.sidebar.button("Deploy Team"):
        with st.spinner(f"Deploying analysts to research {ticker}..."):
            # 1. Data Collection
            fetcher = DataFetcher(ticker)
            try:
                financials = fetcher.get_financials()
                history = fetcher.get_history()
            except Exception as e:
                st.error(f"Mission Failed: Could not fetch data for {ticker}. Error: {e}")
                return

            # 2. Analysis
            fund_analyst = FundamentalAnalyst()
            tech_analyst = TechnicalAnalyst()
            
            fund_report = fund_analyst.analyze(financials)
            tech_report = tech_analyst.analyze(history)
            
            # 3. Strategy
            chief = ChiefStrategist()
            final_report = chief.generate_report(ticker, fund_report, tech_report)
            
            # --- Display Results ---
            
            # Top Level Verdict
            st.success("Mission Complete. Report Generated.")
            st.markdown(final_report)
            
            # Detailed Breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Fundamental Deep Dive")
                st.json(fund_report)
                
                st.subheader("Financial Statements (Recent)")
                if financials['income_stmt'] is not None:
                    st.write("Income Statement", financials['income_stmt'].head())
                if financials['balance_sheet'] is not None:
                    st.write("Balance Sheet", financials['balance_sheet'].head())

            with col2:
                st.subheader("📈 Technical Deep Dive")
                st.write(f"Technical Signal: **{tech_report['signal']}**")
                st.write("Details:")
                for detail in tech_report['details']:
                    st.write(f"- {detail}")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history.index, y=history['Close'], name='Price'))
                fig.update_layout(title=f"{ticker} Price History", height=400)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
