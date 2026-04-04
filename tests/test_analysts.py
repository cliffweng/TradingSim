import sys
import os
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_analyst_team import DataFetcher, FundamentalAnalyst, TechnicalAnalyst, ChiefStrategist

def test_analysts():
    print("Testing DataFetcher...")
    fetcher = DataFetcher("AAPL")
    financials = fetcher.get_financials()
    history = fetcher.get_history(period="1mo")
    
    assert financials['info'] is not None, "Info should not be None"
    assert not history.empty, "History should not be empty"
    print("DataFetcher OK")

    print("Testing FundamentalAnalyst...")
    fund_analyst = FundamentalAnalyst()
    fund_report = fund_analyst.analyze(financials)
    print("Fundamental Report Keys:", fund_report.keys())
    assert "health_score" in fund_report, "Report should have health_score"
    print("FundamentalAnalyst OK")

    print("Testing TechnicalAnalyst...")
    tech_analyst = TechnicalAnalyst()
    tech_report = tech_analyst.analyze(history)
    print("Technical Report:", tech_report)
    assert "signal" in tech_report, "Report should have signal"
    print("TechnicalAnalyst OK")

    print("Testing ChiefStrategist...")
    chief = ChiefStrategist()
    final_report = chief.generate_report("AAPL", fund_report, tech_report)
    assert len(final_report) > 0, "Final report should not be empty"
    print("ChiefStrategist OK")
    
    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_analysts()
