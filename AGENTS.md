# AGENTS.md - Agent Coding Guidelines

## Project Overview

This is a Python-based trading simulation application built with Streamlit. It includes trading strategy implementations, data visualization with Plotly, and financial analysis tools.

## Build & Run Commands

```bash
# Main trading simulator
streamlit run trading_sim.py

# Sector rotation strategy
streamlit run sector_rotation.py

# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_analysts.py -v

# Run a single test function
pytest tests/test_analysts.py::test_analysts -v

# Run with specific test pattern
pytest tests/ -k "test_analysts" -v
```

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party, then local
- Group imports by type (stdlib, third-party, local) with blank lines between groups
- Use absolute imports from package root when possible

```python
# Correct order
import os
import sys
from datetime import datetime, timedelta
from functools import lru_cache

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
from typing import List

from strategies import TradingStrategy, MACrossoverStrategy
```

### Formatting
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Use blank lines to separate logical sections within functions
- Two blank lines between top-level definitions (classes, functions)

### Type Hints
- Use type hints for function parameters and return values
- Use `pd.DataFrame` for strategy data, `List` for collections
- Use `datetime` for date/time objects

### Naming Conventions
- **Classes**: PascalCase (`TradingStrategy`, `MACrossoverStrategy`)
- **Functions/methods**: snake_case (`generate_signals`, `calculate_trade_metrics`)
- **Variables**: snake_case (`short_window`, `starting_cash`)
- **Constants**: UPPER_SNAKE_CASE (if needed)
- **Dataclasses**: PascalCase with `@dataclass` decorator

### Classes & Functions
- Keep classes focused on single responsibility
- Use abstract base class `TradingStrategy` for strategy pattern
- Use `@dataclass` for simple data containers (`TradeRecord`)
- Use `@lru_cache` for expensive caching (data fetching)

### Error Handling
- Wrap external API calls (yfinance) in try/except blocks
- Provide user-friendly error messages in Streamlit apps
- Log errors appropriately for debugging

### DataFrame Conventions
- Always use meaningful column names
- Return standardized DataFrames from strategies with columns: `signal`, `positions`, `price`
- Use `.loc[]` and `.iloc[]` for DataFrame access (avoid chained assignment)
- Handle NaN values explicitly with `.fillna()` or `.ffill()`

### Streamlit Best Practices
- Use `@st.cache_data` or `@lru_cache` for expensive data fetching
- Set page config at top: `st.set_page_config(layout="wide")`
- Use sidebar for user inputs: `st.sidebar.header()`, `st.sidebar.text_input()`
- Use `st.spinner()` for async operations

### Testing
- Tests reside in `tests/` directory
- Use `pytest` as test runner
- Add parent directory to path: `sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
- Use assertions with descriptive messages

### Project Structure
```
TradingSim/
├── trading_sim.py      # Main Streamlit app
├── sector_rotation.py  # Sector rotation strategy
├── strategies.py       # Trading strategy implementations
├── financial_analyst_team.py
├── tests/
│   └── test_analysts.py
└── yf_cache/           # Data cache directory
```

### Key Patterns

1. **Strategy Pattern**: All strategies inherit from `TradingStrategy` base class
2. **Data Caching**: Use `@lru_cache` decorator for performance
3. **Dataclasses**: Use `@dataclass` for structured data (TradeRecord)

### Dependencies
- streamlit
- yfinance
- pandas
- numpy
- plotly
- pytest (for testing)