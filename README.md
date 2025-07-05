# TradingSim

A simple trading simulator for backtesting trading strategies using historical data.

## Features

- Load historical price data (CSV)
- Define and test trading strategies
- Simulate trades and track portfolio performance
- Visualize results with charts

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/yourusername/TradingSim.git
cd TradingSim
pip install -r requirements.txt
```

### Usage

1. Place your historical data CSV in the `data/` folder.
2. Define your strategy in `strategies/`.
3. Run the simulator:

```bash
python main.py --data data/BTCUSD.csv --strategy strategies/simple_ma.py
```

### Example Strategy

```python
# strategies/simple_ma.py
def generate_signals(data):
    data['ma'] = data['Close'].rolling(window=20).mean()
    data['signal'] = 0
    data.loc[data['Close'] > data['ma'], 'signal'] = 1
    data.loc[data['Close'] < data['ma'], 'signal'] = -1
    return data
```

## Project Structure

```
TradingSim/
├── data/
├── strategies/
├── main.py
├── requirements.txt
└── README.md
```

## License

MIT License