import numpy as np
from abc import ABC, abstractmethod
import pandas as pd

# Abstract base class for trading strategies
class TradingStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

# Moving Average Crossover Strategy
class MACrossoverStrategy(TradingStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate moving averages
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window).mean()
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals.index[self.short_window:], 'signal'] = \
            np.where(signals['short_mavg'][self.short_window:] > signals['long_mavg'][self.short_window:], 1, 0)
        # Generate buy/sell positions
        signals['positions'] = signals['signal'].diff()
        return signals

# RSI Strategy
class RSIStrategy(TradingStrategy):
    def __init__(self, rsi_period: int = 14, overbought: float = 70, oversold: float = 30):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        signals['rsi'] = 100 - (100 / (1 + rs))
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals.index[self.rsi_period:], 'signal'] = \
            np.where(signals['rsi'][self.rsi_period:] < self.oversold, 1, 
                     np.where(signals['rsi'][self.rsi_period:] > self.overbought, 0, np.nan))
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        # Generate buy/sell positions
        signals['positions'] = signals['signal'].diff()
        return signals

# Bollinger Bands Breakout Strategy
class BollingerBandsStrategy(TradingStrategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate Bollinger Bands
        rolling_mean = data['Close'].rolling(window=self.window).mean()
        rolling_std = data['Close'].rolling(window=self.window).std()
        signals['upper_band'] = rolling_mean + (rolling_std * self.num_std)
        signals['lower_band'] = rolling_mean - (rolling_std * self.num_std)
        # Generate signals: Buy when price crosses above lower band, sell when price crosses below upper band
        signals['signal'] = 0
        signals.loc[signals.index[self.window:], 'signal'] = np.where(
            data['Close'][self.window:] < signals['lower_band'][self.window:], 1,
            np.where(data['Close'][self.window:] > signals['upper_band'][self.window:], 0, np.nan)
        )
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals
