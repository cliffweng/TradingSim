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

# Price Momentum Strategy
class PriceMomentumStrategy(TradingStrategy):
    def __init__(self, momentum_window: int = 10):
        self.momentum_window = momentum_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate momentum (difference between current close and close n periods ago)
        signals['momentum'] = data['Close'] - data['Close'].shift(self.momentum_window)
        # Generate signals: Buy if momentum > 0, Sell if momentum < 0
        signals['signal'] = 0
        signals.loc[signals.index[self.momentum_window:], 'signal'] = np.where(
            signals['momentum'][self.momentum_window:] > 0, 1,
            np.where(signals['momentum'][self.momentum_window:] < 0, 0, np.nan)
        )
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals


class MACDStrategy(TradingStrategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate MACD
        exp1 = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.slow_period, adjust=False).mean()
        signals['macd'] = exp1 - exp2
        signals['signal_line'] = signals['macd'].ewm(span=self.signal_period, adjust=False).mean()
        signals['histogram'] = signals['macd'] - signals['signal_line']
        # Generate signals: Buy when MACD crosses above signal line, sell when it crosses below
        signals['signal'] = 0
        signals.loc[signals.index[self.slow_period:], 'signal'] = np.where(
            signals['macd'][self.slow_period:] > signals['signal_line'][self.slow_period:], 1,
            np.where(signals['macd'][self.slow_period:] < signals['signal_line'][self.slow_period:], 0, np.nan)
        )
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals


class MeanReversionStrategy(TradingStrategy):
    def __init__(self, window: int = 20, z_threshold: float = 2.0):
        self.window = window
        self.z_threshold = z_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate z-score
        rolling_mean = data['Close'].rolling(window=self.window).mean()
        rolling_std = data['Close'].rolling(window=self.window).std()
        signals['z_score'] = (data['Close'] - rolling_mean) / rolling_std
        # Generate signals: Buy when price is far below mean (oversold), sell when far above (overbought)
        signals['signal'] = 0
        signals.loc[signals.index[self.window:], 'signal'] = np.where(
            signals['z_score'][self.window:] < -self.z_threshold, 1,
            np.where(signals['z_score'][self.window:] > self.z_threshold, 0, np.nan)
        )
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals


class DonchianChannelStrategy(TradingStrategy):
    def __init__(self, channel_period: int = 20):
        self.channel_period = channel_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate Donchian Channels
        signals['upper_channel'] = data['High'].rolling(window=self.channel_period).max()
        signals['lower_channel'] = data['Low'].rolling(window=self.channel_period).min()
        signals['middle_channel'] = (signals['upper_channel'] + signals['lower_channel']) / 2
        # Generate signals: Buy on breakout above upper channel, sell on breakdown below lower channel
        signals['signal'] = 0
        signals.loc[signals.index[self.channel_period:], 'signal'] = np.where(
            data['Close'][self.channel_period:] > signals['upper_channel'][self.channel_period:], 1,
            np.where(data['Close'][self.channel_period:] < signals['lower_channel'][self.channel_period:], 0, np.nan)
        )
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals


class StochasticOscillatorStrategy(TradingStrategy):
    def __init__(self, k_period: int = 14, d_period: int = 3, overbought: float = 80, oversold: float = 20):
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate Stochastic Oscillator
        low_min = data['Low'].rolling(window=self.k_period).min()
        high_max = data['High'].rolling(window=self.k_period).max()
        signals['k'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
        signals['d'] = signals['k'].rolling(window=self.d_period).mean()
        # Generate signals: Buy when %K crosses above %D in oversold, sell when crosses below in overbought
        signals['signal'] = 0
        signals.loc[signals.index[self.k_period:], 'signal'] = np.where(
            (signals['k'][self.k_period:] < self.oversold) & (signals['k'][self.k_period:] > signals['d'][self.k_period:]), 1,
            np.where((signals['k'][self.k_period:] > self.overbought) & (signals['k'][self.k_period:] < signals['d'][self.k_period:]), 0, np.nan)
        )
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals


class ATRStrategy(TradingStrategy):
    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate True Range and ATR
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        signals['atr'] = tr.rolling(window=self.atr_period).mean()
        # Calculate trailing stop based on ATR
        signals['trailing_stop'] = data['Close'] - (signals['atr'] * self.atr_multiplier)
        # Generate signals: Buy when price crosses above ATR-based stop, sell when price falls below
        signals['signal'] = 0
        shifted_stop = signals['trailing_stop'].shift(1)
        for i in range(self.atr_period, len(signals)):
            idx = signals.index[i]
            if data.loc[idx, 'Close'] > shifted_stop.loc[idx]:
                signals.loc[idx, 'signal'] = 1
            elif data.loc[idx, 'Close'] < shifted_stop.loc[idx]:
                signals.loc[idx, 'signal'] = 0
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals


class DualThrustStrategy(TradingStrategy):
    def __init__(self, lookback_period: int = 20, k_value: float = 0.5):
        self.lookback_period = lookback_period
        self.k_value = k_value

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        # Calculate Dual Thrust levels
        hh = data['High'].rolling(window=self.lookback_period).max()
        ll = data['Low'].rolling(window=self.lookback_period).min()
        hc = data['Close'].shift().rolling(window=self.lookback_period).max()
        lc = data['Close'].shift().rolling(window=self.lookback_period).min()
        range_val = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
        signals['upper_level'] = data['Close'].shift() + (self.k_value * range_val)
        signals['lower_level'] = data['Close'].shift() - (self.k_value * range_val)
        # Generate signals: Buy when price breaks above upper level, sell when breaks below lower level
        signals['signal'] = 0
        signals.loc[signals.index[self.lookback_period:], 'signal'] = np.where(
            data['Close'][self.lookback_period:] > signals['upper_level'][self.lookback_period:], 1,
            np.where(data['Close'][self.lookback_period:] < signals['lower_level'][self.lookback_period:], 0, np.nan)
        )
        signals['signal'] = signals['signal'].ffill()
        signals['signal'] = signals['signal'].fillna(0)
        signals['positions'] = signals['signal'].diff()
        return signals
