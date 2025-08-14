"""
Signal Generator V13.0 - Analysis and Generation Engine
========================================================
PART 2: Technical Analysis, ML Integration, and Signal Generation

This module provides comprehensive market analysis with:
- Advanced technical indicators (RSI, MACD, Stoch, Ichimoku, etc.)
- Volume profile analysis
- ML-based predictions
- News sentiment integration
- Multi-timeframe validation
"""

import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import from Part 1
from signals.signal_gen_v13_core import (
    Signal, SignalStatus, SignalQuality, TimeFrame, MarketRegime,
    SystemConfiguration, SignalCriteria, StateManager,
    AnalysisModule, MarketContext
)

# ===========================
# TECHNICAL INDICATORS MODULE
# ===========================

class TechnicalIndicators:
    """Comprehensive technical indicator calculations"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            # Price-based indicators
            df = TechnicalIndicators._calculate_moving_averages(df)
            df = TechnicalIndicators._calculate_rsi(df)
            df = TechnicalIndicators._calculate_macd(df)
            df = TechnicalIndicators._calculate_stochastic(df)
            df = TechnicalIndicators._calculate_bollinger_bands(df)
            df = TechnicalIndicators._calculate_atr(df)
            df = TechnicalIndicators._calculate_ichimoku(df)
            
            # Volume indicators
            df = TechnicalIndicators._calculate_volume_indicators(df)
            
            # Momentum indicators
            df = TechnicalIndicators._calculate_momentum_indicators(df)
            
            # Volatility indicators
            df = TechnicalIndicators._calculate_volatility_indicators(df)
            
            # Pattern recognition
            df = TechnicalIndicators._calculate_candlestick_patterns(df)
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df
    
    @staticmethod
    def _calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        # Simple Moving Averages
        for period in [10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [9, 12, 21, 26, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Hull Moving Average
        df['hma_9'] = TechnicalIndicators._hull_moving_average(df['close'], 9)
        
        # Volume Weighted Average Price
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI with multiple periods"""
        for p in [14, 21, 28]:
            df[f'rsi_{p}'] = talib.RSI(df['close'], timeperiod=p)
        
        # Default RSI
        df['rsi'] = df['rsi_14']
        
        # RSI divergence detection
        df['rsi_divergence'] = TechnicalIndicators._detect_divergence(
            df['close'], df['rsi'], lookback=20
        )
        
        return df
    
    @staticmethod
    def _calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        # Standard MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Custom MACD for different timeframes
        df['macd_fast'], df['macd_fast_signal'], _ = talib.MACD(
            df['close'], fastperiod=5, slowperiod=13, signalperiod=8
        )
        
        # MACD crossover signals
        df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        return df
    
    @staticmethod
    def _calculate_stochastic(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic oscillators"""
        # Standard Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Stochastic RSI
        df['stoch_rsi_k'], df['stoch_rsi_d'] = talib.STOCHRSI(
            df['close'], timeperiod=14, fastk_period=3, fastd_period=3
        )
        
        # Slow Stochastic
        df['slow_stoch_k'], df['slow_stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=21, slowk_period=5, slowd_period=5
        )
        
        return df
    
    @staticmethod
    def _calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands with multiple deviations"""
        period = 20
        
        for std_dev in [1, 2, 3]:
            upper, middle, lower = talib.BBANDS(
                df['close'], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            df[f'bb_upper_{std_dev}'] = upper
            df[f'bb_lower_{std_dev}'] = lower
        
        df['bb_middle'] = middle
        
        # BB Width and %B
        df['bb_width'] = (df['bb_upper_2'] - df['bb_lower_2']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower_2']) / (df['bb_upper_2'] - df['bb_lower_2'])
        
        # Squeeze detection
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=120).mean() * 0.75
        
        return df
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR and related metrics"""
        # Multiple ATR periods
        for period in [14, 21, 28]:
            df[f'atr_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        df['atr'] = df['atr_14']
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Keltner Channels
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        atr_20 = df['atr_14'].rolling(window=20).mean()
        
        df['kc_upper'] = ema_20 + (2 * atr_20)
        df['kc_lower'] = ema_20 - (2 * atr_20)
        df['kc_middle'] = ema_20
        
        return df
    
    @staticmethod
    def _calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators"""
        # Tenkan-sen (Conversion Line)
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['ichimoku_chikou'] = df['close'].shift(-26)
        
        # Cloud signals
        df['ichimoku_cloud_top'] = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
        df['ichimoku_cloud_bottom'] = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)
        df['ichimoku_cloud_thickness'] = df['ichimoku_cloud_top'] - df['ichimoku_cloud_bottom']
        
        # Position relative to cloud
        df['above_cloud'] = df['close'] > df['ichimoku_cloud_top']
        df['below_cloud'] = df['close'] < df['ichimoku_cloud_bottom']
        df['in_cloud'] = ~(df['above_cloud'] | df['below_cloud'])
        
        return df
    
    @staticmethod
    def _calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        # On-Balance Volume
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volume Moving Averages
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Accumulation/Distribution Line
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Money Flow Index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Chaikin Money Flow
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)
        mf_volume = mfm * df['volume']
        df['cmf'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Volume Rate of Change
        df['vroc'] = ((df['volume'] - df['volume'].shift(10)) / df['volume'].shift(10)) * 100
        
        # Price-Volume Trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        
        return df
    
    @staticmethod
    def _calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # Rate of Change
        for period in [10, 20, 30]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
        
        # Commodity Channel Index
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Ultimate Oscillator
        df['ultimate_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['momentum_pct'] = (df['momentum'] / df['close'].shift(10)) * 100
        
        # TSI (True Strength Index)
        df['tsi'] = TechnicalIndicators._calculate_tsi(df['close'])
        
        return df
    
    @staticmethod
    def _calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        # Historical Volatility
        df['returns'] = df['close'].pct_change()
        df['hvol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['hvol_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        
        # Garman-Klass Volatility
        df['gk_vol'] = TechnicalIndicators._garman_klass_volatility(df)
        
        # Parkinson Volatility
        df['parkinson_vol'] = TechnicalIndicators._parkinson_volatility(df)
        
        # Choppiness Index
        df['chop'] = TechnicalIndicators._choppiness_index(df)
        
        return df
    
    @staticmethod
    def _calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect candlestick patterns"""
        # Bullish patterns
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['bullish_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        
        # Bearish patterns
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['bearish_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) * -1
        df['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        
        # Neutral patterns
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['spinning_top'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
        
        return df
    
    @staticmethod
    def _hull_moving_average(series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = series.rolling(window=half_period).apply(
            lambda x: np.sum(x * np.arange(1, half_period + 1)) / np.sum(np.arange(1, half_period + 1))
        )
        wma_full = series.rolling(window=period).apply(
            lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1))
        )
        
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1))
        )
        
        return hma
    
    @staticmethod
    def _detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 20) -> pd.Series:
        """Detect divergence between price and indicator"""
        divergence = pd.Series(index=price.index, dtype=float)
        
        for i in range(lookback, len(price)):
            price_slice = price.iloc[i-lookback:i]
            indicator_slice = indicator.iloc[i-lookback:i]
            
            # Find peaks and troughs
            price_highs = (price_slice > price_slice.shift(1)) & (price_slice > price_slice.shift(-1))
            price_lows = (price_slice < price_slice.shift(1)) & (price_slice < price_slice.shift(-1))
            
            # Check for divergence
            if price_highs.any() and not indicator_slice[price_highs].empty:
                # Bearish divergence: price makes higher high, indicator makes lower high
                if price_slice[price_highs].iloc[-1] > price_slice[price_highs].iloc[0]:
                    if indicator_slice[price_highs].iloc[-1] < indicator_slice[price_highs].iloc[0]:
                        divergence.iloc[i] = -1  # Bearish
            
            if price_lows.any() and not indicator_slice[price_lows].empty:
                # Bullish divergence: price makes lower low, indicator makes higher low
                if price_slice[price_lows].iloc[-1] < price_slice[price_lows].iloc[0]:
                    if indicator_slice[price_lows].iloc[-1] > indicator_slice[price_lows].iloc[0]:
                        divergence.iloc[i] = 1  # Bullish
        
        return divergence.fillna(0)
    
    @staticmethod
    def _calculate_tsi(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
        """Calculate True Strength Index"""
        momentum = close.diff()
        abs_momentum = momentum.abs()
        
        ema_r = momentum.ewm(span=r, adjust=False).mean()
        ema_r_s = ema_r.ewm(span=s, adjust=False).mean()
        
        abs_ema_r = abs_momentum.ewm(span=r, adjust=False).mean()
        abs_ema_r_s = abs_ema_r.ewm(span=s, adjust=False).mean()
        
        tsi = 100 * (ema_r_s / abs_ema_r_s)
        return tsi
    
    @staticmethod
    def _garman_klass_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility"""
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        
        gk = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=window).mean())
        return gk * np.sqrt(252)
    
    @staticmethod
    def _parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Parkinson volatility"""
        log_hl = np.log(df['high'] / df['low']) ** 2
        parkinson = np.sqrt(log_hl.rolling(window=window).mean() / (4 * np.log(2)))
        return parkinson * np.sqrt(252)
    
    @staticmethod
    def _choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Choppiness Index"""
        high_low = df['high'] - df['low']
        sum_high_low = high_low.rolling(window=period).sum()
        
        highest = df['high'].rolling(window=period).max()
        lowest = df['low'].rolling(window=period).min()
        
        chop = 100 * np.log10(sum_high_low / (highest - lowest)) / np.log10(period)
        return chop

# ===========================
# VOLUME PROFILE ANALYZER
# ===========================

class VolumeProfileAnalyzer:
    """Advanced volume profile analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_volume_profile(self, df: pd.DataFrame, num_bins: int = 50) -> Dict:
        """Create and analyze volume profile"""
        try:
            # Calculate price levels
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, num_bins)
            
            # Volume at each price level
            volume_profile = np.zeros(num_bins - 1)
            
            for idx, row in df.iterrows():
                # Distribute volume across the candle's range
                candle_low = row['low']
                candle_high = row['high']
                candle_volume = row['volume']
                
                # Find bins that this candle spans
                low_bin = np.searchsorted(price_bins, candle_low)
                high_bin = np.searchsorted(price_bins, candle_high)
                
                if low_bin == high_bin:
                    if low_bin > 0 and low_bin < num_bins:
                        volume_profile[low_bin - 1] += candle_volume
                else:
                    # Distribute volume proportionally
                    for bin_idx in range(max(0, low_bin - 1), min(num_bins - 1, high_bin)):
                        volume_profile[bin_idx] += candle_volume / (high_bin - low_bin + 1)
            
            # Find key levels
            poc_idx = np.argmax(volume_profile)  # Point of Control
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Value Area (70% of volume)
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.7
            
            # Expand from POC to find value area
            value_area_low_idx = poc_idx
            value_area_high_idx = poc_idx
            accumulated_volume = volume_profile[poc_idx]
            
            while accumulated_volume < value_area_volume:
                if value_area_low_idx > 0 and value_area_high_idx < num_bins - 2:
                    if volume_profile[value_area_low_idx - 1] > volume_profile[value_area_high_idx + 1]:
                        value_area_low_idx -= 1
                        accumulated_volume += volume_profile[value_area_low_idx]
                    else:
                        value_area_high_idx += 1
                        accumulated_volume += volume_profile[value_area_high_idx]
                elif value_area_low_idx > 0:
                    value_area_low_idx -= 1
                    accumulated_volume += volume_profile[value_area_low_idx]
                elif value_area_high_idx < num_bins - 2:
                    value_area_high_idx += 1
                    accumulated_volume += volume_profile[value_area_high_idx]
                else:
                    break
            
            value_area_high = (price_bins[value_area_high_idx] + price_bins[value_area_high_idx + 1]) / 2
            value_area_low = (price_bins[value_area_low_idx] + price_bins[value_area_low_idx + 1]) / 2
            
            # High Volume Nodes (HVN) and Low Volume Nodes (LVN)
            mean_volume = volume_profile.mean()
            std_volume = volume_profile.std()
            
            hvn_threshold = mean_volume + std_volume
            lvn_threshold = mean_volume - std_volume
            
            hvn_prices = []
            lvn_prices = []
            
            for i, vol in enumerate(volume_profile):
                price = (price_bins[i] + price_bins[i + 1]) / 2
                if vol > hvn_threshold:
                    hvn_prices.append(price)
                elif vol < lvn_threshold and vol > 0:
                    lvn_prices.append(price)
            
            # Current price position
            current_price = df['close'].iloc[-1]
            
            return {
                'poc': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'hvn_levels': hvn_prices[:5],  # Top 5 HVN levels
                'lvn_levels': lvn_prices[:5],  # Top 5 LVN levels
                'current_position': self._get_position_in_profile(
                    current_price, poc_price, value_area_high, value_area_low
                ),
                'volume_distribution': volume_profile,
                'price_bins': price_bins
            }
            
        except Exception as e:
            self.logger.error(f"Volume profile analysis error: {e}")
            return {}
    
    def _get_position_in_profile(self, current_price: float, poc: float, 
                                 vah: float, val: float) -> str:
        """Determine current price position in volume profile"""
        if current_price > vah:
            return 'above_value_area'
        elif current_price < val:
            return 'below_value_area'
        elif abs(current_price - poc) / poc < 0.005:
            return 'at_poc'
        elif current_price > poc:
            return 'above_poc_in_value'
        else:
            return 'below_poc_in_value'

# ===========================
# MARKET STRUCTURE ANALYZER
# ===========================

class MarketStructureAnalyzer:
    """Analyze market structure and identify key levels"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        """Comprehensive market structure analysis"""
        try:
            # Identify swing points
            swing_highs, swing_lows = self._identify_swing_points(df)
            
            # Determine trend structure
            trend_structure = self._analyze_trend_structure(swing_highs, swing_lows)
            
            # Find support and resistance zones
            support_resistance = self._find_support_resistance_zones(df, swing_highs, swing_lows)
            
            # Detect chart patterns
            patterns = self._detect_chart_patterns(df, swing_highs, swing_lows)
            
            # Market regime
            regime = self._determine_market_regime(df, trend_structure)
            
            # Liquidity zones
            liquidity_zones = self._identify_liquidity_zones(df)
            
            return {
                'trend_structure': trend_structure,
                'support_zones': support_resistance['support'],
                'resistance_zones': support_resistance['resistance'],
                'patterns': patterns,
                'market_regime': regime,
                'liquidity_zones': liquidity_zones,
                'swing_highs': swing_highs[-5:],  # Last 5 swing highs
                'swing_lows': swing_lows[-5:]      # Last 5 swing lows
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return {}
    
    def _identify_swing_points(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[List, List]:
        """Identify swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Swing high
            if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
                swing_highs.append({
                    'index': i,
                    'price': df['high'].iloc[i],
                    'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                })
            
            # Swing low
            if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
                swing_lows.append({
                    'index': i,
                    'price': df['low'].iloc[i],
                    'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                })
        
        return swing_highs, swing_lows
    
    def _analyze_trend_structure(self, swing_highs: List, swing_lows: List) -> Dict:
        """Analyze trend based on swing points"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'trend': 'undefined', 'strength': 0}
        
        # Check for higher highs and higher lows (uptrend)
        hh = swing_highs[-1]['price'] > swing_highs[-2]['price']
        hl = swing_lows[-1]['price'] > swing_lows[-2]['price']
        
        # Check for lower highs and lower lows (downtrend)
        lh = swing_highs[-1]['price'] < swing_highs[-2]['price']
        ll = swing_lows[-1]['price'] < swing_lows[-2]['price']
        
        if hh and hl:
            trend = 'uptrend'
            strength = 0.8
        elif lh and ll:
            trend = 'downtrend'
            strength = 0.8
        elif hh and ll:
            trend = 'expanding'
            strength = 0.3
        elif lh and hl:
            trend = 'contracting'
            strength = 0.3
        else:
            trend = 'ranging'
            strength = 0.5
        
        return {
            'trend': trend,
            'strength': strength,
            'last_swing_high': swing_highs[-1]['price'] if swing_highs else None,
            'last_swing_low': swing_lows[-1]['price'] if swing_lows else None
        }
    
    def _find_support_resistance_zones(self, df: pd.DataFrame, 
                                      swing_highs: List, swing_lows: List) -> Dict:
        """Identify support and resistance zones"""
        # Collect all significant levels
        levels = []
        
        # Add swing points
        for sh in swing_highs[-10:]:  # Last 10 swing highs
            levels.append({'price': sh['price'], 'type': 'resistance', 'strength': 1})
        
        for sl in swing_lows[-10:]:  # Last 10 swing lows
            levels.append({'price': sl['price'], 'type': 'support', 'strength': 1})
        
        # Add psychological levels (round numbers)
        current_price = df['close'].iloc[-1]
        round_levels = self._get_round_number_levels(current_price)
        for level in round_levels:
            levels.append({'price': level, 'type': 'psychological', 'strength': 0.5})
        
        # Cluster nearby levels into zones
        zones = self._cluster_levels_into_zones(levels)
        
        # Separate support and resistance
        support_zones = [z for z in zones if z['type'] in ['support', 'psychological'] 
                        and z['price'] < current_price]
        resistance_zones = [z for z in zones if z['type'] in ['resistance', 'psychological'] 
                           and z['price'] > current_price]
        
        # Sort by distance from current price
        support_zones.sort(key=lambda x: current_price - x['price'])
        resistance_zones.sort(key=lambda x: x['price'] - current_price)
        
        return {
            'support': support_zones[:5],     # Nearest 5 support zones
            'resistance': resistance_zones[:5] # Nearest 5 resistance zones
        }
    
    def _detect_chart_patterns(self, df: pd.DataFrame, 
                              swing_highs: List, swing_lows: List) -> List[Dict]:
        """Detect common chart patterns"""
        patterns = []
        
        # Need enough data
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return patterns
        
        # Head and Shoulders
        if self._is_head_and_shoulders(swing_highs[-5:]):
            patterns.append({
                'name': 'head_and_shoulders',
                'type': 'bearish',
                'confidence': 0.7
            })
        
        # Inverse Head and Shoulders
        if self._is_inverse_head_and_shoulders(swing_lows[-5:]):
            patterns.append({
                'name': 'inverse_head_and_shoulders',
                'type': 'bullish',
                'confidence': 0.7
            })
        
        # Double Top
        if self._is_double_top(swing_highs[-4:]):
            patterns.append({
                'name': 'double_top',
                'type': 'bearish',
                'confidence': 0.6
            })
        
        # Double Bottom
        if self._is_double_bottom(swing_lows[-4:]):
            patterns.append({
                'name': 'double_bottom',
                'type': 'bullish',
                'confidence': 0.6
            })
        
        # Triangle patterns
        triangle = self._detect_triangle_pattern(swing_highs[-5:], swing_lows[-5:])
        if triangle:
            patterns.append(triangle)
        
        return patterns
    
    def _determine_market_regime(self, df: pd.DataFrame, trend_structure: Dict) -> MarketRegime:
        """Determine the current market regime"""
        # Calculate volatility
        returns = df['close'].pct_change()
        volatility = returns.tail(20).std()
        
        # Get trend
        trend = trend_structure.get('trend', 'ranging')
        strength = trend_structure.get('strength', 0.5)
        
        # Determine regime
        if volatility > 0.03:  # High volatility threshold
            return MarketRegime.VOLATILE
        elif volatility < 0.005:  # Low volatility (squeeze)
            return MarketRegime.SQUEEZE
        elif trend == 'uptrend':
            if strength > 0.7:
                return MarketRegime.STRONG_TREND_UP
            else:
                return MarketRegime.TREND_UP
        elif trend == 'downtrend':
            if strength > 0.7:
                return MarketRegime.STRONG_TREND_DOWN
            else:
                return MarketRegime.TREND_DOWN
        elif trend == 'ranging':
            # Check bias
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            if df['close'].iloc[-1] > sma_20:
                return MarketRegime.RANGING_BULLISH
            else:
                return MarketRegime.RANGING_BEARISH
        else:
            return MarketRegime.RANGING
    
    def _identify_liquidity_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Identify areas of high liquidity (consolidation zones)"""
        liquidity_zones = []
        
        # Look for consolidation areas
        for i in range(20, len(df) - 20, 5):
            window = df.iloc[i:i+20]
            
            # Calculate range
            high = window['high'].max()
            low = window['low'].min()
            range_pct = (high - low) / low
            
            # Check if it's a consolidation (tight range with high volume)
            if range_pct < 0.05:  # Less than 5% range
                avg_volume = window['volume'].mean()
                if avg_volume > df['volume'].mean():
                    liquidity_zones.append({
                        'high': high,
                        'low': low,
                        'center': (high + low) / 2,
                        'strength': avg_volume / df['volume'].mean(),
                        'start_idx': i,
                        'end_idx': i + 20
                    })
        
        # Sort by strength
        liquidity_zones.sort(key=lambda x: x['strength'], reverse=True)
        
        return liquidity_zones[:5]  # Top 5 liquidity zones
    
    def _get_round_number_levels(self, price: float) -> List[float]:
        """Get psychological round number levels"""
        if price < 1:
            increment = 0.1
        elif price < 10:
            increment = 1
        elif price < 100:
            increment = 10
        elif price < 1000:
            increment = 100
        else:
            increment = 1000
        
        base = (price // increment) * increment
        levels = [
            base - 2 * increment,
            base - increment,
            base,
            base + increment,
            base + 2 * increment
        ]
        
        return [l for l in levels if l > 0]
    
    def _cluster_levels_into_zones(self, levels: List[Dict], threshold: float = 0.005) -> List[Dict]:
        """Cluster nearby levels into zones"""
        if not levels:
            return []
        
        # Sort by price
        levels.sort(key=lambda x: x['price'])
        
        zones = []
        current_zone = [levels[0]]
        
        for level in levels[1:]:
            # Check if level is close to current zone
            zone_center = sum(l['price'] for l in current_zone) / len(current_zone)
            if abs(level['price'] - zone_center) / zone_center < threshold:
                current_zone.append(level)
            else:
                # Create zone from current levels
                zone_price = sum(l['price'] for l in current_zone) / len(current_zone)
                zone_strength = sum(l['strength'] for l in current_zone)
                zone_type = max(current_zone, key=lambda x: x['strength'])['type']
                
                zones.append({
                    'price': zone_price,
                    'type': zone_type,
                    'strength': zone_strength,
                    'touch_count': len(current_zone)
                })
                
                current_zone = [level]
        
        # Add last zone
        if current_zone:
            zone_price = sum(l['price'] for l in current_zone) / len(current_zone)
            zone_strength = sum(l['strength'] for l in current_zone)
            zone_type = max(current_zone, key=lambda x: x['strength'])['type']
            
            zones.append({
                'price': zone_price,
                'type': zone_type,
                'strength': zone_strength,
                'touch_count': len(current_zone)
            })
        
        return zones
    
    def _is_head_and_shoulders(self, highs: List[Dict]) -> bool:
        """Check for head and shoulders pattern"""
        if len(highs) < 3:
            return False
        
        # Left shoulder < Head > Right shoulder
        # And shoulders should be roughly equal
        left_shoulder = highs[-3]['price']
        head = highs[-2]['price']
        right_shoulder = highs[-1]['price']
        
        if head > left_shoulder and head > right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff < 0.02:  # Shoulders within 2% of each other
                return True
        
        return False
    
    def _is_inverse_head_and_shoulders(self, lows: List[Dict]) -> bool:
        """Check for inverse head and shoulders pattern"""
        if len(lows) < 3:
            return False
        
        left_shoulder = lows[-3]['price']
        head = lows[-2]['price']
        right_shoulder = lows[-1]['price']
        
        if head < left_shoulder and head < right_shoulder:
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff < 0.02:
                return True
        
        return False
    
    def _is_double_top(self, highs: List[Dict]) -> bool:
        """Check for double top pattern"""
        if len(highs) < 2:
            return False
        
        first_top = highs[-2]['price']
        second_top = highs[-1]['price']
        
        diff = abs(first_top - second_top) / first_top
        return diff < 0.01  # Tops within 1% of each other
    
    def _is_double_bottom(self, lows: List[Dict]) -> bool:
        """Check for double bottom pattern"""
        if len(lows) < 2:
            return False
        
        first_bottom = lows[-2]['price']
        second_bottom = lows[-1]['price']
        
        diff = abs(first_bottom - second_bottom) / first_bottom
        return diff < 0.01  # Bottoms within 1% of each other
    
    def _detect_triangle_pattern(self, highs: List[Dict], lows: List[Dict]) -> Optional[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Calculate trends of highs and lows
        high_prices = [h['price'] for h in highs]
        low_prices = [l['price'] for l in lows]
        
        high_trend = (high_prices[-1] - high_prices[0]) / high_prices[0]
        low_trend = (low_prices[-1] - low_prices[0]) / low_prices[0]
        
        # Ascending triangle: flat top, rising bottom
        if abs(high_trend) < 0.01 and low_trend > 0.02:
            return {
                'name': 'ascending_triangle',
                'type': 'bullish',
                'confidence': 0.65
            }
        
        # Descending triangle: falling top, flat bottom
        if high_trend < -0.02 and abs(low_trend) < 0.01:
            return {
                'name': 'descending_triangle',
                'type': 'bearish',
                'confidence': 0.65
            }
        
        # Symmetrical triangle: converging lines
        if high_trend < -0.01 and low_trend > 0.01:
            return {
                'name': 'symmetrical_triangle',
                'type': 'neutral',
                'confidence': 0.6
            }
        
        return None

    def detect_breakout(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Detect if price is breaking out of key levels
        Returns detailed breakout information
        """
        try:
            # Get recent price action (last 20-50 candles)
            lookback_short = 20
            lookback_long = 50
            
            # Recent high/low
            recent_high_short = df['high'].tail(lookback_short).max()
            recent_low_short = df['low'].tail(lookback_short).min()
            recent_high_long = df['high'].tail(lookback_long).max()
            recent_low_long = df['low'].tail(lookback_long).min()
            
            # Calculate ATR for dynamic thresholds
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else (recent_high_short - recent_low_short) * 0.1
            
            breakout_info = {
                'has_breakout': False,
                'type': None,
                'strength': 0,
                'levels': {
                    'resistance_short': recent_high_short,
                    'support_short': recent_low_short,
                    'resistance_long': recent_high_long,
                    'support_long': recent_low_long
                },
                'volume_confirmed': False,
                'pattern_confirmed': False
            }
            
            # Check for resistance breakout
            if current_price > recent_high_short * 0.998:  # Within 0.2% or breaking
                breakout_strength = (current_price - recent_high_short) / recent_high_short
                breakout_info.update({
                    'has_breakout': True,
                    'type': 'resistance_break',
                    'strength': breakout_strength,
                    'level_broken': recent_high_short
                })
                
                # Check if also breaking longer-term resistance
                if current_price > recent_high_long * 0.998:
                    breakout_info['strength'] *= 1.5  # Stronger signal
                    breakout_info['type'] = 'major_resistance_break'
            
            # Check for support breakdown
            elif current_price < recent_low_short * 1.002:  # Within 0.2% or breaking
                breakdown_strength = (recent_low_short - current_price) / recent_low_short
                breakout_info.update({
                    'has_breakout': True,
                    'type': 'support_break',
                    'strength': breakdown_strength,
                    'level_broken': recent_low_short
                })
                
                # Check if also breaking longer-term support
                if current_price < recent_low_long * 1.002:
                    breakout_info['strength'] *= 1.5
                    breakout_info['type'] = 'major_support_break'
            
            # Volume confirmation
            if breakout_info['has_breakout']:
                volume_avg = df['volume'].tail(20).mean()
                current_volume = df['volume'].iloc[-1]
                volume_spike = df['volume'].iloc[-3:].mean()  # Last 3 candles
                
                if current_volume > volume_avg * 1.5 or volume_spike > volume_avg * 1.3:
                    breakout_info['volume_confirmed'] = True
                    breakout_info['strength'] *= 1.3
                    breakout_info['volume_ratio'] = current_volume / volume_avg
            
            # Pattern confirmation (check if breakout aligns with patterns)
            if breakout_info['has_breakout']:
                # Check for consolidation before breakout
                recent_range = recent_high_short - recent_low_short
                longer_range = recent_high_long - recent_low_long
                
                if recent_range < longer_range * 0.5:  # Consolidation detected
                    breakout_info['pattern_confirmed'] = True
                    breakout_info['pattern'] = 'consolidation_breakout'
                    breakout_info['strength'] *= 1.2
            
            return breakout_info
            
        except Exception as e:
            self.logger.error(f"Breakout detection error: {e}")
            return {'has_breakout': False, 'type': None, 'strength': 0}

# ===========================
# EXPORTS
# ===========================

__all__ = [
    'TechnicalIndicators',
    'VolumeProfileAnalyzer',
    'MarketStructureAnalyzer'
]