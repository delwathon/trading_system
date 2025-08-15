"""
Signal Analyzer V14.0 - Technical Analysis and Market Intelligence
==================================================================
Provides technical analysis, market structure, and volume profiling.
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import json
from collections import defaultdict
from scipy import stats

# ===========================
# SIGNAL ANALYZER CLASS
# ===========================

class SignalAnalyzer:
    """
    Comprehensive market analyzer providing technical analysis
    and structure detection.
    """
    
    def __init__(self, config=None):
        """Initialize the analyzer"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Technical analysis settings
        self.swing_lookback = 5
        self.support_resistance_lookback = 50
        
    # ===========================
    # TECHNICAL INDICATORS
    # ===========================
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.
        Optimized for signal generation.
        """
        try:
            # Ensure we have enough data
            if len(df) < 50:
                self.logger.warning("Insufficient data for indicators")
                return df
            
            # Price-based indicators
            df = self._calculate_moving_averages(df)
            df = self._calculate_oscillators(df)
            df = self._calculate_volatility_indicators(df)
            df = self._calculate_volume_indicators(df)
            df = self._calculate_momentum_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error: {e}")
            return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        # Simple Moving Averages
        for period in [10, 20, 50, 100, 200]:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [9, 12, 20, 26, 50]:
            if len(df) >= period:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Hull Moving Average (9-period for responsiveness)
        if len(df) >= 9:
            df['hma_9'] = self._hull_moving_average(df['close'], 9)
        
        return df
    
    def _calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate oscillator indicators"""
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_fast'] = talib.RSI(df['close'], timeperiod=7)
        df['rsi_slow'] = talib.RSI(df['close'], timeperiod=21)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Stochastic RSI
        df['stoch_rsi_k'], df['stoch_rsi_d'] = talib.STOCHRSI(
            df['close'], timeperiod=14, fastk_period=3, fastd_period=3
        )
        
        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        # ATR
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channels
        if 'ema_20' in df.columns:
            kc_multiplier = 2
            df['kc_upper'] = df['ema_20'] + (df['atr'] * kc_multiplier)
            df['kc_lower'] = df['ema_20'] - (df['atr'] * kc_multiplier)
            
            # Squeeze indicator (BB inside KC)
            df['squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        
        # Historical Volatility
        df['returns'] = df['close'].pct_change()
        df['hvol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        # Volume moving averages
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # On-Balance Volume
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Money Flow Index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Accumulation/Distribution Line
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)
        mf_volume = mfm * df['volume']
        df['cmf'] = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Volume Rate of Change
        df['vroc'] = ((df['volume'] - df['volume'].shift(10)) / df['volume'].shift(10)) * 100
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # Rate of Change
        df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
        df['roc_20'] = talib.ROC(df['close'], timeperiod=20)
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['momentum_pct'] = (df['momentum'] / df['close'].shift(10)) * 100
        
        # Ultimate Oscillator
        df['ultimate_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
        
        # TSI (True Strength Index) - simplified version
        df['tsi'] = self._calculate_tsi(df['close'])
        
        return df
    
    def _hull_moving_average(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average for better responsiveness"""
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
    
    def _calculate_tsi(self, close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
        """Calculate True Strength Index"""
        momentum = close.diff()
        abs_momentum = momentum.abs()
        
        ema_r = momentum.ewm(span=r, adjust=False).mean()
        ema_r_s = ema_r.ewm(span=s, adjust=False).mean()
        
        abs_ema_r = abs_momentum.ewm(span=r, adjust=False).mean()
        abs_ema_r_s = abs_ema_r.ewm(span=s, adjust=False).mean()
        
        tsi = 100 * (ema_r_s / abs_ema_r_s)
        return tsi
    
    # ===========================
    # MARKET STRUCTURE ANALYSIS
    # ===========================
    
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Analyze market structure including trend, support/resistance,
        and key price levels.
        """
        try:
            # Identify swing points
            swing_highs, swing_lows = self._identify_swing_points(df)
            
            # Determine trend
            trend = self._determine_trend(df, swing_highs, swing_lows)
            
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance(
                df, swing_highs, swing_lows
            )
            
            # Check proximity to levels
            current_price = df['close'].iloc[-1]
            near_support = self._check_near_level(current_price, support_levels, 0.01)
            near_resistance = self._check_near_level(current_price, resistance_levels, 0.01)
            
            # Calculate volatility regime
            volatility = self._calculate_volatility_regime(df)
            
            return {
                'trend': trend,
                'support_levels': support_levels[:3],  # Top 3 support levels
                'resistance_levels': resistance_levels[:3],  # Top 3 resistance levels
                'near_support': near_support,
                'near_resistance': near_resistance,
                'swing_highs': swing_highs[-3:] if swing_highs else [],
                'swing_lows': swing_lows[-3:] if swing_lows else [],
                'volatility': volatility,
                'range_bound': self._is_range_bound(df),
                'trend_strength': self._calculate_trend_strength(df)
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return {}
    
    def _identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            # Swing high
            if df['high'].iloc[i] == df['high'].iloc[i-self.swing_lookback:i+self.swing_lookback+1].max():
                swing_highs.append(df['high'].iloc[i])
            
            # Swing low
            if df['low'].iloc[i] == df['low'].iloc[i-self.swing_lookback:i+self.swing_lookback+1].min():
                swing_lows.append(df['low'].iloc[i])
        
        return swing_highs, swing_lows
    
    def _determine_trend(self, df: pd.DataFrame, swing_highs: List, swing_lows: List) -> str:
        """Determine market trend based on price action and moving averages"""
        # Moving average trend
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma20 = df['sma_20'].iloc[-1]
            sma50 = df['sma_50'].iloc[-1]
            current = df['close'].iloc[-1]
            
            ma_trend = 'neutral'
            if current > sma20 > sma50:
                ma_trend = 'uptrend'
            elif current < sma20 < sma50:
                ma_trend = 'downtrend'
        else:
            ma_trend = 'neutral'
        
        # Swing point trend
        swing_trend = 'neutral'
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Higher highs and higher lows = uptrend
            if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
                swing_trend = 'uptrend'
            # Lower highs and lower lows = downtrend
            elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
                swing_trend = 'downtrend'
        
        # Combine trends
        if ma_trend == swing_trend:
            if ma_trend == 'uptrend':
                # Check strength
                if df['close'].iloc[-1] > df['close'].iloc[-20] * 1.05:
                    return 'strong_uptrend'
                return 'uptrend'
            elif ma_trend == 'downtrend':
                if df['close'].iloc[-1] < df['close'].iloc[-20] * 0.95:
                    return 'strong_downtrend'
                return 'downtrend'
        
        # Check for ranging with bias
        close_20_ago = df['close'].iloc[-20] if len(df) >= 20 else df['close'].iloc[0]
        change_pct = (df['close'].iloc[-1] - close_20_ago) / close_20_ago
        
        if abs(change_pct) < 0.03:  # Less than 3% change
            if df['close'].iloc[-1] > df['sma_50'].iloc[-1] if 'sma_50' in df.columns else True:
                return 'ranging_bullish'
            else:
                return 'ranging_bearish'
        
        return ma_trend if ma_trend != 'neutral' else 'ranging'
    
    def _find_support_resistance(self, df: pd.DataFrame, swing_highs: List, 
                                swing_lows: List) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        current_price = df['close'].iloc[-1]
        
        # Support levels (below current price)
        support_candidates = []
        
        # Add swing lows
        support_candidates.extend(swing_lows)
        
        # Add recent lows
        if len(df) >= self.support_resistance_lookback:
            recent_lows = df['low'].tail(self.support_resistance_lookback).nsmallest(5).tolist()
            support_candidates.extend(recent_lows)
        
        # Add psychological levels
        psychological_levels = self._get_psychological_levels(current_price)
        support_candidates.extend([p for p in psychological_levels if p < current_price])
        
        # Filter and sort
        support_levels = sorted(list(set([s for s in support_candidates if s < current_price])), reverse=True)
        
        # Resistance levels (above current price)
        resistance_candidates = []
        
        # Add swing highs
        resistance_candidates.extend(swing_highs)
        
        # Add recent highs
        if len(df) >= self.support_resistance_lookback:
            recent_highs = df['high'].tail(self.support_resistance_lookback).nlargest(5).tolist()
            resistance_candidates.extend(recent_highs)
        
        # Add psychological levels
        resistance_candidates.extend([p for p in psychological_levels if p > current_price])
        
        # Filter and sort
        resistance_levels = sorted(list(set([r for r in resistance_candidates if r > current_price])))
        
        return support_levels, resistance_levels
    
    def _get_psychological_levels(self, price: float) -> List[float]:
        """Get psychological round number levels"""
        if price < 1:
            increment = 0.1
        elif price < 10:
            increment = 1
        elif price < 100:
            increment = 10
        elif price < 1000:
            increment = 100
        elif price < 10000:
            increment = 1000
        else:
            increment = 5000
        
        base = (price // increment) * increment
        levels = [
            base - 2 * increment,
            base - increment,
            base,
            base + increment,
            base + 2 * increment
        ]
        
        return [l for l in levels if l > 0]
    
    def _check_near_level(self, price: float, levels: List[float], threshold: float) -> bool:
        """Check if price is near any level"""
        for level in levels:
            if abs(price - level) / level <= threshold:
                return True
        return False
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine volatility regime"""
        if 'atr_percent' not in df.columns:
            return 'normal'
        
        atr_pct = df['atr_percent'].iloc[-1]
        avg_atr = df['atr_percent'].tail(20).mean()
        
        if atr_pct > avg_atr * 1.5:
            return 'high'
        elif atr_pct < avg_atr * 0.5:
            return 'low'
        else:
            return 'normal'
    
    def _is_range_bound(self, df: pd.DataFrame) -> bool:
        """Check if market is range-bound"""
        if len(df) < 20:
            return False
        
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        range_pct = (recent_high - recent_low) / recent_low
        
        return range_pct < 0.05  # Less than 5% range
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-1)"""
        if len(df) < 20:
            return 0.5
        
        # ADX would be ideal here, but we'll use a simplified version
        close_changes = df['close'].pct_change().tail(20)
        
        # Count directional days
        up_days = (close_changes > 0).sum()
        down_days = (close_changes < 0).sum()
        
        # Calculate strength
        if up_days > down_days:
            strength = up_days / 20
        else:
            strength = down_days / 20
        
        return min(1.0, strength)
    
    # ===========================
    # VOLUME PROFILE ANALYSIS
    # ===========================
    
    def analyze_volume_profile(self, df: pd.DataFrame, num_bins: int = 30) -> Dict:
        """
        Analyze volume profile to identify high volume nodes (HVN),
        low volume nodes (LVN), and point of control (POC).
        """
        try:
            if len(df) < 20:
                return {}
            
            # Calculate price levels
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, num_bins)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(num_bins - 1)
            
            for idx, row in df.iterrows():
                candle_low = row['low']
                candle_high = row['high']
                candle_volume = row['volume']
                
                # Find bins that this candle spans
                low_bin = np.searchsorted(price_bins, candle_low)
                high_bin = np.searchsorted(price_bins, candle_high)
                
                if low_bin == high_bin:
                    if 0 < low_bin < num_bins:
                        volume_profile[low_bin - 1] += candle_volume
                else:
                    # Distribute volume proportionally
                    for bin_idx in range(max(0, low_bin - 1), min(num_bins - 1, high_bin)):
                        volume_profile[bin_idx] += candle_volume / max(1, high_bin - low_bin + 1)
            
            # Find Point of Control (highest volume price)
            poc_idx = np.argmax(volume_profile)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Calculate Value Area (70% of volume)
            total_volume = volume_profile.sum()
            value_area_volume = total_volume * 0.7
            
            # Expand from POC to find value area
            value_area_low_idx = poc_idx
            value_area_high_idx = poc_idx
            accumulated_volume = volume_profile[poc_idx]
            
            while accumulated_volume < value_area_volume:
                expand_low = expand_high = False
                
                if value_area_low_idx > 0 and value_area_high_idx < num_bins - 2:
                    if volume_profile[value_area_low_idx - 1] > volume_profile[value_area_high_idx + 1]:
                        expand_low = True
                    else:
                        expand_high = True
                elif value_area_low_idx > 0:
                    expand_low = True
                elif value_area_high_idx < num_bins - 2:
                    expand_high = True
                else:
                    break
                
                if expand_low:
                    value_area_low_idx -= 1
                    accumulated_volume += volume_profile[value_area_low_idx]
                elif expand_high:
                    value_area_high_idx += 1
                    accumulated_volume += volume_profile[value_area_high_idx]
            
            value_area_high = (price_bins[value_area_high_idx] + price_bins[value_area_high_idx + 1]) / 2
            value_area_low = (price_bins[value_area_low_idx] + price_bins[value_area_low_idx + 1]) / 2
            
            # Identify HVN and LVN
            mean_volume = volume_profile.mean()
            std_volume = volume_profile.std()
            
            hvn_threshold = mean_volume + std_volume
            lvn_threshold = mean_volume - std_volume
            
            hvn_levels = []
            lvn_levels = []
            
            for i, vol in enumerate(volume_profile):
                if vol > hvn_threshold:
                    price = (price_bins[i] + price_bins[i + 1]) / 2
                    hvn_levels.append(price)
                elif 0 < vol < lvn_threshold:
                    price = (price_bins[i] + price_bins[i + 1]) / 2
                    lvn_levels.append(price)
            
            # Current price position
            current_price = df['close'].iloc[-1]
            
            return {
                'poc': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'hvn_levels': hvn_levels[:5],  # Top 5 HVN
                'lvn_levels': lvn_levels[:5],  # Top 5 LVN
                'current_position': self._get_position_in_profile(
                    current_price, poc_price, value_area_high, value_area_low
                ),
                'volume_balance': self._calculate_volume_balance(df)
            }
            
        except Exception as e:
            self.logger.error(f"Volume profile analysis error: {e}")
            return {}
    
    def _get_position_in_profile(self, current: float, poc: float, vah: float, val: float) -> str:
        """Determine current price position in volume profile"""
        if current > vah:
            return 'above_value'
        elif current < val:
            return 'below_value'
        elif abs(current - poc) / poc < 0.005:
            return 'at_poc'
        elif current > poc:
            return 'above_poc'
        else:
            return 'below_poc'
    
    def _calculate_volume_balance(self, df: pd.DataFrame) -> str:
        """Calculate buying vs selling volume balance"""
        if len(df) < 10:
            return 'neutral'
        
        # Simple approximation: up candles = buying, down candles = selling
        recent = df.tail(10)
        buying_volume = recent[recent['close'] > recent['open']]['volume'].sum()
        selling_volume = recent[recent['close'] <= recent['open']]['volume'].sum()
        
        total = buying_volume + selling_volume
        if total == 0:
            return 'neutral'
        
        buy_ratio = buying_volume / total
        
        if buy_ratio > 0.6:
            return 'buying_pressure'
        elif buy_ratio < 0.4:
            return 'selling_pressure'
        else:
            return 'neutral'
    
    # ===========================
    # PATTERN DETECTION
    # ===========================
    
    def detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect chart patterns and candlestick patterns"""
        patterns = []
        
        try:
            # Chart patterns
            if self._detect_double_bottom(df):
                patterns.append('double_bottom')
            if self._detect_double_top(df):
                patterns.append('double_top')
            if self._detect_ascending_triangle(df):
                patterns.append('ascending_triangle')
            if self._detect_descending_triangle(df):
                patterns.append('descending_triangle')
            if self._detect_flag_pattern(df):
                patterns.append('flag_pattern')
            
            # Candlestick patterns (using TA-Lib)
            if len(df) >= 5:
                latest_idx = -1
                
                # Bullish patterns
                if talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']).iloc[latest_idx] > 0:
                    patterns.append('hammer')
                if talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[latest_idx] > 0:
                    patterns.append('morning_star')
                if talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[latest_idx] > 0:
                    patterns.append('bullish_engulfing')
                
                # Bearish patterns
                if talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[latest_idx] > 0:
                    patterns.append('shooting_star')
                if talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close']).iloc[latest_idx] > 0:
                    patterns.append('evening_star')
                if talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[latest_idx] < 0:
                    patterns.append('bearish_engulfing')
                
                # Neutral patterns
                if talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']).iloc[latest_idx] != 0:
                    patterns.append('doji')
            
        except Exception as e:
            self.logger.debug(f"Pattern detection error: {e}")
        
        return patterns
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> bool:
        """Detect double bottom pattern"""
        if len(df) < 20:
            return False
        
        recent = df.tail(20)
        lows = recent['low'].values
        
        # Find two prominent lows
        min_indices = []
        for i in range(2, len(lows) - 2):
            if lows[i] == min(lows[i-2:i+3]):
                min_indices.append(i)
        
        if len(min_indices) >= 2:
            # Check if lows are similar (within 1%)
            low1 = lows[min_indices[-2]]
            low2 = lows[min_indices[-1]]
            if abs(low1 - low2) / low1 < 0.01:
                # Check for rise after second bottom
                if recent['close'].iloc[-1] > max(low1, low2) * 1.02:
                    return True
        
        return False
    
    def _detect_double_top(self, df: pd.DataFrame) -> bool:
        """Detect double top pattern"""
        if len(df) < 20:
            return False
        
        recent = df.tail(20)
        highs = recent['high'].values
        
        # Find two prominent highs
        max_indices = []
        for i in range(2, len(highs) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                max_indices.append(i)
        
        if len(max_indices) >= 2:
            # Check if highs are similar (within 1%)
            high1 = highs[max_indices[-2]]
            high2 = highs[max_indices[-1]]
            if abs(high1 - high2) / high1 < 0.01:
                # Check for decline after second top
                if recent['close'].iloc[-1] < min(high1, high2) * 0.98:
                    return True
        
        return False
    
    def _detect_ascending_triangle(self, df: pd.DataFrame) -> bool:
        """Detect ascending triangle pattern"""
        if len(df) < 15:
            return False
        
        recent = df.tail(15)
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Check for flat top (resistance)
        high_std = np.std(highs[-5:])
        high_mean = np.mean(highs[-5:])
        
        # Check for rising lows
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if high_std / high_mean < 0.01 and low_trend > 0:
            return True
        
        return False
    
    def _detect_descending_triangle(self, df: pd.DataFrame) -> bool:
        """Detect descending triangle pattern"""
        if len(df) < 15:
            return False
        
        recent = df.tail(15)
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Check for flat bottom (support)
        low_std = np.std(lows[-5:])
        low_mean = np.mean(lows[-5:])
        
        # Check for declining highs
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        
        if low_std / low_mean < 0.01 and high_trend < 0:
            return True
        
        return False
    
    def _detect_flag_pattern(self, df: pd.DataFrame) -> bool:
        """Detect flag pattern (consolidation after strong move)"""
        if len(df) < 20:
            return False
        
        # Check for strong move in previous 10 candles
        move_period = df.iloc[-20:-10]
        consolidation = df.iloc[-10:]
        
        move_change = (move_period['close'].iloc[-1] - move_period['close'].iloc[0]) / move_period['close'].iloc[0]
        consolidation_range = (consolidation['high'].max() - consolidation['low'].min()) / consolidation['low'].min()
        
        # Strong move followed by tight consolidation
        if abs(move_change) > 0.05 and consolidation_range < 0.02:
            return True
        
        return False
    
    # ===========================
    # BREAKOUT DETECTION
    # ===========================
    
    def detect_breakout(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Detect if price is breaking out of key levels.
        Critical for order type determination.
        """
        try:
            lookback = min(20, len(df) - 1)
            
            # Recent high/low
            recent_high = df['high'].tail(lookback).max()
            recent_low = df['low'].tail(lookback).min()
            
            # Longer-term levels
            if len(df) >= 50:
                longer_high = df['high'].tail(50).max()
                longer_low = df['low'].tail(50).min()
            else:
                longer_high = recent_high
                longer_low = recent_low
            
            # ATR for dynamic thresholds
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else (recent_high - recent_low) * 0.1
            
            breakout_info = {
                'has_breakout': False,
                'type': None,
                'strength': 0,
                'levels': {
                    'recent_high': recent_high,
                    'recent_low': recent_low,
                    'longer_high': longer_high,
                    'longer_low': longer_low
                },
                'volume_confirmed': False
            }
            
            # Check for resistance breakout
            if current_price > recent_high * 0.998:  # Within 0.2% or above
                breakout_strength = (current_price - recent_high) / recent_high
                breakout_info.update({
                    'has_breakout': True,
                    'type': 'resistance_break',
                    'strength': breakout_strength,
                    'level_broken': recent_high
                })
                
                # Check if also breaking longer-term
                if current_price > longer_high * 0.998:
                    breakout_info['strength'] *= 1.5
                    breakout_info['type'] = 'major_resistance_break'
            
            # Check for support breakdown
            elif current_price < recent_low * 1.002:  # Within 0.2% or below
                breakdown_strength = (recent_low - current_price) / recent_low
                breakout_info.update({
                    'has_breakout': True,
                    'type': 'support_break',
                    'strength': breakdown_strength,
                    'level_broken': recent_low
                })
                
                # Check if also breaking longer-term
                if current_price < longer_low * 1.002:
                    breakout_info['strength'] *= 1.5
                    breakout_info['type'] = 'major_support_break'
            
            # Volume confirmation
            if breakout_info['has_breakout'] and 'volume_ratio' in df.columns:
                volume_ratio = df['volume_ratio'].iloc[-1]
                if volume_ratio > 1.5:
                    breakout_info['volume_confirmed'] = True
                    breakout_info['strength'] *= 1.2
            
            return breakout_info
            
        except Exception as e:
            self.logger.error(f"Breakout detection error: {e}")
            return {'has_breakout': False, 'type': None, 'strength': 0}
    
# ===========================
# UTILITY FUNCTIONS
# ===========================

def calculate_position_size(account_balance: float, risk_per_trade: float,
                           entry_price: float, stop_loss: float) -> float:
    """Calculate position size based on risk management"""
    risk_amount = account_balance * risk_per_trade
    stop_distance = abs(entry_price - stop_loss)
    
    if stop_distance > 0:
        position_size = risk_amount / stop_distance
    else:
        position_size = 0
    
    return position_size

def calculate_risk_reward_ratio(entry: float, stop: float, target: float) -> float:
    """Calculate risk/reward ratio"""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    
    if risk > 0:
        return reward / risk
    return 0

def format_signal_message(signal: Dict) -> str:
    """Format signal for display/notification"""
    return f"""
    🎯 NEW SIGNAL ALERT
    ========================
    Symbol: {signal.get('symbol')}
    Side: {signal.get('side')}
    Quality: {signal.get('quality')} ({signal.get('confidence', 0):.1f}% confidence)
    
    Entry: {signal.get('entry_price', 0):.6f}
    Stop Loss: {signal.get('stop_loss', 0):.6f}
    Target 1: {signal.get('take_profit_1', 0):.6f}
    Target 2: {signal.get('take_profit_2', 0):.6f}
    
    Order Type: {signal.get('order_type')}
    R/R Ratio: {signal.get('risk_reward_ratio', 0):.2f}
    
    Market Regime: {signal.get('market_regime')}
    
    ⚠️ Warnings: {', '.join(signal.get('warnings', [])) or 'None'}
    """

# ===========================
# EXPORTS
# ===========================

__all__ = [
    'SignalAnalyzer',
    'calculate_position_size',
    'calculate_risk_reward_ratio',
    'format_signal_message'
]