"""
Enhanced Technical Analysis with 50+ indicators including FIXED Ichimoku and Stochastic RSI.
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import Dict
from config.config import EnhancedSystemConfig


class EnhancedTechnicalAnalysis:
    """Professional-grade technical analysis with 50+ indicators including FIXED Ichimoku and Stochastic RSI"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: EnhancedSystemConfig = None) -> pd.DataFrame:
        """Calculate comprehensive technical indicators with FIXED implementations"""
        if df.empty or len(df) < 50:
            return df
        
        # Use default config if none provided
        if config is None:
            config = EnhancedSystemConfig()
        
        try:
            # Ensure proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Basic Moving Averages
            df['sma_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_overbought'] = df['rsi'] > 70
            df['rsi_oversold'] = df['rsi'] < 30
            
            df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
            df['stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
            
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            df['macd_bullish'] = df['macd'] > df['macd_signal']
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility Indicators
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['keltner_upper'] = df['sma_20'] + 2 * df['atr']
            df['keltner_lower'] = df['sma_20'] - 2 * df['atr']
            
            # Volume Indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_spike'] = df['volume_ratio'] > 2.0
            
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['ad_line'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
            df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            
            # VWAP
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (df['typical_price'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # Support and Resistance
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            # Trend Indicators
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            
            # Custom Indicators
            df['price_momentum'] = df['close'].pct_change(5) * 100
            df['volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()

            # FIXED: Stochastic RSI with proper error handling
            try:
                stoch_rsi = ta.momentum.StochRSIIndicator(
                    close=df['close'], 
                    window=config.stoch_rsi_window, 
                    smooth1=config.stoch_rsi_smooth_k, 
                    smooth2=config.stoch_rsi_smooth_d
                )
                df['stoch_rsi_k'] = stoch_rsi.stochrsi_k() * 100  # Scale to 0-100
                df['stoch_rsi_d'] = stoch_rsi.stochrsi_d() * 100  # Scale to 0-100
                df['stoch_rsi_overbought'] = df['stoch_rsi_k'] > 80
                df['stoch_rsi_oversold'] = df['stoch_rsi_k'] < 20
                logging.debug("✅ Stochastic RSI calculated successfully")
            except Exception as e:
                logging.warning(f"Stochastic RSI calculation failed: {e}")
                df['stoch_rsi_k'] = 50
                df['stoch_rsi_d'] = 50
                df['stoch_rsi_overbought'] = False
                df['stoch_rsi_oversold'] = False
            
            # FIXED: Ichimoku Cloud with proper error handling
            try:
                # Method 1: Try using ta library with correct method names
                ichimoku = ta.trend.IchimokuIndicator(
                    high=df['high'], 
                    low=df['low'], 
                    window1=config.ichimoku_window1,  # Tenkan-sen period
                    window2=config.ichimoku_window2,  # Kijun-sen period
                    window3=config.ichimoku_window3   # Senkou Span B period
                )
                
                df['ichimoku_tenkan'] = ichimoku.ichimoku_conversion_line()
                df['ichimoku_kijun'] = ichimoku.ichimoku_base_line()
                df['ichimoku_span_a'] = ichimoku.ichimoku_a()
                df['ichimoku_span_b'] = ichimoku.ichimoku_b()
                
                # FIXED: Calculate Chikou Span manually (no ichimoku_chikou_line method exists)
                df['ichimoku_chikou'] = df['close'].shift(-config.ichimoku_window2)
                
                logging.debug("✅ Ichimoku indicators calculated using ta library")
                
            except Exception as e:
                logging.warning(f"Ichimoku ta library failed: {e}. Using manual calculation...")
                
                # Method 2: Manual calculation as fallback
                try:
                    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
                    tenkan_high = df['high'].rolling(window=config.ichimoku_window1).max()
                    tenkan_low = df['low'].rolling(window=config.ichimoku_window1).min()
                    df['ichimoku_tenkan'] = (tenkan_high + tenkan_low) / 2
                    
                    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
                    kijun_high = df['high'].rolling(window=config.ichimoku_window2).max()
                    kijun_low = df['low'].rolling(window=config.ichimoku_window2).min()
                    df['ichimoku_kijun'] = (kijun_high + kijun_low) / 2
                    
                    # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods forward
                    df['ichimoku_span_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(config.ichimoku_window2)
                    
                    # Senkou Span B: (52-period high + 52-period low) / 2, shifted 26 periods forward
                    span_b_high = df['high'].rolling(window=config.ichimoku_window3).max()
                    span_b_low = df['low'].rolling(window=config.ichimoku_window3).min()
                    df['ichimoku_span_b'] = ((span_b_high + span_b_low) / 2).shift(config.ichimoku_window2)
                    
                    # Chikou Span: Current close price, shifted 26 periods backward
                    df['ichimoku_chikou'] = df['close'].shift(-config.ichimoku_window2)
                    
                    logging.debug("✅ Ichimoku indicators calculated manually")
                    
                except Exception as e2:
                    logging.error(f"Manual Ichimoku calculation also failed: {e2}")
                    # Add default values
                    for col in ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a', 'ichimoku_span_b', 'ichimoku_chikou']:
                        df[col] = 0
            
            # Calculate Ichimoku signals
            try:
                df['ichimoku_bullish'] = (
                    (df['close'] > df['ichimoku_span_a']) & 
                    (df['close'] > df['ichimoku_span_b']) & 
                    (df['ichimoku_tenkan'] > df['ichimoku_kijun'])
                )
                
                df['ichimoku_bearish'] = (
                    (df['close'] < df['ichimoku_span_a']) & 
                    (df['close'] < df['ichimoku_span_b']) & 
                    (df['ichimoku_tenkan'] < df['ichimoku_kijun'])
                )
                
                # Additional Ichimoku signals
                cloud_top = np.maximum(df['ichimoku_span_a'], df['ichimoku_span_b'])
                cloud_bottom = np.minimum(df['ichimoku_span_a'], df['ichimoku_span_b'])
                df['ichimoku_in_cloud'] = (df['close'] >= cloud_bottom) & (df['close'] <= cloud_top)
                
                # TK Cross signals
                df['ichimoku_tk_cross_bullish'] = (
                    (df['ichimoku_tenkan'] > df['ichimoku_kijun']) & 
                    (df['ichimoku_tenkan'].shift(1) <= df['ichimoku_kijun'].shift(1))
                )
                
                df['ichimoku_tk_cross_bearish'] = (
                    (df['ichimoku_tenkan'] < df['ichimoku_kijun']) & 
                    (df['ichimoku_tenkan'].shift(1) >= df['ichimoku_kijun'].shift(1))
                )
                
            except Exception as e:
                logging.error(f"Error calculating Ichimoku signals: {e}")
                # Add default boolean values
                for col in ['ichimoku_bullish', 'ichimoku_bearish', 'ichimoku_in_cloud', 'ichimoku_tk_cross_bullish', 'ichimoku_tk_cross_bearish']:
                    df[col] = False
            
            # Fibonacci Levels (Dynamic)
            df = EnhancedTechnicalAnalysis.add_fibonacci_levels(df)
            
            # Pattern Recognition
            df = EnhancedTechnicalAnalysis.add_pattern_recognition(df)
            
            # Market Structure
            df = EnhancedTechnicalAnalysis.add_market_structure(df)
            
            # Fill NaN values safely
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Forward fill then backward fill, finally fill remaining with 0
            df = df.ffill().bfill().fillna(0)
            
            logging.debug("✅ All technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logging.error(f"Error in enhanced technical analysis: {e}")
            # Return dataframe with basic indicators if enhanced calculation fails
            return EnhancedTechnicalAnalysis.add_fallback_indicators(df)
    
    @staticmethod
    def add_fallback_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add basic fallback indicators when enhanced calculation fails"""
        try:
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
            df['support'] = df['low'].rolling(window=20).min()
            df['resistance'] = df['high'].rolling(window=20).max()
            
            # Add default Ichimoku and Stochastic RSI values
            ichimoku_columns = [
                'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a', 
                'ichimoku_span_b', 'ichimoku_chikou'
            ]
            for col in ichimoku_columns:
                df[col] = 0
            
            ichimoku_bool_columns = [
                'ichimoku_bullish', 'ichimoku_bearish', 'ichimoku_in_cloud',
                'ichimoku_tk_cross_bullish', 'ichimoku_tk_cross_bearish'
            ]
            for col in ichimoku_bool_columns:
                df[col] = False
            
            df['stoch_rsi_k'] = 50
            df['stoch_rsi_d'] = 50
            df['stoch_rsi_overbought'] = False
            df['stoch_rsi_oversold'] = False
            
            # Basic Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Basic MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Fill NaN values
            df = df.ffill().bfill().fillna(0)
            
            logging.info("✅ Fallback indicators added successfully")
            return df
            
        except Exception as e2:
            logging.error(f"Fallback technical analysis also failed: {e2}")
            return df
    
    @staticmethod
    def add_fibonacci_levels(df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        """Add dynamic Fibonacci retracement levels"""
        try:
            # Calculate rolling high and low for Fibonacci levels
            rolling_high = df['high'].rolling(window=period).max()
            rolling_low = df['low'].rolling(window=period).min()
            
            # Calculate Fibonacci levels
            fib_range = rolling_high - rolling_low
            df['fib_0'] = rolling_high
            df['fib_236'] = rolling_high - (fib_range * 0.236)
            df['fib_382'] = rolling_high - (fib_range * 0.382)
            df['fib_500'] = rolling_high - (fib_range * 0.500)
            df['fib_618'] = rolling_high - (fib_range * 0.618)
            df['fib_786'] = rolling_high - (fib_range * 0.786)
            df['fib_100'] = rolling_low
            
            return df
        except Exception:
            return df
    
    @staticmethod
    def add_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        try:
            # Basic pattern recognition
            body_size = abs(df['close'] - df['open'])
            candle_range = df['high'] - df['low']
            upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
            lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
            
            # Doji
            df['doji'] = (body_size < candle_range * 0.1).astype(int)
            
            # Hammer
            df['hammer'] = ((lower_shadow > body_size * 2) & 
                           (upper_shadow < body_size * 0.1) & 
                           (df['close'] > df['open'])).astype(int)
            
            # Shooting Star
            df['shooting_star'] = ((upper_shadow > body_size * 2) & 
                                  (lower_shadow < body_size * 0.1) & 
                                  (df['close'] < df['open'])).astype(int)
            
            # Engulfing Patterns
            prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
            curr_body = abs(df['close'] - df['open'])
            
            df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                      (df['close'].shift(1) < df['open'].shift(1)) &
                                      (curr_body > prev_body * 1.5) &
                                      (df['close'] > df['open'].shift(1)) &
                                      (df['open'] < df['close'].shift(1))).astype(int)
            
            df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                      (df['close'].shift(1) > df['open'].shift(1)) &
                                      (curr_body > prev_body * 1.5) &
                                      (df['close'] < df['open'].shift(1)) &
                                      (df['open'] > df['close'].shift(1))).astype(int)
            
            return df
        except Exception:
            return df
    
    @staticmethod
    def add_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure analysis"""
        try:
            # Higher Highs and Lower Lows
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
            
            # Swing Points
            df['swing_high'] = df['high'].rolling(window=5, center=True).max() == df['high']
            df['swing_low'] = df['low'].rolling(window=5, center=True).min() == df['low']
            
            # Trend Strength
            price_change = df['close'].pct_change(20)
            df['trend_strength'] = price_change.rolling(window=5).mean()
            
            # Market Regime
            volatility = df['close'].pct_change().rolling(window=20).std()
            df['volatility_regime'] = (volatility > volatility.rolling(window=50).mean()).astype(int)
            
            return df
        except Exception:
            return df