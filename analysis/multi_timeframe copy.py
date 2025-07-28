"""
Multi-Timeframe Signal Analyzer for the Enhanced Bybit Trading System.
"""

import pandas as pd
import numpy as np
import ta
import time
import logging
from typing import Dict, List, Optional
from config.config import EnhancedSystemConfig


class MultiTimeframeAnalyzer:
    """Multi-timeframe signal confirmation analyzer"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_symbol_multi_timeframe(self, symbol: str, primary_signal: Dict) -> Dict:
        """Analyze symbol across multiple timeframes for signal confirmation"""
        try:
            self.logger.debug(f"🔍 Multi-timeframe analysis for {symbol}")
            
            confirmation_results = {
                'confirmed_timeframes': [],
                'conflicting_timeframes': [],
                'neutral_timeframes': [],
                'confirmation_strength': 0.0,
                'mtf_confidence_boost': 0.0,
                'timeframe_signals': {}
            }
            
            primary_side = primary_signal['side'].lower()
            
            # Analyze each confirmation timeframe
            for timeframe in self.config.confirmation_timeframes:
                try:
                    self.logger.debug(f"   Analyzing {timeframe} timeframe...")
                    
                    # Fetch data for this timeframe
                    df = self.fetch_timeframe_data(symbol, timeframe)
                    if df.empty or len(df) < 50:
                        self.logger.debug(f"   Insufficient {timeframe} data for {symbol}")
                        confirmation_results['neutral_timeframes'].append(timeframe)
                        continue
                    
                    # Calculate indicators for this timeframe
                    df = self.calculate_timeframe_indicators(df)
                    
                    # Generate signal for this timeframe
                    timeframe_signal = self.generate_timeframe_signal(df, timeframe)

                    # self.logger.info(f"   primary signal: {primary_signal}")
                    # self.logger.info(f"   {timeframe} signal: {timeframe_signal}") 
                    
                    if timeframe_signal:
                        confirmation_results['timeframe_signals'][timeframe] = timeframe_signal

                        if timeframe_signal['side'] == primary_side:
                            confirmation_results['confirmed_timeframes'].append(timeframe)
                            self.logger.debug(f"   ✅ {timeframe}: {symbol} {timeframe_signal['side'].upper()} signal confirmed")
                        else:
                            confirmation_results['conflicting_timeframes'].append(timeframe)
                            self.logger.debug(f"   ⚠️ {timeframe}: {symbol} {timeframe_signal['side'].upper()} signal conflicts")
                    else:
                        confirmation_results['neutral_timeframes'].append(timeframe)
                        self.logger.debug(f"   ➖ {timeframe}: {symbol} No clear signal")
                    
                    # Add small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"   Error analyzing {timeframe}: {e}")
                    confirmation_results['neutral_timeframes'].append(timeframe)
            
            # Calculate confirmation strength
            total_timeframes = len(self.config.confirmation_timeframes)
            confirmed_count = len(confirmation_results['confirmed_timeframes'])
            conflicting_count = len(confirmation_results['conflicting_timeframes'])
            
            if total_timeframes > 0:
                confirmation_results['confirmation_strength'] = confirmed_count / total_timeframes
                
                # Calculate confidence boost
                if confirmed_count > 0:
                    # Boost based on confirmed timeframes minus conflicts
                    net_confirmation = confirmed_count - conflicting_count
                    confirmation_results['mtf_confidence_boost'] = max(0, 
                        (net_confirmation / total_timeframes) * (self.config.mtf_weight_multiplier - 1) * 100
                    )
            
            self.logger.debug(f"   📊 Confirmation: {confirmed_count}/{total_timeframes} timeframes")
            self.logger.debug(f"   💪 Strength: {confirmation_results['confirmation_strength']:.1%}")
            
            return confirmation_results
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe analysis error for {symbol}: {e}")
            return {
                'confirmed_timeframes': [],
                'conflicting_timeframes': [],
                'neutral_timeframes': self.config.confirmation_timeframes.copy(),
                'confirmation_strength': 0.0,
                'mtf_confidence_boost': 0.0,
                'timeframe_signals': {}
            }
    
    def fetch_timeframe_data(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        """Fetch OHLCV data for specific timeframe with database-configured limit"""
        try:
            # Use database-configured limit if not specified
            if limit is None:
                limit = self.config.ohlcv_limit_mtf
                
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {symbol} {timeframe} data: {e}")
            return pd.DataFrame()
    
    def calculate_timeframe_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential indicators for timeframe analysis"""
        try:
            if df.empty or len(df) < 20:
                return df
            
            # Ensure proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Essential indicators for signal generation
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Add Stochastic RSI and Ichimoku for MTF analysis
            try:
                # Stochastic RSI
                stoch_rsi = ta.momentum.StochRSIIndicator(
                    close=df['close'], 
                    window=14, 
                    smooth1=3, 
                    smooth2=3
                )
                df['stoch_rsi_k'] = stoch_rsi.stochrsi_k() * 100
                df['stoch_rsi_d'] = stoch_rsi.stochrsi_d() * 100
            except Exception:
                df['stoch_rsi_k'] = 50
                df['stoch_rsi_d'] = 50
            
            # Fill NaN values
            df = df.ffill().bfill().fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe indicators: {e}")
            return df
    
    def generate_timeframe_signal(self, df: pd.DataFrame, timeframe: str) -> Optional[Dict]:
        """Generate signal for specific timeframe with indicators"""
        try:
            if df.empty or len(df) < 20:
                return None
            
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Existing indicators
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            ema_12 = latest.get('ema_12', current_price)
            ema_26 = latest.get('ema_26', current_price)
            volume_ratio = latest.get('volume_ratio', 1)
            bb_position = latest.get('bb_position', 0.5)
            
            # Stochastic RSI
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            stoch_rsi_d = latest.get('stoch_rsi_d', 50)
            
            signal = None
            
            # BUY signal conditions
            buy_conditions = [
                30 < rsi < 75,
                macd > macd_signal,
                ema_12 > ema_26,
                current_price > latest.get('sma_20', current_price),
                volume_ratio > 1.0,
                bb_position < 0.8,
                stoch_rsi_k > stoch_rsi_d,  # Stochastic RSI bullish
                stoch_rsi_k < 80            # Not overbought
            ]
            
            # SELL signal conditions
            sell_conditions = [
                25 < rsi < 70,
                macd < macd_signal,
                ema_12 < ema_26,
                current_price < latest.get('sma_20', current_price),
                volume_ratio > 1.0,
                bb_position > 0.2,
                stoch_rsi_k < stoch_rsi_d,  # Stochastic RSI bearish
                stoch_rsi_k > 20            # Not oversold
            ]
            
            buy_score = sum(buy_conditions)
            sell_score = sum(sell_conditions)
            
            if buy_score >= 6:  # Increased threshold due to added conditions
                signal = {
                    'side': 'buy',
                    'timeframe': timeframe,
                    'strength': buy_score / len(buy_conditions),
                    'price': current_price,
                    'rsi': rsi,
                    'macd_bullish': macd > macd_signal,
                    'ema_bullish': ema_12 > ema_26,
                    'trend_bullish': current_price > latest.get('sma_20', current_price),
                    'volume_sufficient': volume_ratio > 1.0,
                    'stoch_rsi_bullish': stoch_rsi_k > stoch_rsi_d
                }
            elif sell_score >= 6:
                signal = {
                    'side': 'sell',
                    'timeframe': timeframe,
                    'strength': sell_score / len(sell_conditions),
                    'price': current_price,
                    'rsi': rsi,
                    'macd_bullish': macd > macd_signal,
                    'ema_bullish': ema_12 > ema_26,
                    'trend_bullish': current_price > latest.get('sma_20', current_price),
                    'volume_sufficient': volume_ratio > 1.0,
                    'stoch_rsi_bullish': stoch_rsi_k > stoch_rsi_d
                }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating {timeframe} signal: {e}")
            return None