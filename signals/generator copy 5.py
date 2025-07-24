"""
Enhanced Signal Generation for the Bybit Trading System.
UPDATED: Reduced strictness while maintaining quality
- More flexible signal thresholds
- Balanced market structure analysis
- Improved ranking logic
- Reasonable risk management
- Multi-layer validation system
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from config.config import EnhancedSystemConfig


class SignalGenerator:
    """Enhanced signal generation and ranking system with balanced approach"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_enhanced_signal(self, df: pd.DataFrame, symbol_data: Dict, 
                                volume_entry: Dict, confluence_zones: List[Dict]) -> Optional[Dict]:
            """FIXED: Generate proper trading signals based on correct technical analysis"""
            try:
                latest = df.iloc[-1]
                current_price = symbol_data['current_price']
                symbol = symbol_data['symbol']
                
                # Get all indicator values with safety checks
                rsi = latest.get('rsi', 50)
                macd = latest.get('macd', 0)
                macd_signal = latest.get('macd_signal', 0)
                ema_12 = latest.get('ema_12', current_price)
                ema_26 = latest.get('ema_26', current_price)
                sma_20 = latest.get('sma_20', current_price)
                sma_50 = latest.get('sma_50', current_price)
                sma_200 = latest.get('sma_200', current_price)
                volume_ratio = latest.get('volume_ratio', 1)
                bb_position = latest.get('bb_position', 0.5)
                bb_upper = latest.get('bb_upper', current_price * 1.02)
                bb_lower = latest.get('bb_lower', current_price * 0.98)
                atr = latest.get('atr', current_price * 0.02)
                
                # Stochastic RSI values
                stoch_rsi_k = latest.get('stoch_rsi_k', 50)
                stoch_rsi_d = latest.get('stoch_rsi_d', 50)
                
                # Ichimoku signals
                ichimoku_bullish = latest.get('ichimoku_bullish', False)
                ichimoku_bearish = latest.get('ichimoku_bearish', False)
                
                # Market structure analysis
                market_structure = self.analyze_market_structure(df, current_price)
                
                signal = None
                
                # ===== FIXED: PROPER BUY SIGNAL CONDITIONS =====
                # BUY when price is OVERSOLD and near SUPPORT levels
                
                # Primary BUY conditions (FIXED)
                buy_technical_conditions = [
                    rsi < 30,                                    # FIXED: Truly oversold (was > 40)
                    macd > macd_signal,                         # FIXED: Bullish momentum
                    current_price < sma_20,                     # FIXED: Below recent average (potential bounce)
                    current_price > sma_200 * 0.95,            # FIXED: Above long-term support
                    volume_ratio > 1.2,                         # Strong volume on decline
                    bb_position < 0.3,                          # FIXED: Near lower BB (oversold)
                ]
                
                # BUY trend conditions (FIXED)
                buy_trend_conditions = [
                    ema_12 > ema_26 * 0.98,                     # FIXED: Trend not completely broken
                    current_price > sma_50 * 0.95,             # FIXED: Above medium-term support
                    sma_20 > sma_50 * 0.98,                     # FIXED: Medium-term trend intact
                    current_price > bb_lower * 1.02,           # FIXED: Above lower Bollinger Band
                ]
                
                # BUY Stochastic RSI conditions (FIXED)
                buy_stoch_conditions = [
                    stoch_rsi_k < 25,                           # FIXED: Truly oversold (was > 70)
                    stoch_rsi_k > stoch_rsi_d,                  # FIXED: Bullish crossover (was <)
                    stoch_rsi_k > 5,                            # FIXED: Not at absolute bottom
                    stoch_rsi_d < 30,                           # FIXED: Signal line also oversold
                ]
                
                # Market structure for BUY (FIXED)
                buy_structure_conditions = [
                    market_structure.get('near_support', False),        # Near support level
                    not market_structure.get('strong_downtrend', True), # Not in strong downtrend
                    market_structure.get('bounce_potential', False),    # Bounce potential
                    current_price < latest.get('vwap', current_price),  # Below VWAP (discount)
                ]
                
                # Count conditions
                buy_tech_score = sum(buy_technical_conditions)
                buy_trend_score = sum(buy_trend_conditions)
                buy_stoch_score = sum(buy_stoch_conditions)
                buy_structure_score = sum(buy_structure_conditions)
                
                # STRICT BUY requirements (FIXED)
                if (buy_tech_score >= 4 and buy_trend_score >= 2 and 
                    buy_stoch_score >= 2 and buy_structure_score >= 1):
                    
                    signal = self.create_buy_signal(
                        symbol_data, current_price, latest, volume_entry, 
                        confluence_zones, buy_tech_score, buy_trend_score, 
                        buy_stoch_score, buy_structure_score
                    )
                
                # ===== FIXED: PROPER SELL SIGNAL CONDITIONS =====  
                # SELL when price is OVERBOUGHT and near RESISTANCE levels
                if not signal:  # Only if no BUY signal
                    
                    # Primary SELL conditions (FIXED)
                    sell_technical_conditions = [
                        rsi > 75,                                   # FIXED: Truly overbought (was < 65)
                        macd < macd_signal,                         # FIXED: Bearish momentum
                        current_price > sma_20,                     # FIXED: Above recent average (overextended)
                        current_price < sma_200 * 1.10,            # FIXED: Below long-term resistance
                        volume_ratio > 1.5,                         # Strong volume on rise
                        bb_position > 0.7,                          # FIXED: Near upper BB (overbought)
                    ]
                    
                    # SELL trend conditions (FIXED)
                    sell_trend_conditions = [
                        ema_12 < ema_26 * 1.02,                    # FIXED: Trend turning bearish
                        current_price < sma_50 * 1.08,             # FIXED: Below medium-term resistance  
                        sma_20 < sma_50 * 1.05,                    # FIXED: Medium-term trend turning
                        current_price < bb_upper * 0.98,           # FIXED: Rejected from upper BB
                    ]
                    
                    # SELL Stochastic RSI conditions (FIXED)
                    sell_stoch_conditions = [
                        stoch_rsi_k > 80,                           # FIXED: Truly overbought (was < 70)
                        stoch_rsi_k < stoch_rsi_d,                  # FIXED: Bearish crossover
                        stoch_rsi_k < 95,                           # FIXED: Not at absolute top
                        stoch_rsi_d > 75,                           # FIXED: Signal line also overbought
                    ]
                    
                    # Market structure for SELL (FIXED)
                    sell_structure_conditions = [
                        market_structure.get('near_resistance', False),     # Near resistance level
                        market_structure.get('reversal_signals', False),    # Reversal signals
                        not market_structure.get('strong_uptrend', True),   # Not in strong uptrend
                        current_price > latest.get('vwap', current_price),  # Above VWAP (premium)
                    ]
                    
                    # Count conditions
                    sell_tech_score = sum(sell_technical_conditions)
                    sell_trend_score = sum(sell_trend_conditions)
                    sell_stoch_score = sum(sell_stoch_conditions)
                    sell_structure_score = sum(sell_structure_conditions)
                    
                    # STRICT SELL requirements (FIXED)
                    if (sell_tech_score >= 4 and sell_trend_score >= 2 and 
                        sell_stoch_score >= 2 and sell_structure_score >= 1):
                        
                        signal = self.create_sell_signal(
                            symbol_data, current_price, latest, volume_entry, 
                            confluence_zones, sell_tech_score, sell_trend_score, 
                            sell_stoch_score, sell_structure_score
                        )
                
                return signal
                
            except Exception as e:
                self.logger.error(f"Signal generation error for {symbol_data.get('symbol', 'unknown')}: {e}")
                return None
        
    def analyze_market_structure(self, df: pd.DataFrame, current_price: float) -> Dict:
        """FIXED: Analyze market structure for proper signal context"""
        try:
            # Calculate recent price action
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            price_range = recent_high - recent_low
            
            # Check proximity to support/resistance
            near_support = current_price < (recent_low + price_range * 0.3)
            near_resistance = current_price > (recent_high - price_range * 0.3)
            
            # Trend analysis
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            strong_uptrend = (sma_20 > sma_50 * 1.05) and (current_price > sma_20 * 1.02)
            strong_downtrend = (sma_20 < sma_50 * 0.95) and (current_price < sma_20 * 0.98)
            
            # Reversal signals
            recent_candles = df.tail(5)
            reversal_signals = False
            
            # Check for doji, hammer, shooting star patterns
            for _, candle in recent_candles.iterrows():
                body_size = abs(candle['close'] - candle['open'])
                candle_range = candle['high'] - candle['low']
                
                if body_size < candle_range * 0.3:  # Small body (doji-like)
                    reversal_signals = True
                    break
            
            # Volume analysis for distribution/accumulation
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['volume'].tail(5).mean()
            high_volume = recent_volume > volume_ma * 1.5
            
            return {
                'near_support': near_support,
                'near_resistance': near_resistance,
                'strong_uptrend': strong_uptrend,
                'strong_downtrend': strong_downtrend,
                'reversal_signals': reversal_signals,
                'bounce_potential': near_support and not strong_downtrend,
                'distribution_signs': near_resistance and high_volume,
                'price_range_position': (current_price - recent_low) / price_range if price_range > 0 else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return {
                'near_support': False,
                'near_resistance': False,
                'strong_uptrend': False,
                'strong_downtrend': False,
                'reversal_signals': False,
                'bounce_potential': False,
                'distribution_signs': False,
                'price_range_position': 0.5
            }
       
    def get_default_market_structure(self) -> Dict:
        """Default market structure when insufficient data"""
        return {
            'nearest_support': 0,
            'nearest_resistance': 0,
            'support_distance': 0.1,
            'resistance_distance': 0.1,
            'near_support': False,
            'near_resistance': False,
            'strong_uptrend': False,
            'strong_downtrend': False,
            'bounce_potential': True,  # Allow signals when no data
            'reversal_signals': True,  # Allow signals when no data
            'distribution_signs': False,
            'consolidation': True
        }
    
    def is_very_strong_uptrend(self, df: pd.DataFrame) -> bool:
        """Check if we're in a very strong uptrend (stricter than regular check)"""
        try:
            if len(df) < 20:
                return False
            
            recent = df.tail(20)
            latest = df.iloc[-1]
            
            # Multiple timeframe trend check
            sma_20 = latest.get('sma_20', 0)
            sma_50 = latest.get('sma_50', 0)
            ema_12 = latest.get('ema_12', 0)
            ema_26 = latest.get('ema_26', 0)
            
            # Very strong uptrend conditions (stricter)
            conditions = [
                sma_20 > sma_50 * 1.05,  # 20 SMA well above 50 SMA (was 1.02)
                ema_12 > ema_26 * 1.03,  # EMAs aligned bullishly (was 1.01)
                latest['close'] > sma_20 * 1.02,  # Price well above 20 SMA
                recent['close'].iloc[-1] > recent['close'].iloc[-10] * 1.05,  # 5%+ gain over 10 periods (was 1.03)
                recent['volume'].mean() > df['volume'].tail(50).mean() * 1.2  # Above average volume
            ]
            
            return sum(conditions) >= 4  # Need 4/5 conditions
            
        except Exception as e:
            self.logger.error(f"Very strong uptrend check error: {e}")
            return False
    
    def create_buy_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                         volume_entry: Dict, confluence_zones: List[Dict],
                         tech_score: int, trend_score: int, stoch_score: int, structure_score: int) -> Dict:
        """FIXED: Create proper BUY signal at support levels"""
        try:
            # FIXED: Entry price logic for BUY (buy near support)
            entry_candidates = [current_price]
            
            # Use support levels from confluence zones  
            support_zones = [zone for zone in confluence_zones
                           if zone['zone_type'] == 'support' and 
                           zone['price'] < current_price * 1.02 and  # Within 2% below current price
                           zone['price'] > current_price * 0.95]     # Not too far below
            
            if support_zones:
                entry_candidates.append(support_zones[0]['price'])
            
            # Use volume-based entry if it's a support level
            if (volume_entry.get('confidence', 0) > 0.6 and 
                volume_entry.get('entry_price', current_price) < current_price):
                entry_candidates.append(volume_entry['entry_price'])
            
            # Use lower Bollinger Band as support
            bb_lower = latest.get('bb_lower', current_price * 0.98)
            if bb_lower < current_price and bb_lower > current_price * 0.95:
                entry_candidates.append(bb_lower)
            
            # FIXED: Choose optimal entry (slightly below current for better fill)
            optimal_entry = min(entry_candidates)  # FIXED: Take the lowest (best) price for BUY
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            # FIXED: Order type logic for BUY
            if distance_pct > 0.01 or latest.get('stoch_rsi_k', 50) < 25:  # Use limit if far or oversold
                order_type = 'limit'
            else:
                order_type = 'market'
                optimal_entry = current_price
            
            # FIXED: Stop loss and take profit for BUY
            atr = latest.get('atr', current_price * 0.02)
            stop_loss = optimal_entry - (2.0 * atr)      # FIXED: Stop below entry
            take_profit_1 = optimal_entry + (3.0 * atr)  # FIXED: TP above entry
            take_profit_2 = optimal_entry + (6.0 * atr)  # FIXED: TP2 higher above entry
            
            # FIXED: Risk-reward calculation for BUY
            risk_amount = optimal_entry - stop_loss      # FIXED: Risk = entry - stop
            reward_amount = take_profit_2 - optimal_entry # FIXED: Reward = tp - entry
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # FIXED: Confidence calculation for BUY
            base_confidence = 50
            
            # Score-based confidence (FIXED)
            base_confidence += tech_score * 4      # Up to 24 points
            base_confidence += trend_score * 4     # Up to 16 points  
            base_confidence += stoch_score * 4     # Up to 16 points
            base_confidence += structure_score * 3 # Up to 12 points
            
            # Bonus for good support levels
            if len(support_zones) > 0:
                base_confidence += 8
            if volume_entry.get('confidence', 0) > 0.7:
                base_confidence += 6
            
            confidence = min(90, base_confidence)  # Cap at 90%
            
            return {
                'symbol': symbol_data['symbol'],
                'side': 'buy',
                'order_type': order_type,
                'entry_price': optimal_entry,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence': confidence,
                'signal_type': 'fixed_technical_buy',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'entry_methods': {
                    'technical_score': tech_score,
                    'trend_score': trend_score,
                    'stoch_score': stoch_score,
                    'structure_score': structure_score,
                    'support_zones_count': len(support_zones),
                    'volume_confidence': volume_entry.get('confidence', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"BUY signal creation error: {e}")
            return None
    
    def create_sell_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                          volume_entry: Dict, confluence_zones: List[Dict],
                          tech_score: int, trend_score: int, stoch_score: int, structure_score: int) -> Dict:
        """FIXED: Create proper SELL signal at resistance levels"""
        try:
            # FIXED: Entry price logic for SELL (sell near resistance)
            entry_candidates = [current_price]
            
            # Use resistance levels from confluence zones  
            resistance_zones = [zone for zone in confluence_zones
                              if zone['zone_type'] == 'resistance' and 
                              zone['price'] > current_price * 0.98 and  # Within 2% above current price
                              zone['price'] < current_price * 1.05]     # Not too far above
            
            if resistance_zones:
                entry_candidates.append(resistance_zones[0]['price'])
            
            # Use volume-based entry if it's a resistance level
            if (volume_entry.get('confidence', 0) > 0.6 and 
                volume_entry.get('entry_price', current_price) > current_price):
                entry_candidates.append(volume_entry['entry_price'])
            
            # Use upper Bollinger Band as resistance
            bb_upper = latest.get('bb_upper', current_price * 1.02)
            if bb_upper > current_price and bb_upper < current_price * 1.05:
                entry_candidates.append(bb_upper)
            
            # FIXED: Choose optimal entry (slightly above current for better fill)
            optimal_entry = max(entry_candidates)  # FIXED: Take the highest (best) price for SELL
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            # FIXED: Order type logic for SELL
            if distance_pct > 0.01 or latest.get('stoch_rsi_k', 50) > 80:  # Use limit if far or overbought
                order_type = 'limit'
            else:
                order_type = 'market'
                optimal_entry = current_price
            
            # FIXED: Stop loss and take profit for SELL
            atr = latest.get('atr', current_price * 0.02)
            stop_loss = optimal_entry + (2.0 * atr)      # FIXED: Stop above entry
            take_profit_1 = optimal_entry - (3.0 * atr)  # FIXED: TP below entry
            take_profit_2 = optimal_entry - (6.0 * atr)  # FIXED: TP2 lower below entry
            
            # FIXED: Risk-reward calculation for SELL
            risk_amount = stop_loss - optimal_entry      # FIXED: Risk = stop - entry
            reward_amount = optimal_entry - take_profit_2 # FIXED: Reward = entry - tp
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # FIXED: Confidence calculation for SELL
            base_confidence = 50
            
            # Score-based confidence (FIXED)
            base_confidence += tech_score * 4      # Up to 24 points
            base_confidence += trend_score * 4     # Up to 16 points  
            base_confidence += stoch_score * 4     # Up to 16 points
            base_confidence += structure_score * 3 # Up to 12 points
            
            # Bonus for good resistance levels
            if len(resistance_zones) > 0:
                base_confidence += 8
            if volume_entry.get('confidence', 0) > 0.7:
                base_confidence += 6
            
            confidence = min(90, base_confidence)  # Cap at 90%
            
            return {
                'symbol': symbol_data['symbol'],
                'side': 'sell',
                'order_type': order_type,
                'entry_price': optimal_entry,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence': confidence,
                'signal_type': 'fixed_technical_sell',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'entry_methods': {
                    'technical_score': tech_score,
                    'trend_score': trend_score,
                    'stoch_score': stoch_score,
                    'structure_score': structure_score,
                    'resistance_zones_count': len(resistance_zones),
                    'volume_confidence': volume_entry.get('confidence', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"SELL signal creation error: {e}")
            return None
         
    def validate_signal_quality(self, signal: Dict, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """Signal quality validation with more lenient thresholds"""
        try:
            if not signal:
                return None
            
            symbol = signal['symbol']
            side = signal['side']
            confidence = signal['confidence']
            risk_reward = signal['risk_reward_ratio']
            
            # Minimum confidence thresholds (lowered)
            min_confidence = 50 if side == 'buy' else 55  # Lowered from 65/70
            if confidence < min_confidence:
                self.logger.debug(f"Signal filtered: {symbol} confidence {confidence}% below minimum {min_confidence}%")
                return None
            
            # Minimum risk-reward ratio (lowered)
            min_rr = 2.0 if side == 'buy' else 2.2  # Lowered from 2.5/3.0
            if risk_reward < min_rr:
                self.logger.debug(f"Signal filtered: {symbol} R/R {risk_reward:.1f} below minimum {min_rr}")
                return None
            
            # Market structure validation (more lenient)
            if side == 'sell':
                # Don't short unless reasonably near resistance
                if not market_structure['near_resistance'] and market_structure['resistance_distance'] > 0.10:  # Increased from 0.05
                    self.logger.debug(f"SELL signal filtered: {symbol} not near resistance")
                    return None
                
                # Don't short in very strong uptrends only
                if self.is_very_strong_uptrend(df):  # Changed from strong_uptrend
                    self.logger.debug(f"SELL signal filtered: {symbol} in very strong uptrend")
                    return None
            
            elif side == 'buy':
                # Don't buy if too far from support
                if market_structure['support_distance'] > 0.12:  # Increased from 0.08
                    self.logger.debug(f"BUY signal filtered: {symbol} too far from support")
                    return None
            
            # Volume validation (more lenient)
            volume_ratio = df.iloc[-1].get('volume_ratio', 1)
            min_volume = 1.0 if side == 'buy' else 1.3  # Lowered from 1.2/2.0
            if volume_ratio < min_volume:
                self.logger.debug(f"Signal filtered: {symbol} volume {volume_ratio:.1f} below minimum {min_volume}")
                return None
            
            # Volatility check (more lenient)
            atr_pct = signal.get('atr_percentage', 0.02)
            if atr_pct > 0.12:  # Increased from 0.08 (12% daily ATR)
                self.logger.debug(f"Signal filtered: {symbol} too volatile (ATR: {atr_pct*100:.1f}%)")
                return None
            
            self.logger.info(f"✅ Signal validated: {symbol} {side.upper()} - {confidence}% confidence, {risk_reward:.1f}:1 R/R")
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return None

    def rank_opportunities_with_mtf(self, signals: List[Dict]) -> List[Dict]:
        """Enhanced ranking that balances signal quality with accessibility"""
        try:
            opportunities = []
            
            for signal in signals:
                # Get signal quality metrics
                confidence = signal['confidence']
                original_confidence = signal.get('original_confidence', confidence)
                mtf_boost = confidence - original_confidence
                
                # ===== BALANCED SCORING SYSTEM =====
                
                # 1. SIGNAL CONFIDENCE (35% weight)
                confidence_score = confidence / 100
                
                # 2. RISK-REWARD RATIO (25% weight)  
                analysis = signal.get('analysis', {})
                risk_assessment = analysis.get('risk_assessment', {})
                rr_ratio = risk_assessment.get('risk_reward_ratio', 1)
                rr_score = min(1.0, rr_ratio / 3.5)  # Target 3.5:1 R/R (lowered from 4.0)
                
                # 3. MARKET STRUCTURE ALIGNMENT (20% weight)
                entry_methods = signal.get('entry_methods', {})
                structure_score = 0
                if 'structure_score' in entry_methods:
                    structure_score = min(1.0, entry_methods['structure_score'] / 3.0)  # Lowered from 4.0
                elif 'confluence_zones' in entry_methods:
                    structure_score = min(1.0, entry_methods['confluence_zones'] / 2.0)  # Lowered from 3.0
                
                # 4. VOLUME QUALITY (10% weight)
                volume_score = min(1.0, signal['volume_24h'] / 30_000_000)  # Lowered from 50M
                
                # 5. MTF CONFIRMATION (15% weight)
                mtf_analysis = signal.get('mtf_analysis', {})
                confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
                total_timeframes = len(self.config.confirmation_timeframes) if hasattr(self.config, 'confirmation_timeframes') else 3
                
                if total_timeframes > 0:
                    mtf_score = confirmed_count / total_timeframes
                    mtf_bonus = mtf_score * 0.15
                else:
                    mtf_bonus = 0
                
                # 6. DISTANCE PENALTY (3% weight)
                distance = abs(signal['entry_price'] - signal['current_price']) / signal['current_price']
                distance_score = max(0, 1 - distance * 8)  # Less penalty (was 10)
                
                # 7. ORDER TYPE (2% weight)
                order_type_score = 1.0 if signal['order_type'] == 'market' else 0.95  # Less penalty
                
                # ===== CALCULATE TOTAL SCORE =====
                total_score = (
                    confidence_score * 0.35 +      # 35% - Signal quality first
                    rr_score * 0.25 +              # 25% - Risk management second  
                    structure_score * 0.20 +       # 20% - Market structure third
                    volume_score * 0.10 +          # 10% - Volume support
                    mtf_bonus +                    # 15% - MTF confirmation (additive)
                    distance_score * 0.03 +        # 3% - Entry distance
                    order_type_score * 0.02        # 2% - Order type
                )
                
                # ===== BALANCED PRIORITY SYSTEM =====
                mtf_status = signal.get('mtf_status', 'NONE')
                
                # Base priority on confidence and R/R (more accessible)
                if confidence >= 75 and rr_ratio >= 3.5:
                    base_priority = 1000  # Exceptional signals
                elif confidence >= 65 and rr_ratio >= 2.8:
                    base_priority = 500   # High quality signals
                elif confidence >= 55 and rr_ratio >= 2.2:
                    base_priority = 200   # Good signals (increased from 100)
                else:
                    base_priority = 50    # Marginal signals (increased from 10)
                
                # MTF modifier (balanced)
                mtf_modifier = {
                    'STRONG': 1.4,   # 40% boost (was 50%)
                    'PARTIAL': 1.15,  # 15% boost (was 20%)
                    'NONE': 1.0,     # No change
                    'DISABLED': 1.0  # No change
                }.get(mtf_status, 1.0)
                
                final_priority = int(base_priority * mtf_modifier)
                
                # Get MTF confirmation details
                confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
                conflicting_timeframes = mtf_analysis.get('conflicting_timeframes', [])
                
                opportunities.append({
                    **signal,
                    'score': total_score,
                    'priority': final_priority,
                    'confidence_score': confidence_score,
                    'rr_score': rr_score,
                    'structure_score': structure_score,
                    'volume_score': volume_score,
                    'mtf_score': mtf_score if 'mtf_score' in locals() else 0,
                    'distance_score': distance_score,
                    'original_confidence': original_confidence,
                    'mtf_boost': mtf_boost,
                    'confirmed_timeframes': confirmed_timeframes,
                    'conflicting_timeframes': conflicting_timeframes
                })
            
            # Sort by priority first, then by score
            opportunities.sort(key=lambda x: (x['priority'], x['score']), reverse=True)
            
            self.logger.info(f"📊 Ranked {len(opportunities)} opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return signals

    def assess_risk(self, signal: Dict, df: pd.DataFrame, market_data: Dict) -> Dict:
        """Enhanced risk assessment with balanced approach"""
        try:
            latest = df.iloc[-1]
            current_price = signal['current_price']
            
            # Volatility risk (more lenient)
            atr = latest.get('atr', current_price * 0.02)
            volatility = atr / current_price
            
            # Distance risk (more lenient)
            distance = abs(signal['entry_price'] - current_price) / current_price
            distance_risk = min(1.0, distance * 8)  # Reduced from 10
            
            # Condition-based risk (more balanced)
            rsi = latest.get('rsi', 50)
            bb_position = latest.get('bb_position', 0.5)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Risk factors (more lenient thresholds)
            extreme_rsi = rsi < 15 or rsi > 85  # More extreme (was 20/80)
            extreme_bb = bb_position < 0.05 or bb_position > 0.95  # More extreme
            low_volume = volume_ratio < 0.7  # Lowered from 0.8
            
            condition_risk = 0
            if extreme_rsi:
                condition_risk += 0.15  # Reduced from 0.2
            if extreme_bb:
                condition_risk += 0.10  # Reduced from 0.15
            if low_volume:
                condition_risk += 0.08  # Reduced from 0.1
            
            # Side-specific risk (more balanced)
            side_risk = 0.05 if signal['side'] == 'sell' else 0.02  # Reduced penalty for shorts
            
            # MTF risk reduction
            mtf_analysis = signal.get('mtf_analysis', {})
            confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
            total_timeframes = len(self.config.confirmation_timeframes) if hasattr(self.config, 'confirmation_timeframes') else 3
            mtf_risk_reduction = (confirmed_count / total_timeframes) * 0.20 if total_timeframes > 0 else 0  # Increased from 0.15
            
            # Calculate total risk score
            base_risk = (volatility * 2.5 + distance_risk * 2.5 + condition_risk + side_risk)  # Reduced multipliers
            total_risk = max(0.1, min(1.0, base_risk - mtf_risk_reduction))
            
            # Enhanced risk-reward calculation
            if signal['stop_loss'] != 0:
                if signal['side'] == 'buy':
                    risk_amount = abs(signal['entry_price'] - signal['stop_loss'])
                    reward_amount = abs(signal['take_profit_1'] - signal['entry_price'])
                else:
                    risk_amount = abs(signal['stop_loss'] - signal['entry_price'])
                    reward_amount = abs(signal['entry_price'] - signal['take_profit_1'])
                
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            else:
                risk_reward_ratio = 0
            
            # Risk level classification (more lenient)
            if total_risk > 0.85:  # Increased from 0.8
                risk_level = 'Very High'
            elif total_risk > 0.65:  # Increased from 0.6
                risk_level = 'High'
            elif total_risk > 0.45:  # Increased from 0.4
                risk_level = 'Medium'
            elif total_risk > 0.25:  # Increased from 0.2
                risk_level = 'Low'
            else:
                risk_level = 'Very Low'
            
            return {
                'total_risk_score': total_risk,
                'volatility_risk': volatility,
                'distance_risk': distance_risk,
                'condition_risk': condition_risk,
                'side_risk': side_risk,
                'mtf_risk_reduction': mtf_risk_reduction,
                'risk_reward_ratio': risk_reward_ratio,
                'risk_level': risk_level,
                'risk_factors': {
                    'extreme_rsi': extreme_rsi,
                    'extreme_bb': extreme_bb,
                    'low_volume': low_volume,
                    'short_in_uptrend': signal['side'] == 'sell' and latest.get('sma_20', 0) > latest.get('sma_50', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return {'total_risk_score': 0.4, 'risk_level': 'Medium'}

    def assess_signal_risk(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Alias for assess_risk method for backward compatibility"""
        return self.assess_risk(signal, df, {})

    def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol_data: Dict, 
                                   volume_profile: Dict, fibonacci_data: Dict, 
                                   confluence_zones: List[Dict]) -> Optional[Dict]:
        """Comprehensive symbol analysis with balanced approach"""
        try:
            if df.empty or len(df) < 20:
                self.logger.warning(f"Insufficient data for {symbol_data['symbol']}")
                return None
            
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Generate signal
            volume_entry = volume_profile.get('optimal_entry', {})
            signal = self.generate_enhanced_signal(df, symbol_data, volume_entry, confluence_zones)
            
            if not signal:
                return None
            
            # Enhanced analysis
            latest = df.iloc[-1]
            
            # Technical summary
            technical_summary = self.create_technical_summary(df, latest)
            
            # Risk assessment
            risk_assessment = self.assess_risk(signal, df, symbol_data)
            
            # Volume analysis
            volume_analysis = self.analyze_volume_patterns(df)
            
            # Trend strength
            trend_strength = self.calculate_trend_strength(df)
            
            # Price action analysis
            price_action = self.analyze_price_action(df)
            
            # Market conditions
            market_conditions = self.assess_market_conditions(df, symbol_data)
            
            return {
                **signal,
                'analysis': {
                    'technical_summary': technical_summary,
                    'risk_assessment': risk_assessment,
                    'volume_analysis': volume_analysis,
                    'trend_strength': trend_strength,
                    'price_action': price_action,
                    'market_conditions': market_conditions,
                    'volume_profile': volume_profile,
                    'fibonacci_data': fibonacci_data,
                    'confluence_zones': confluence_zones
                },
                'timestamp': pd.Timestamp.now(),
                'timeframe': '30m'  # Primary timeframe
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis error for {symbol_data.get('symbol', 'Unknown')}: {e}")
            return None

    def create_technical_summary(self, df: pd.DataFrame, latest: pd.Series = None) -> Dict:
        """Create technical analysis summary"""
        try:
            # If latest is not provided, get it from the DataFrame
            if latest is None:
                latest = df.iloc[-1]
            
            # Trend analysis
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            ema_12 = latest.get('ema_12', latest['close'])
            ema_26 = latest.get('ema_26', latest['close'])
            
            trend_score = 0
            if latest['close'] > sma_20:
                trend_score += 1
            if latest['close'] > sma_50:
                trend_score += 1
            if sma_20 > sma_50:
                trend_score += 1
            if ema_12 > ema_26:
                trend_score += 1
            
            trend_strength = trend_score / 4.0
            trend_direction = 'bullish' if trend_strength > 0.5 else 'bearish' if trend_strength < 0.3 else 'neutral'
            
            # Momentum analysis
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            
            momentum_score = 0
            if 30 < rsi < 70:
                momentum_score += 1  # Healthy RSI
            if macd > macd_signal:
                momentum_score += 1
            if 20 < stoch_rsi_k < 80:
                momentum_score += 1
            
            momentum_strength = momentum_score / 3.0
            
            # Volatility analysis
            atr = latest.get('atr', latest['close'] * 0.02)
            bb_width = latest.get('bb_upper', latest['close'] * 1.02) - latest.get('bb_lower', latest['close'] * 0.98)
            volatility_pct = (atr / latest['close']) * 100
            
            # Volume analysis
            volume_ratio = latest.get('volume_ratio', 1)
            volume_trend = self.get_volume_trend(df)
            
            return {
                'trend': {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'score': trend_score
                },
                'momentum': {
                    'strength': momentum_strength,
                    'rsi': rsi,
                    'macd_bullish': macd > macd_signal,
                    'stoch_rsi': stoch_rsi_k
                },
                'volatility': {
                    'atr_percentage': volatility_pct,
                    'level': 'high' if volatility_pct > 5 else 'medium' if volatility_pct > 2 else 'low'
                },
                'volume': {
                    'ratio': volume_ratio,
                    'trend': volume_trend,
                    'quality': 'strong' if volume_ratio > 1.5 else 'average' if volume_ratio > 0.8 else 'weak'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Technical summary error: {e}")
            return {}

    def get_volume_trend(self, df: pd.DataFrame) -> str:
        """Analyze volume trend over recent periods"""
        try:
            if 'volume' not in df.columns or len(df) < 10:
                return 'unknown'
            
            recent_volume = df['volume'].tail(5).mean()
            older_volume = df['volume'].tail(15).head(10).mean()
            
            if recent_volume > older_volume * 1.2:
                return 'increasing'
            elif recent_volume < older_volume * 0.8:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'

    def analyze_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns and buying/selling pressure"""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return {'pattern': 'insufficient_data', 'strength': 0}
            
            recent_20 = df.tail(20)
            
            # Volume trend
            volume_ma_5 = recent_20['volume'].rolling(5).mean().iloc[-1]
            volume_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
            
            # Buying vs selling pressure
            up_volume = recent_20[recent_20['close'] > recent_20['open']]['volume'].sum()
            down_volume = recent_20[recent_20['close'] < recent_20['open']]['volume'].sum()
            total_volume = up_volume + down_volume
            
            buying_pressure = up_volume / total_volume if total_volume > 0 else 0.5
            
            # Volume pattern classification
            if volume_ma_5 > volume_ma_20 * 1.3:
                pattern = 'surge'
            elif volume_ma_5 > volume_ma_20 * 1.1:
                pattern = 'increasing'
            elif volume_ma_5 < volume_ma_20 * 0.7:
                pattern = 'declining'
            else:
                pattern = 'stable'
            
            return {
                'pattern': pattern,
                'buying_pressure': buying_pressure,
                'volume_ma_ratio': volume_ma_5 / volume_ma_20 if volume_ma_20 > 0 else 1,
                'strength': min(1.0, volume_ma_5 / volume_ma_20) if volume_ma_20 > 0 else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Volume pattern analysis error: {e}")
            return {'pattern': 'unknown', 'strength': 0.5}

    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Calculate trend strength using multiple indicators"""
        try:
            if len(df) < 50:
                return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'low'}
            
            latest = df.iloc[-1]
            recent_50 = df.tail(50)
            
            # Price trend
            price_change_20 = (latest['close'] - recent_50.iloc[-20]['close']) / recent_50.iloc[-20]['close']
            price_change_50 = (latest['close'] - recent_50.iloc[0]['close']) / recent_50.iloc[0]['close']
            
            # Moving average alignment
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            ema_12 = latest.get('ema_12', latest['close'])
            ema_26 = latest.get('ema_26', latest['close'])
            
            ma_alignment_score = 0
            if latest['close'] > sma_20 > sma_50:
                ma_alignment_score += 2
            elif latest['close'] > sma_20:
                ma_alignment_score += 1
            elif latest['close'] < sma_20 < sma_50:
                ma_alignment_score -= 2
            elif latest['close'] < sma_20:
                ma_alignment_score -= 1
            
            if ema_12 > ema_26:
                ma_alignment_score += 1
            else:
                ma_alignment_score -= 1
            
            # Trend consistency
            bullish_candles = len(recent_50[recent_50['close'] > recent_50['open']])
            consistency = bullish_candles / len(recent_50)
            
            # Overall trend strength
            strength = (abs(price_change_20) + abs(ma_alignment_score) / 4 + consistency) / 3
            strength = min(1.0, strength)
            
            # Direction determination
            if price_change_20 > 0.02 and ma_alignment_score > 0:
                direction = 'strong_bullish'
            elif price_change_20 > 0 and ma_alignment_score >= 0:
                direction = 'bullish'
            elif price_change_20 < -0.02 and ma_alignment_score < 0:
                direction = 'strong_bearish'
            elif price_change_20 < 0 and ma_alignment_score <= 0:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            consistency_level = 'high' if consistency > 0.7 else 'medium' if consistency > 0.4 else 'low'
            
            return {
                'strength': strength,
                'direction': direction,
                'consistency': consistency_level,
                'price_change_20': price_change_20,
                'ma_alignment_score': ma_alignment_score
            }
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'low'}

    def analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze recent price action patterns"""
        try:
            if len(df) < 10:
                return {'pattern': 'insufficient_data', 'strength': 0}
            
            recent_10 = df.tail(10)
            latest = df.iloc[-1]
            
            # Candlestick patterns
            body_size = abs(latest['close'] - latest['open']) / latest['open']
            upper_shadow = latest['high'] - max(latest['close'], latest['open'])
            lower_shadow = min(latest['close'], latest['open']) - latest['low']
            
            # Pattern identification
            patterns = []
            
            # Doji-like
            if body_size < 0.001:
                patterns.append('doji')
            
            # Hammer/Shooting star
            if lower_shadow > body_size * 2 and upper_shadow < body_size:
                patterns.append('hammer')
            elif upper_shadow > body_size * 2 and lower_shadow < body_size:
                patterns.append('shooting_star')
            
            # Support/Resistance test
            recent_lows = recent_10['low'].min()
            recent_highs = recent_10['high'].max()
            
            if latest['low'] <= recent_lows * 1.001:
                patterns.append('support_test')
            if latest['high'] >= recent_highs * 0.999:
                patterns.append('resistance_test')
            
            # Price momentum
            closes = recent_10['close'].values
            momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            
            return {
                'patterns': patterns,
                'momentum': momentum,
                'body_size': body_size,
                'shadow_ratio': (upper_shadow + lower_shadow) / body_size if body_size > 0 else 0,
                'strength': min(1.0, abs(momentum) * 10 + body_size * 50)
            }
            
        except Exception as e:
            self.logger.error(f"Price action analysis error: {e}")
            return {'pattern': 'unknown', 'strength': 0.5}

    def assess_market_conditions(self, df: pd.DataFrame, symbol_data: Dict) -> Dict:
        """Assess overall market conditions for the symbol"""
        try:
            latest = df.iloc[-1]
            
            # Liquidity assessment
            volume_24h = symbol_data.get('volume_24h', 0)
            price_change_24h = symbol_data.get('price_change_24h', 0)
            
            # Market cap and volume relationship
            if volume_24h > 10_000_000:
                liquidity = 'high'
            elif volume_24h > 1_000_000:
                liquidity = 'medium'
            else:
                liquidity = 'low'
            
            # Volatility assessment
            atr_pct = latest.get('atr', latest['close'] * 0.02) / latest['close']
            if atr_pct > 0.06:
                volatility_level = 'high'
            elif atr_pct > 0.03:
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
            
            # Market sentiment
            if price_change_24h > 5:
                sentiment = 'very_bullish'
            elif price_change_24h > 2:
                sentiment = 'bullish'
            elif price_change_24h < -5:
                sentiment = 'very_bearish'
            elif price_change_24h < -2:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            # Trading conditions
            conditions = []
            if liquidity == 'high':
                conditions.append('good_liquidity')
            if volatility_level in ['medium', 'high']:
                conditions.append('sufficient_volatility')
            if abs(price_change_24h) > 1:
                conditions.append('active_movement')
            
            return {
                'liquidity': liquidity,
                'volatility_level': volatility_level,
                'sentiment': sentiment,
                'price_change_24h': price_change_24h,
                'volume_24h': volume_24h,
                'trading_conditions': conditions,
                'favorable_for_trading': len(conditions) >= 2
            }
            
        except Exception as e:
            self.logger.error(f"Market conditions assessment error: {e}")
            return {'liquidity': 'unknown', 'volatility_level': 'unknown', 'sentiment': 'neutral'}

    def filter_signals_by_quality(self, signals: List[Dict], max_signals: int = 20) -> List[Dict]:
        """Filter signals by quality metrics with balanced approach"""
        try:
            if not signals:
                return []
            
            # Sort by confidence and risk-reward
            quality_signals = []
            
            for signal in signals:
                confidence = signal.get('confidence', 0)
                rr_ratio = signal.get('risk_reward_ratio', 0)
                volume_24h = signal.get('volume_24h', 0)
                
                # Quality score calculation (more accessible)
                quality_score = (
                    confidence * 0.4 +  # 40% weight on confidence
                    min(100, rr_ratio * 20) * 0.3 +  # 30% weight on R/R (capped)
                    min(100, volume_24h / 1_000_000) * 0.2 +  # 20% weight on volume
                    (100 if signal.get('order_type') == 'market' else 90) * 0.1  # 10% weight on execution
                )
                
                signal['quality_score'] = quality_score
                
                # Quality filters (more lenient)
                if (confidence >= 45 and rr_ratio >= 1.8 and volume_24h >= 500_000):
                    quality_signals.append(signal)
            
            # Sort by quality score
            quality_signals.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # Return top signals
            return quality_signals[:max_signals]
            
        except Exception as e:
            self.logger.error(f"Signal filtering error: {e}")
            return signals[:max_signals]

    def generate_trading_summary(self, opportunities: List[Dict]) -> Dict:
        """Generate comprehensive trading summary"""
        try:
            if not opportunities:
                return {
                    'total_opportunities': 0,
                    'message': 'No trading opportunities found'
                }
            
            # Basic statistics
            total_ops = len(opportunities)
            buy_signals = len([op for op in opportunities if op['side'] == 'buy'])
            sell_signals = len([op for op in opportunities if op['side'] == 'sell'])
            
            # Confidence distribution
            high_confidence = len([op for op in opportunities if op['confidence'] >= 70])
            medium_confidence = len([op for op in opportunities if 50 <= op['confidence'] < 70])
            low_confidence = len([op for op in opportunities if op['confidence'] < 50])
            
            # Risk-reward distribution
            excellent_rr = len([op for op in opportunities if op.get('risk_reward_ratio', 0) >= 3.0])
            good_rr = len([op for op in opportunities if 2.0 <= op.get('risk_reward_ratio', 0) < 3.0])
            fair_rr = len([op for op in opportunities if op.get('risk_reward_ratio', 0) < 2.0])
            
            # Market sentiment analysis
            symbols_analyzed = len(set(op['symbol'] for op in opportunities))
            avg_confidence = sum(op['confidence'] for op in opportunities) / total_ops
            avg_rr = sum(op.get('risk_reward_ratio', 0) for op in opportunities) / total_ops
            
            # MTF analysis
            mtf_confirmed = len([op for op in opportunities if op.get('mtf_status') in ['STRONG', 'PARTIAL']])
            mtf_percentage = (mtf_confirmed / total_ops * 100) if total_ops > 0 else 0
            
            return {
                'total_opportunities': total_ops,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'confidence_distribution': {
                    'high': high_confidence,
                    'medium': medium_confidence,
                    'low': low_confidence
                },
                'risk_reward_distribution': {
                    'excellent': excellent_rr,
                    'good': good_rr,
                    'fair': fair_rr
                },
                'averages': {
                    'confidence': round(avg_confidence, 1),
                    'risk_reward_ratio': round(avg_rr, 2)
                },
                'mtf_confirmation': {
                    'confirmed_signals': mtf_confirmed,
                    'confirmation_rate': round(mtf_percentage, 1)
                },
                'symbols_with_signals': symbols_analyzed,
                'recommendation': self.get_trading_recommendation(opportunities)
            }
            
        except Exception as e:
            self.logger.error(f"Trading summary generation error: {e}")
            return {'total_opportunities': len(opportunities), 'error': str(e)}

    def get_trading_recommendation(self, opportunities: List[Dict]) -> str:
        """Generate trading recommendation based on signal quality"""
        try:
            if not opportunities:
                return "No signals found. Wait for better market conditions."
            
            total_ops = len(opportunities)
            high_quality = len([op for op in opportunities 
                              if op['confidence'] >= 65 and op.get('risk_reward_ratio', 0) >= 2.5])
            
            if high_quality >= 3:
                return f"Excellent conditions: {high_quality} high-quality signals available."
            elif high_quality >= 1:
                return f"Good conditions: {high_quality} quality signal(s) found."
            elif total_ops >= 5:
                return f"Fair conditions: {total_ops} signals available, use careful position sizing."
            else:
                return f"Limited opportunities: Only {total_ops} signal(s) found."
                
        except Exception:
            return "Review signals carefully before trading."
        
    def get_signal_explanation(self, signal: Dict) -> str:
        """Get human-readable explanation of why signal was generated"""
        if not signal:
            return "No signal generated"
        
        side = signal['side'].upper()
        symbol = signal['symbol']
        confidence = signal['confidence']
        
        explanation = f"{side} signal for {symbol} ({confidence:.1f}% confidence)\n"
        
        if side == 'BUY':
            explanation += "Reasons: Oversold conditions, price near support, bounce potential"
        else:
            explanation += "Reasons: Overbought conditions, price near resistance, reversal potential"
        
        return explanation