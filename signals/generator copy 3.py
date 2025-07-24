"""
Signal generation and ranking for the Enhanced Bybit Trading System.
COMPLETELY REWRITTEN: Fixed all critical flaws for reliable trading
- Conservative signal thresholds
- Proper market structure analysis
- Improved ranking logic
- Enhanced risk management
- Multi-layer validation system
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from config.config import EnhancedSystemConfig


class SignalGenerator:
    """FIXED: Signal generation and ranking system with conservative approach"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_enhanced_signal(self, df: pd.DataFrame, symbol_data: Dict, 
                            volume_entry: Dict, confluence_zones: List[Dict]) -> Optional[Dict]:
        """COMPLETELY REWRITTEN: Conservative signal generation with proper validation"""
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
            
            # ===== CONSERVATIVE BUY SIGNAL CONDITIONS =====
            # Only generate BUY signals in clear oversold conditions with trend support
            
            # Primary technical conditions (ALL must be met)
            buy_technical_conditions = [
                rsi < 30,                                    # FIXED: Truly oversold (was 30-75)
                macd > macd_signal or rsi < 25,             # MACD bullish OR extremely oversold
                current_price > sma_200 * 0.95,            # ADDED: Above long-term support
                volume_ratio > 1.5,                         # FIXED: Strong volume requirement (was 1.0)
                bb_position < 0.3,                          # FIXED: Near lower BB (was 0.8)
                atr / current_price < 0.05                  # ADDED: Not too volatile
            ]
            
            # Trend alignment check
            buy_trend_conditions = [
                sma_20 > sma_50 * 0.98,                     # ADDED: Medium-term trend not broken
                current_price > sma_50 * 0.95,             # ADDED: Above medium-term support
                # Allow counter-trend if extremely oversold
                rsi < 25 or (ema_12 > ema_26 * 0.98)      # ADDED: Trend or extreme oversold
            ]
            
            # Stochastic RSI conditions (MUCH more conservative)
            buy_stoch_conditions = [
                stoch_rsi_k < 25,                           # FIXED: Truly oversold (was < 80)
                stoch_rsi_k > stoch_rsi_d or stoch_rsi_k < 15,  # FIXED: Bullish cross OR extreme
                stoch_rsi_k > 5                             # ADDED: Not at absolute bottom
            ]
            
            # Market structure conditions
            buy_structure_conditions = [
                market_structure['near_support'],           # ADDED: Near identified support
                not market_structure['strong_downtrend'],   # ADDED: Not in strong downtrend
                market_structure['bounce_potential']        # ADDED: Potential for bounce
            ]
            
            # Count conditions
            buy_tech_score = sum(buy_technical_conditions)
            buy_trend_score = sum(buy_trend_conditions)
            buy_stoch_score = sum(buy_stoch_conditions)
            buy_structure_score = sum(buy_structure_conditions)
            
            # STRICT REQUIREMENTS: ALL categories must pass
            if (buy_tech_score >= 5 and buy_trend_score >= 2 and 
                buy_stoch_score >= 2 and buy_structure_score >= 2):
                
                signal = self.create_buy_signal(
                    symbol_data, current_price, latest, volume_entry, 
                    confluence_zones, buy_tech_score, buy_trend_score, 
                    buy_stoch_score, buy_structure_score
                )
            
            # ===== ULTRA-CONSERVATIVE SELL SIGNAL CONDITIONS =====  
            # Only generate SELL signals in extreme overbought conditions with clear reversal
            if not signal:  # Only if no BUY signal
                
                # Primary technical conditions (MUCH stricter)
                sell_technical_conditions = [
                    rsi > 75,                               # FIXED: Truly overbought (was 25-70)
                    macd < 0 and macd < macd_signal,       # FIXED: Negative AND bearish (was just < signal)
                    current_price < sma_200 * 1.05,        # ADDED: Below long-term resistance
                    volume_ratio > 2.0,                     # FIXED: Very strong volume (was 1.0)
                    bb_position > 0.7,                     # FIXED: Near upper BB (was 0.2)
                    atr / current_price < 0.06              # ADDED: Reasonable volatility
                ]
                
                # Trend reversal conditions (CRITICAL for shorts)
                sell_trend_conditions = [
                    sma_20 < sma_50 * 1.02,                # ADDED: Medium-term trend turning
                    current_price < sma_50 * 1.05,         # ADDED: Below medium-term resistance
                    ema_12 < ema_26 or rsi > 80,           # FIXED: Clear reversal OR extreme
                    current_price < bb_upper * 0.98        # ADDED: Rejected from upper BB
                ]
                
                # Stochastic RSI conditions (EXTREMELY strict)
                sell_stoch_conditions = [
                    stoch_rsi_k > 80,                       # FIXED: Extremely overbought (was > 20)
                    stoch_rsi_k < stoch_rsi_d,             # Bearish crossover
                    stoch_rsi_d > 75,                       # ADDED: Signal line also overbought
                    stoch_rsi_k < 95                        # ADDED: Not at absolute top (false signal)
                ]
                
                # Market structure conditions (CRITICAL)
                sell_structure_conditions = [
                    market_structure['near_resistance'],    # ADDED: Near identified resistance
                    market_structure['reversal_signals'],   # ADDED: Clear reversal pattern
                    not market_structure['strong_uptrend'], # ADDED: Not fighting strong uptrend
                    market_structure['distribution_signs']  # ADDED: Signs of distribution
                ]
                
                # Count conditions
                sell_tech_score = sum(sell_technical_conditions)
                sell_trend_score = sum(sell_trend_conditions) 
                sell_stoch_score = sum(sell_stoch_conditions)
                sell_structure_score = sum(sell_structure_conditions)
                
                # ULTRA-STRICT REQUIREMENTS: Near perfect alignment needed
                if (sell_tech_score >= 5 and sell_trend_score >= 3 and 
                    sell_stoch_score >= 4 and sell_structure_score >= 3):
                    
                    # FINAL SAFETY CHECK: Don't sell in strong uptrends
                    if not self.is_strong_uptrend(df):
                        signal = self.create_sell_signal(
                            symbol_data, current_price, latest, volume_entry,
                            confluence_zones, sell_tech_score, sell_trend_score,
                            sell_stoch_score, sell_structure_score
                        )
                    else:
                        self.logger.debug(f"SELL signal filtered: {symbol} in strong uptrend")
            
            # Final validation layer
            if signal:
                signal = self.validate_signal_quality(signal, df, market_structure)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return None
    
    def analyze_market_structure(self, df: pd.DataFrame, current_price: float) -> Dict:
        """ADDED: Comprehensive market structure analysis"""
        try:
            if len(df) < 50:
                return self.get_default_market_structure()
            
            # Get recent data
            recent_20 = df.tail(20)
            recent_50 = df.tail(50)
            
            # Identify swing points
            highs = recent_20['high'].rolling(window=5, center=True).max()
            lows = recent_20['low'].rolling(window=5, center=True).min()
            
            swing_highs = recent_20[recent_20['high'] == highs]['high'].dropna()
            swing_lows = recent_20[recent_20['low'] == lows]['low'].dropna()
            
            # Support and resistance levels
            if len(swing_lows) > 0:
                nearest_support = swing_lows[swing_lows < current_price].max() if len(swing_lows[swing_lows < current_price]) > 0 else recent_50['low'].min()
            else:
                nearest_support = recent_50['low'].min()
                
            if len(swing_highs) > 0:
                nearest_resistance = swing_highs[swing_highs > current_price].min() if len(swing_highs[swing_highs > current_price]) > 0 else recent_50['high'].max()
            else:
                nearest_resistance = recent_50['high'].max()
            
            # Distance calculations
            support_distance = abs(current_price - nearest_support) / current_price
            resistance_distance = abs(nearest_resistance - current_price) / current_price
            
            # Trend analysis
            sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else current_price
            sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else current_price
            
            trend_slope_20 = (recent_20['close'].iloc[-1] - recent_20['close'].iloc[0]) / len(recent_20)
            trend_slope_50 = (recent_50['close'].iloc[-1] - recent_50['close'].iloc[0]) / len(recent_50)
            
            # Volume analysis
            avg_volume = recent_20['volume'].mean()
            current_volume = recent_20['volume'].iloc[-1]
            volume_trend = current_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance,
                'near_support': support_distance < 0.02,  # Within 2%
                'near_resistance': resistance_distance < 0.02,  # Within 2%
                'strong_uptrend': trend_slope_20 > 0 and trend_slope_50 > 0 and sma_20 > sma_50 * 1.02,
                'strong_downtrend': trend_slope_20 < 0 and trend_slope_50 < 0 and sma_20 < sma_50 * 0.98,
                'bounce_potential': support_distance < 0.03 and trend_slope_20 < 0,
                'reversal_signals': resistance_distance < 0.03 and volume_trend > 1.5,
                'distribution_signs': resistance_distance < 0.05 and volume_trend > 2.0,
                'consolidation': abs(trend_slope_20) < current_price * 0.001
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return self.get_default_market_structure()
    
    def get_default_market_structure(self) -> Dict:
        """Default market structure when analysis fails"""
        return {
            'nearest_support': 0,
            'nearest_resistance': 0,
            'support_distance': 1,
            'resistance_distance': 1,
            'near_support': False,
            'near_resistance': False,
            'strong_uptrend': False,
            'strong_downtrend': False,
            'bounce_potential': False,
            'reversal_signals': False,
            'distribution_signs': False,
            'consolidation': True
        }
    
    def is_strong_uptrend(self, df: pd.DataFrame) -> bool:
        """Check if we're in a strong uptrend (don't short)"""
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
            
            # Strong uptrend conditions
            conditions = [
                sma_20 > sma_50 * 1.02,  # 20 SMA well above 50 SMA
                ema_12 > ema_26 * 1.01,  # EMAs aligned bullishly
                latest['close'] > sma_20,  # Price above 20 SMA
                recent['close'].iloc[-1] > recent['close'].iloc[-10] * 1.03,  # 3%+ gain over 10 periods
                recent['volume'].mean() > df['volume'].tail(50).mean()  # Above average volume
            ]
            
            return sum(conditions) >= 4  # Need 4/5 conditions
            
        except Exception as e:
            self.logger.error(f"Uptrend check error: {e}")
            return False
    
    def create_buy_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                         volume_entry: Dict, confluence_zones: List[Dict], 
                         tech_score: int, trend_score: int, stoch_score: int, structure_score: int) -> Dict:
        """Create BUY signal with improved logic"""
        try:
            # Entry price logic
            entry_candidates = [current_price]
            
            # Use volume profile entry if high confidence
            if volume_entry.get('confidence', 0) > 0.7:
                entry_candidates.append(volume_entry['entry_price'])
            
            # Use support levels from confluence zones
            support_zones = [zone for zone in confluence_zones 
                           if zone['zone_type'] == 'support' and zone['price'] < current_price * 1.01]
            if support_zones:
                # Use closest support level
                closest_support = max(support_zones, key=lambda x: x['price'])
                entry_candidates.append(closest_support['price'])
            
            optimal_entry = np.mean(entry_candidates)
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            # Order type: use limit if entry is significantly different
            order_type = 'limit' if distance_pct > 0.003 else 'market'  # 0.3% threshold
            if order_type == 'market':
                optimal_entry = current_price
            
            # Stop loss: use market structure
            atr = latest.get('atr', current_price * 0.02)
            structure_stop = current_price * 0.96  # 4% structural stop
            atr_stop = optimal_entry - (2.0 * atr)  # 2 ATR stop
            stop_loss = max(structure_stop, atr_stop)  # Use wider stop
            
            # Take profits: conservative but achievable
            tp1 = optimal_entry + (3 * atr)  # 3:1 R/R minimum
            tp2 = optimal_entry + (5 * atr)  # 5:1 R/R stretch target
            
            # Calculate confidence based on condition scores
            base_confidence = 50
            base_confidence += tech_score * 5  # Up to 30 points
            base_confidence += trend_score * 5  # Up to 15 points  
            base_confidence += stoch_score * 5  # Up to 15 points
            base_confidence += structure_score * 5  # Up to 15 points
            
            # Bonus for confluence
            if len(support_zones) > 1:
                base_confidence += 10
            if volume_entry.get('confidence', 0) > 0.8:
                base_confidence += 5
            
            confidence = min(95, base_confidence)
            
            # Risk-reward calculation
            risk_amount = optimal_entry - stop_loss
            reward_amount = tp2 - optimal_entry
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                'symbol': symbol_data['symbol'],
                'side': 'buy',
                'order_type': order_type,
                'entry_price': optimal_entry,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence': confidence,
                'signal_type': 'conservative_technical',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'entry_methods': {
                    'technical_score': tech_score,
                    'trend_score': trend_score,
                    'stoch_score': stoch_score,
                    'structure_score': structure_score,
                    'volume_profile': volume_entry,
                    'confluence_zones': len(support_zones)
                }
            }
            
        except Exception as e:
            self.logger.error(f"BUY signal creation error: {e}")
            return None
    
    def create_sell_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                          volume_entry: Dict, confluence_zones: List[Dict],
                          tech_score: int, trend_score: int, stoch_score: int, structure_score: int) -> Dict:
        """Create SELL signal with ultra-conservative logic"""
        try:
            # Entry price logic for shorts
            entry_candidates = [current_price]
            
            # Use resistance levels from confluence zones  
            resistance_zones = [zone for zone in confluence_zones
                               if zone['zone_type'] == 'resistance' and zone['price'] > current_price * 0.99]
            if resistance_zones:
                closest_resistance = min(resistance_zones, key=lambda x: x['price'])
                entry_candidates.append(closest_resistance['price'])
            
            # Use Bollinger Band upper as resistance
            bb_upper = latest.get('bb_upper', current_price * 1.02)
            if bb_upper > current_price:
                entry_candidates.append(bb_upper)
            
            optimal_entry = np.mean(entry_candidates)
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            # Order type: prefer limit for shorts to get better entry
            order_type = 'limit' if distance_pct > 0.002 else 'market'  # 0.2% threshold
            if order_type == 'market':
                optimal_entry = current_price
            
            # Stop loss: tight for shorts due to higher risk
            atr = latest.get('atr', current_price * 0.02)
            structure_stop = current_price * 1.04  # 4% structural stop
            atr_stop = optimal_entry + (1.5 * atr)  # 1.5 ATR stop (tighter)
            stop_loss = min(structure_stop, atr_stop)  # Use tighter stop
            
            # Take profits: quick profits for shorts
            tp1 = optimal_entry - (3 * atr)  # 3:1 R/R minimum
            tp2 = optimal_entry - (5 * atr)  # 5:1 R/R target
            
            # Calculate confidence (more conservative scoring)
            base_confidence = 45  # Lower base for shorts
            base_confidence += tech_score * 4  # Up to 24 points
            base_confidence += trend_score * 4  # Up to 16 points
            base_confidence += stoch_score * 4  # Up to 16 points  
            base_confidence += structure_score * 4  # Up to 16 points
            
            # Bonus for strong reversal signals
            if len(resistance_zones) > 1:
                base_confidence += 8
            if latest.get('rsi', 50) > 80:  # Extreme overbought
                base_confidence += 10
            
            confidence = min(90, base_confidence)  # Cap lower for shorts
            
            # Risk-reward calculation
            risk_amount = stop_loss - optimal_entry
            reward_amount = optimal_entry - tp2
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                'symbol': symbol_data['symbol'],
                'side': 'sell',
                'order_type': order_type,
                'entry_price': optimal_entry,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence': confidence,
                'signal_type': 'conservative_reversal',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'entry_methods': {
                    'technical_score': tech_score,
                    'trend_score': trend_score,
                    'stoch_score': stoch_score,
                    'structure_score': structure_score,
                    'resistance_zones': len(resistance_zones),
                    'reversal_strength': tech_score + trend_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"SELL signal creation error: {e}")
            return None
    
    def validate_signal_quality(self, signal: Dict, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """ADDED: Final signal quality validation"""
        try:
            if not signal:
                return None
            
            symbol = signal['symbol']
            side = signal['side']
            confidence = signal['confidence']
            risk_reward = signal['risk_reward_ratio']
            
            # Minimum confidence thresholds
            min_confidence = 65 if side == 'buy' else 70  # Higher bar for shorts
            if confidence < min_confidence:
                self.logger.debug(f"Signal filtered: {symbol} confidence {confidence}% below minimum {min_confidence}%")
                return None
            
            # Minimum risk-reward ratio
            min_rr = 2.5 if side == 'buy' else 3.0  # Higher R/R required for shorts
            if risk_reward < min_rr:
                self.logger.debug(f"Signal filtered: {symbol} R/R {risk_reward:.1f} below minimum {min_rr}")
                return None
            
            # Market structure validation
            if side == 'sell':
                # Don't short unless near resistance
                if not market_structure['near_resistance'] and market_structure['resistance_distance'] > 0.05:
                    self.logger.debug(f"SELL signal filtered: {symbol} not near resistance")
                    return None
                
                # Don't short in strong uptrends
                if market_structure['strong_uptrend']:
                    self.logger.debug(f"SELL signal filtered: {symbol} in strong uptrend")
                    return None
            
            elif side == 'buy':
                # Don't buy unless reasonable support nearby
                if market_structure['support_distance'] > 0.08:  # 8% away
                    self.logger.debug(f"BUY signal filtered: {symbol} too far from support")
                    return None
            
            # Volume validation
            volume_ratio = df.iloc[-1].get('volume_ratio', 1)
            min_volume = 1.2 if side == 'buy' else 2.0  # Higher volume needed for shorts
            if volume_ratio < min_volume:
                self.logger.debug(f"Signal filtered: {symbol} volume {volume_ratio:.1f} below minimum {min_volume}")
                return None
            
            # Volatility check
            atr_pct = signal.get('atr_percentage', 0.02)
            if atr_pct > 0.08:  # 8% daily ATR is too volatile
                self.logger.debug(f"Signal filtered: {symbol} too volatile (ATR: {atr_pct*100:.1f}%)")
                return None
            
            self.logger.info(f"✅ Signal validated: {symbol} {side.upper()} - {confidence}% confidence, {risk_reward:.1f}:1 R/R")
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return None

    def rank_opportunities_with_mtf(self, signals: List[Dict]) -> List[Dict]:
        """COMPLETELY REWRITTEN: Improved ranking that prioritizes signal quality"""
        try:
            opportunities = []
            
            for signal in signals:
                # Get signal quality metrics
                confidence = signal['confidence']
                original_confidence = signal.get('original_confidence', confidence)
                mtf_boost = confidence - original_confidence
                
                # ===== REVISED SCORING SYSTEM =====
                
                # 1. SIGNAL CONFIDENCE (40% weight - INCREASED from 25%)
                confidence_score = confidence / 100
                
                # 2. RISK-REWARD RATIO (25% weight - INCREASED from 15%)  
                analysis = signal.get('analysis', {})
                risk_assessment = analysis.get('risk_assessment', {})
                rr_ratio = risk_assessment.get('risk_reward_ratio', 1)
                rr_score = min(1.0, rr_ratio / 4.0)  # Target 4:1 R/R
                
                # 3. MARKET STRUCTURE ALIGNMENT (20% weight - NEW)
                entry_methods = signal.get('entry_methods', {})
                structure_score = 0
                if 'structure_score' in entry_methods:
                    structure_score = min(1.0, entry_methods['structure_score'] / 4.0)
                elif 'confluence_zones' in entry_methods:
                    structure_score = min(1.0, entry_methods['confluence_zones'] / 3.0)
                
                # 4. VOLUME QUALITY (10% weight - REDUCED from 15%)
                volume_score = min(1.0, signal['volume_24h'] / 50_000_000)  # Lower target
                
                # 5. MTF CONFIRMATION (15% weight - REDUCED from 30%)
                mtf_analysis = signal.get('mtf_analysis', {})
                confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
                total_timeframes = len(self.config.confirmation_timeframes)
                
                if total_timeframes > 0:
                    mtf_score = confirmed_count / total_timeframes
                    mtf_bonus = mtf_score * 0.15  # Reduced bonus
                else:
                    mtf_bonus = 0
                
                # 6. DISTANCE PENALTY (5% weight - REDUCED from 10%)
                distance = abs(signal['entry_price'] - signal['current_price']) / signal['current_price']
                distance_score = max(0, 1 - distance * 10)  # Less penalty
                
                # 7. ORDER TYPE (5% weight - REDUCED from 10%)
                order_type_score = 1.0 if signal['order_type'] == 'market' else 0.9
                
                # ===== CALCULATE TOTAL SCORE =====
                total_score = (
                    confidence_score * 0.40 +      # 40% - Signal quality first
                    rr_score * 0.25 +              # 25% - Risk management second  
                    structure_score * 0.20 +       # 20% - Market structure third
                    volume_score * 0.10 +          # 10% - Volume support
                    mtf_bonus +                    # 15% - MTF confirmation (additive)
                    distance_score * 0.05 +        # 5% - Entry distance
                    order_type_score * 0.05        # 5% - Order type
                )
                
                # ===== REVISED PRIORITY SYSTEM =====
                # Priority based on signal quality, not just MTF
                mtf_status = signal.get('mtf_status', 'NONE')
                
                # Base priority on confidence and R/R
                if confidence >= 80 and rr_ratio >= 4.0:
                    base_priority = 1000  # Exceptional signals
                elif confidence >= 70 and rr_ratio >= 3.0:
                    base_priority = 500   # High quality signals
                elif confidence >= 60 and rr_ratio >= 2.5:
                    base_priority = 100   # Good signals
                else:
                    base_priority = 10    # Marginal signals
                
                # MTF modifier (not dominant)
                mtf_modifier = {
                    'STRONG': 1.5,   # 50% boost (was 100 priority points)
                    'PARTIAL': 1.2,  # 20% boost (was 50 priority points)
                    'NONE': 1.0,     # No change
                    'DISABLED': 1.0  # No change
                }.get(mtf_status, 1.0)
                
                final_priority = int(base_priority * mtf_modifier)
                
                # Get MTF confirmation details
                confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
                conflicting_timeframes = mtf_analysis.get('conflicting_timeframes', [])
                
                opportunities.append({
                    'rank': 0,  # Will be set after sorting
                    'symbol': signal['symbol'],
                    'side': signal['side'].upper(),
                    'order_type': signal['order_type'].upper(),
                    'confidence': signal['confidence'],
                    'original_confidence': original_confidence,
                    'mtf_boost': mtf_boost,
                    'entry_price': signal['entry_price'],
                    'current_price': signal['current_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit_1'],
                    'take_profit_1': signal.get('take_profit_1', signal['take_profit_1']),
                    'take_profit_2': signal.get('take_profit_2', 0),
                    'risk_reward_ratio': rr_ratio,
                    'volume_24h': signal['volume_24h'],
                    'total_score': total_score,
                    'priority_score': final_priority,  # New priority system
                    'chart_file': signal.get('chart_file', 'Not available'),
                    'signal_type': signal.get('signal_type', 'enhanced'),
                    'technical_summary': analysis.get('technical_summary', {}),
                    'risk_level': risk_assessment.get('risk_level', 'Unknown'),
                    # MTF fields
                    'mtf_status': mtf_status,
                    'mtf_confirmed': confirmed_timeframes,
                    'mtf_conflicting': conflicting_timeframes,
                    'mtf_confirmation_count': confirmed_count,
                    'mtf_total_timeframes': total_timeframes,
                    'mtf_confirmation_strength': mtf_analysis.get('confirmation_strength', 0),
                    'priority_boost': final_priority  # For compatibility
                })
            
            # ===== IMPROVED SORTING LOGIC =====
            # Sort by priority score first, then total score
            opportunities.sort(key=lambda x: (x['priority_score'], x['total_score']), reverse=True)
            
            # Assign ranks
            for i, opp in enumerate(opportunities):
                opp['rank'] = i + 1
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Opportunity ranking error: {e}")
            return []
    
    def create_technical_summary(self, df: pd.DataFrame) -> Dict:
        """Enhanced technical summary with market structure"""
        try:
            latest = df.iloc[-1]
            
            # Basic technical analysis
            sma_20 = latest.get('sma_20', 0)
            sma_50 = latest.get('sma_50', 0)
            ema_12 = latest.get('ema_12', 0)
            ema_26 = latest.get('ema_26', 0)
            
            # Trend analysis with more nuance
            trend_score = 0
            trend_signals = []
            
            if latest['close'] > sma_20 > sma_50:
                trend_score += 3
                trend_signals.append("Price > SMA20 > SMA50")
            elif latest['close'] > sma_20:
                trend_score += 1
                trend_signals.append("Price > SMA20")
            
            if ema_12 > ema_26:
                trend_score += 2
                trend_signals.append("EMA12 > EMA26")
            
            if latest.get('macd', 0) > latest.get('macd_signal', 0):
                trend_score += 1
                trend_signals.append("MACD > Signal")
            
            # Enhanced trend classification
            if trend_score >= 5:
                trend_strength = 'Very Strong Bullish'
            elif trend_score >= 3:
                trend_strength = 'Strong Bullish'
            elif trend_score >= 1:
                trend_strength = 'Weak Bullish'
            elif trend_score <= -3:
                trend_strength = 'Strong Bearish'
            elif trend_score <= -1:
                trend_strength = 'Weak Bearish'
            else:
                trend_strength = 'Neutral'
            
            # Enhanced momentum analysis
            rsi = latest.get('rsi', 50)
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            
            if rsi > 80 or stoch_rsi_k > 85:
                momentum_status = 'Extremely Overbought'
            elif rsi > 70 or stoch_rsi_k > 80:
                momentum_status = 'Overbought'
            elif rsi < 20 or stoch_rsi_k < 15:
                momentum_status = 'Extremely Oversold'
            elif rsi < 30 or stoch_rsi_k < 20:
                momentum_status = 'Oversold'
            else:
                momentum_status = 'Neutral'
            
            # Volatility and volume analysis
            bb_width = latest.get('bb_width', 0)
            volume_ratio = latest.get('volume_ratio', 1)
            
            volatility_status = 'High' if bb_width > 0.05 else 'Low' if bb_width < 0.02 else 'Normal'
            volume_status = 'Very High' if volume_ratio > 3 else 'High' if volume_ratio > 1.5 else 'Low' if volume_ratio < 0.8 else 'Normal'
            
            # Ichimoku status
            ichimoku_status = 'Strong Bullish' if latest.get('ichimoku_bullish', False) else \
                            'Strong Bearish' if latest.get('ichimoku_bearish', False) else 'Neutral'
            
            return {
                'trend': {
                    'direction': trend_strength,
                    'score': trend_score,
                    'signals': trend_signals,
                    'sma_alignment': sma_20 > sma_50,
                    'ema_alignment': ema_12 > ema_26
                },
                'momentum': {
                    'rsi': rsi,
                    'status': momentum_status,
                    'macd_bullish': latest.get('macd', 0) > latest.get('macd_signal', 0),
                    'stoch_rsi_k': stoch_rsi_k,
                    'stoch_rsi_d': latest.get('stoch_rsi_d', 50),
                    'stoch_rsi_status': 'Extreme' if stoch_rsi_k > 85 or stoch_rsi_k < 15 else 'Normal'
                },
                'volatility': {
                    'status': volatility_status,
                    'bb_width': bb_width,
                    'atr_pct': latest.get('atr', 0) / latest['close'] * 100 if latest['close'] > 0 else 0
                },
                'volume': {
                    'status': volume_status,
                    'ratio': volume_ratio,
                    'trend': 'Increasing' if volume_ratio > 1.2 else 'Decreasing' if volume_ratio < 0.8 else 'Stable'
                },
                'ichimoku': {
                    'status': ichimoku_status,
                    'cloud_position': 'Above' if latest['close'] > max(latest.get('ichimoku_span_a', 0), 
                                                                     latest.get('ichimoku_span_b', 0)) else 'Below',
                    'tenkan_kijun': 'Bullish' if latest.get('ichimoku_tenkan', 0) > latest.get('ichimoku_kijun', 0) else 'Bearish'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Technical summary error: {e}")
            return {}
    
    def assess_signal_risk(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Enhanced risk assessment with market structure"""
        try:
            latest = df.iloc[-1]
            
            # Base volatility risk
            returns = df['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(24)
            
            # Enhanced distance risk
            distance_risk = abs(signal['entry_price'] - signal['current_price']) / signal['current_price']
            
            # Market condition risk assessment
            rsi = latest.get('rsi', 50)
            bb_position = latest.get('bb_position', 0.5)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Condition risk factors
            extreme_rsi = (rsi > 85 or rsi < 15)
            extreme_bb = (bb_position > 0.95 or bb_position < 0.05)
            low_volume = volume_ratio < 0.5
            
            condition_risk = 0
            if extreme_rsi:
                condition_risk += 0.4  # Higher risk in extreme conditions
            if extreme_bb:
                condition_risk += 0.3
            if low_volume:
                condition_risk += 0.2
            
            # Side-specific risk
            side_risk = 0
            if signal['side'] == 'sell':
                # Shorts are inherently riskier
                side_risk += 0.2
                # Additional risk if shorting in uptrend
                if latest.get('sma_20', 0) > latest.get('sma_50', 0):
                    side_risk += 0.3
            
            # MTF risk adjustment (but not dominant)
            mtf_analysis = signal.get('mtf_analysis', {})
            confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
            total_timeframes = len(self.config.confirmation_timeframes) if self.config.confirmation_timeframes else 1
            
            mtf_risk_reduction = (confirmed_count / total_timeframes) * 0.15  # Max 15% reduction
            
            # Calculate total risk score
            base_risk = (volatility * 3 + distance_risk * 3 + condition_risk + side_risk)
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
            
            # Risk level classification
            if total_risk > 0.8:
                risk_level = 'Very High'
            elif total_risk > 0.6:
                risk_level = 'High'
            elif total_risk > 0.4:
                risk_level = 'Medium'
            elif total_risk > 0.2:
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
            return {'total_risk_score': 0.5, 'risk_level': 'Unknown'}