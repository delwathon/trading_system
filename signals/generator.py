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
        """Enhanced signal generation with balanced validation"""
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
            
            # ===== BALANCED BUY SIGNAL CONDITIONS =====
            # Generate BUY signals in oversold conditions with reasonable trend support
            
            # Primary technical conditions (more flexible)
            buy_technical_conditions = [
                rsi < 40,                                    # Oversold but not extreme (was 30)
                macd > macd_signal or rsi < 35,             # MACD bullish OR oversold
                current_price > sma_200 * 0.90,            # Above long-term support (was 0.95)
                volume_ratio > 1.2,                         # Good volume requirement (was 1.5)
                bb_position < 0.4,                          # Near lower BB (was 0.3)
                atr / current_price < 0.06                  # Reasonable volatility (was 0.05)
            ]
            
            # Trend alignment check (more flexible)
            buy_trend_conditions = [
                sma_20 > sma_50 * 0.95,                     # Medium-term trend not severely broken (was 0.98)
                current_price > sma_50 * 0.92,             # Above medium-term support (was 0.95)
                # Allow counter-trend if oversold
                rsi < 35 or (ema_12 > ema_26 * 0.96)      # Trend or oversold (was 0.98)
            ]
            
            # Stochastic RSI conditions (more flexible)
            # buy_stoch_conditions = [
            #     stoch_rsi_k < 35,                           # Oversold but not extreme (was 25)
            #     stoch_rsi_k > stoch_rsi_d or stoch_rsi_k < 20,  # Bullish cross OR extreme
            #     stoch_rsi_k > 3                             # Not at absolute bottom (was 5)
            # ]
            buy_stoch_conditions = [
                stoch_rsi_k < 35,                           # Oversold
                stoch_rsi_k > stoch_rsi_d,                  # FIXED: Only bullish crossover for BUY
                stoch_rsi_k > 3                             # Not at absolute bottom
            ]
            
            # Market structure conditions (more flexible)
            buy_structure_conditions = [
                market_structure['near_support'] or market_structure['support_distance'] < 0.12,  # Near support or reasonable distance
                not market_structure['strong_downtrend'],   # Not in strong downtrend
                market_structure['bounce_potential'] or rsi < 35  # Potential for bounce OR oversold
            ]
            
            # Count conditions
            buy_tech_score = sum(buy_technical_conditions)
            buy_trend_score = sum(buy_trend_conditions)
            buy_stoch_score = sum(buy_stoch_conditions)
            buy_structure_score = sum(buy_structure_conditions)
            
            # RELAXED REQUIREMENTS: Allow more signals
            if (buy_tech_score >= 4 and buy_trend_score >= 1 and 
                buy_stoch_score >= 1 and buy_structure_score >= 1):
                
                signal = self.create_buy_signal(
                    symbol_data, current_price, latest, volume_entry, 
                    confluence_zones, buy_tech_score, buy_trend_score, 
                    buy_stoch_score, buy_structure_score
                )
            
            # ===== BALANCED SELL SIGNAL CONDITIONS =====  
            # Generate SELL signals in overbought conditions with reasonable reversal
            if not signal:  # Only if no BUY signal
                
                # Primary technical conditions (more flexible)
                sell_technical_conditions = [
                    rsi > 65,                               # Overbought but not extreme (was 75)
                    macd < macd_signal or rsi > 70,        # MACD bearish OR overbought (more flexible)
                    current_price < sma_200 * 1.10,        # Below long-term resistance (was 1.05)
                    volume_ratio > 1.5,                     # Strong volume (was 2.0)
                    bb_position > 0.6,                     # Near upper BB (was 0.7)
                    atr / current_price < 0.07              # Reasonable volatility (was 0.06)
                ]
                
                # Trend reversal conditions (more flexible)
                sell_trend_conditions = [
                    sma_20 < sma_50 * 1.05,                # Medium-term trend turning (was 1.02)
                    current_price < sma_50 * 1.08,         # Below medium-term resistance (was 1.05)
                    ema_12 < ema_26 or rsi > 75,           # Clear reversal OR extreme (was 80)
                    current_price < bb_upper * 0.96        # Rejected from upper BB (was 0.98)
                ]
                
                # Stochastic RSI conditions (more flexible)
                # sell_stoch_conditions = [
                #     stoch_rsi_k > 70,                       # Overbought but not extreme (was 80)
                #     stoch_rsi_k < stoch_rsi_d or stoch_rsi_k > 85,  # Bearish crossover OR extreme
                #     stoch_rsi_d > 65,                       # Signal line also overbought (was 75)
                #     stoch_rsi_k < 97                        # Not at absolute top (was 95)
                # ]
                sell_stoch_conditions = [
                    stoch_rsi_k > 70,                       # Overbought
                    stoch_rsi_k < stoch_rsi_d,             # FIXED: Only bearish crossover for SELL
                    stoch_rsi_d > 65,                       # Signal line also overbought
                    stoch_rsi_k < 97                        # Not at absolute top
                ]
                
                # Market structure conditions (more flexible)
                sell_structure_conditions = [
                    market_structure['near_resistance'] or market_structure['resistance_distance'] < 0.08,  # Near resistance or reasonable distance
                    market_structure['reversal_signals'] or rsi > 70,  # Clear reversal pattern OR overbought
                    not market_structure['strong_uptrend'] or rsi > 75,  # Not fighting strong uptrend OR extreme
                    market_structure['distribution_signs'] or bb_position > 0.7  # Distribution signs OR near upper BB
                ]
                
                # Count conditions
                sell_tech_score = sum(sell_technical_conditions)
                sell_trend_score = sum(sell_trend_conditions) 
                sell_stoch_score = sum(sell_stoch_conditions)
                sell_structure_score = sum(sell_structure_conditions)
                
                # RELAXED REQUIREMENTS: Allow more signals
                if (sell_tech_score >= 4 and sell_trend_score >= 2 and 
                    sell_stoch_score >= 2 and sell_structure_score >= 2):
                    
                    # SAFETY CHECK: Don't sell in very strong uptrends
                    if not self.is_very_strong_uptrend(df):
                        signal = self.create_sell_signal(
                            symbol_data, current_price, latest, volume_entry,
                            confluence_zones, sell_tech_score, sell_trend_score,
                            sell_stoch_score, sell_structure_score
                        )
                    else:
                        self.logger.debug(f"SELL signal filtered: {symbol} in very strong uptrend")
            
            # Final validation layer (more lenient)
            if signal:
                signal = self.validate_signal_quality(signal, df, market_structure)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return None
    
    def analyze_market_structure(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Comprehensive market structure analysis"""
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
            
            # Calculate distances
            support_distance = abs(current_price - nearest_support) / current_price
            resistance_distance = abs(nearest_resistance - current_price) / current_price
            
            # Trend analysis
            sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else current_price
            sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else current_price
            
            # Recent price momentum
            price_change_20 = (current_price - recent_20['close'].iloc[0]) / recent_20['close'].iloc[0]
            
            # Volume analysis
            avg_volume = recent_20['volume'].mean() if 'volume' in recent_20.columns else 1
            current_volume = recent_20['volume'].iloc[-1] if 'volume' in recent_20.columns else 1
            volume_surge = current_volume > avg_volume * 1.5
            
            return {
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance,
                'near_support': support_distance < 0.05,  # Within 5%
                'near_resistance': resistance_distance < 0.05,  # Within 5%
                'strong_uptrend': sma_20 > sma_50 * 1.03 and price_change_20 > 0.05,  # Strong trend
                'strong_downtrend': sma_20 < sma_50 * 0.97 and price_change_20 < -0.05,  # Strong trend
                'bounce_potential': support_distance < 0.08 and price_change_20 < 0,  # Near support and declining
                'reversal_signals': resistance_distance < 0.08 and price_change_20 > 0.03,  # Near resistance and rising
                'distribution_signs': volume_surge and resistance_distance < 0.1,  # High volume near resistance
                'consolidation': abs(price_change_20) < 0.03  # Sideways movement
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return self.get_default_market_structure()
    
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
        """Create BUY signal with balanced logic"""
        try:
            # Entry price logic
            entry_candidates = [current_price]
            
            # Use volume profile entry if reasonable confidence
            if volume_entry.get('confidence', 0) > 0.6:  # Lowered from 0.7
                entry_candidates.append(volume_entry['entry_price'])
            
            # Use support levels from confluence zones
            support_zones = [zone for zone in confluence_zones 
                           if zone['zone_type'] == 'support' and zone['price'] < current_price * 1.02]  # More flexible
            if support_zones:
                # Use closest support level
                closest_support = max(support_zones, key=lambda x: x['price'])
                entry_candidates.append(closest_support['price'])
            
            optimal_entry = np.mean(entry_candidates)
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            # Order type: use limit if entry is significantly different
            # order_type = 'limit' if distance_pct > 0.005 else 'market'  # Increased from 0.003
            # if order_type == 'market':
            #     optimal_entry = current_price
            
            # Smart order type based on distance ranges
            if distance_pct <= 0.001:  # Very close (0.1%)
                order_type = 'market'  # Immediate execution
            elif 0.001 < distance_pct <= 0.005:  # Close range (0.1-0.5%)
                order_type = 'market'  # Still use market for better fills
            else:  # Further away (>0.5%)
                order_type = 'limit'   # Use limit for better price

            if order_type == 'market':
                optimal_entry = current_price
            
            # Stop loss: use market structure (more reasonable)
            atr = latest.get('atr', current_price * 0.02)
            structure_stop = current_price * 0.94  # 6% structural stop (was 4%)
            atr_stop = optimal_entry - (2.5 * atr)  # 2.5 ATR stop (was 2.0)
            stop_loss = max(structure_stop, atr_stop)  # Use wider stop
            
            # Take profits: balanced but achievable
            tp1 = optimal_entry + (2.5 * atr)  # 2.5:1 R/R minimum (was 3)
            tp2 = optimal_entry + (4.5 * atr)  # 4.5:1 R/R stretch target (was 5)
            
            # Calculate confidence based on condition scores (more generous)
            base_confidence = 45  # Lowered from 50
            base_confidence += tech_score * 6  # Up to 36 points (was 5)
            base_confidence += trend_score * 6  # Up to 18 points (was 5)
            base_confidence += stoch_score * 6  # Up to 18 points (was 5)
            base_confidence += structure_score * 6  # Up to 18 points (was 5)
            
            # Bonus for confluence (more generous)
            if len(support_zones) > 1:
                base_confidence += 8  # Was 10
            if volume_entry.get('confidence', 0) > 0.7:
                base_confidence += 6  # Was 5
            
            confidence = min(92, base_confidence)  # Lowered cap from 95
            
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
                'signal_type': 'balanced_technical',
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
        """Create SELL signal with balanced logic"""
        try:
            # Entry price logic for shorts
            entry_candidates = [current_price]
            
            # Use resistance levels from confluence zones  
            resistance_zones = [zone for zone in confluence_zones
                               if zone['zone_type'] == 'resistance' and zone['price'] > current_price * 0.98]  # More flexible
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
            # order_type = 'limit' if distance_pct > 0.003 else 'market'  # Increased from 0.002
            # if order_type == 'market':
            #     optimal_entry = current_price

            # Smart order type based on distance ranges  
            if distance_pct <= 0.001:  # Very close (0.1%)
                order_type = 'market'  # Immediate execution
            elif 0.001 < distance_pct <= 0.005:  # Close range (0.1-0.5%)
                order_type = 'market'  # Still use market for better fills
            else:  # Further away (>0.5%)
                order_type = 'limit'   # Use limit for better price

            if order_type == 'market':
                optimal_entry = current_price
            
            # Stop loss: reasonable for shorts (not too tight)
            atr = latest.get('atr', current_price * 0.02)
            structure_stop = current_price * 1.06  # 6% structural stop (was 4%)
            atr_stop = optimal_entry + (2.0 * atr)  # 2.0 ATR stop (was 1.5)
            stop_loss = min(structure_stop, atr_stop)  # Use reasonable stop
            
            # Take profits: reasonable profits for shorts
            tp1 = optimal_entry - (2.5 * atr)  # 2.5:1 R/R minimum (was 3)
            tp2 = optimal_entry - (4.5 * atr)  # 4.5:1 R/R target (was 5)
            
            # Calculate confidence (more generous scoring)
            base_confidence = 40  # Lowered from 45
            base_confidence += tech_score * 5  # Up to 30 points (was 4)
            base_confidence += trend_score * 5  # Up to 20 points (was 4)
            base_confidence += stoch_score * 5  # Up to 20 points (was 4)
            base_confidence += structure_score * 5  # Up to 20 points (was 4)
            
            # Bonus for strong reversal signals (more generous)
            if len(resistance_zones) > 1:
                base_confidence += 6  # Was 8
            if latest.get('rsi', 50) > 75:  # Extreme overbought (was 80)
                base_confidence += 8  # Was 10
            
            confidence = min(88, base_confidence)  # Cap lower for shorts (was 90)
            
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
                'signal_type': 'balanced_reversal',
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