"""
Enhanced Signal Generation for the Bybit Trading System.
TUNED DOWN: Slightly reduced strictness while maintaining quality
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
    """Enhanced signal generation and ranking system with tuned approach"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_enhanced_signal(self, df: pd.DataFrame, symbol_data: Dict, 
                                volume_entry: Dict, confluence_zones: List[Dict]) -> Optional[Dict]:
            """TUNED: Generate proper trading signals with slightly relaxed conditions"""
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
                
                # ===== TUNED: BUY SIGNAL CONDITIONS =====
                # BUY when price is oversold and near support levels (relaxed conditions)
                
                # Primary BUY conditions (TUNED: more flexible)
                buy_technical_conditions = [
                    rsi < 35,                                    # TUNED: Slightly less strict (was 30)
                    macd > macd_signal * 0.95,                  # TUNED: Allow slight MACD lag
                    current_price < sma_20 * 1.01,             # TUNED: Allow slightly above SMA20
                    current_price > sma_200 * 0.93,            # TUNED: More flexible support (was 0.95)
                    volume_ratio > 1.1,                         # TUNED: Reduced volume requirement (was 1.2)
                    bb_position < 0.35,                         # TUNED: Slightly more flexible (was 0.3)
                ]
                
                # BUY trend conditions (TUNED: more flexible)
                buy_trend_conditions = [
                    ema_12 > ema_26 * 0.96,                     # TUNED: More flexible trend (was 0.98)
                    current_price > sma_50 * 0.93,             # TUNED: More flexible support (was 0.95)
                    sma_20 > sma_50 * 0.96,                     # TUNED: More flexible trend (was 0.98)
                    current_price > bb_lower * 1.01,           # TUNED: More flexible BB position
                ]
                
                # BUY Stochastic RSI conditions (TUNED: more flexible)
                buy_stoch_conditions = [
                    stoch_rsi_k < 30,                           # TUNED: Less strict (was 25)
                    stoch_rsi_k > stoch_rsi_d * 0.95,          # TUNED: Allow for close values
                    stoch_rsi_k > 3,                            # TUNED: Not at absolute bottom (was 5)
                    stoch_rsi_d < 35,                           # TUNED: More flexible (was 30)
                ]
                
                # Market structure for BUY (TUNED: more flexible)
                buy_structure_conditions = [
                    market_structure.get('near_support', False) or market_structure.get('bounce_potential', False),
                    not market_structure.get('strong_downtrend', False),
                    market_structure.get('price_range_position', 0.5) < 0.4,  # In lower part of range
                    current_price <= latest.get('vwap', current_price) * 1.01,  # At or slightly above VWAP
                ]
                
                # Count conditions
                buy_tech_score = sum(buy_technical_conditions)
                buy_trend_score = sum(buy_trend_conditions)
                buy_stoch_score = sum(buy_stoch_conditions)
                buy_structure_score = sum(buy_structure_conditions)
                
                # TUNED: Relaxed BUY requirements
                if (buy_tech_score >= 3 and buy_trend_score >= 2 and 
                    buy_stoch_score >= 2 and buy_structure_score >= 1):
                    
                    signal = self.create_buy_signal(
                        symbol_data, current_price, latest, volume_entry, 
                        confluence_zones, buy_tech_score, buy_trend_score, 
                        buy_stoch_score, buy_structure_score
                    )
                
                # ===== TUNED: SELL SIGNAL CONDITIONS =====  
                # SELL when price is overbought and near resistance levels (relaxed conditions)
                if not signal:  # Only if no BUY signal
                    
                    # Primary SELL conditions (TUNED: more flexible)
                    sell_technical_conditions = [
                        rsi > 70,                                   # TUNED: Less strict (was 75)
                        macd < macd_signal * 1.05,                 # TUNED: Allow slight MACD lag
                        current_price > sma_20 * 0.99,             # TUNED: Allow slightly below SMA20
                        current_price < sma_200 * 1.12,            # TUNED: More flexible resistance (was 1.10)
                        volume_ratio > 1.3,                         # TUNED: Reduced volume requirement (was 1.5)
                        bb_position > 0.65,                         # TUNED: More flexible (was 0.7)
                    ]
                    
                    # SELL trend conditions (TUNED: more flexible)
                    sell_trend_conditions = [
                        ema_12 < ema_26 * 1.04,                    # TUNED: More flexible (was 1.02)
                        current_price < sma_50 * 1.10,             # TUNED: More flexible (was 1.08)
                        sma_20 < sma_50 * 1.07,                    # TUNED: More flexible (was 1.05)
                        current_price < bb_upper * 0.99,           # TUNED: More flexible rejection
                    ]
                    
                    # SELL Stochastic RSI conditions (TUNED: more flexible)
                    sell_stoch_conditions = [
                        stoch_rsi_k > 75,                           # TUNED: Less strict (was 80)
                        stoch_rsi_k < stoch_rsi_d * 1.05,          # TUNED: Allow for close values
                        stoch_rsi_k < 97,                           # TUNED: Not at absolute top (was 95)
                        stoch_rsi_d > 70,                           # TUNED: More flexible (was 75)
                    ]
                    
                    # Market structure for SELL (TUNED: more flexible)
                    sell_structure_conditions = [
                        market_structure.get('near_resistance', False) or market_structure.get('distribution_signs', False),
                        market_structure.get('reversal_signals', False) or rsi > 75,  # Allow extreme RSI as reversal signal
                        not market_structure.get('strong_uptrend', False),
                        current_price >= latest.get('vwap', current_price) * 0.99,  # At or slightly below VWAP
                    ]
                    
                    # Count conditions
                    sell_tech_score = sum(sell_technical_conditions)
                    sell_trend_score = sum(sell_trend_conditions)
                    sell_stoch_score = sum(sell_stoch_conditions)
                    sell_structure_score = sum(sell_structure_conditions)
                    
                    # TUNED: Relaxed SELL requirements
                    if (sell_tech_score >= 3 and sell_trend_score >= 2 and 
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
        """TUNED: Analyze market structure with balanced approach"""
        try:
            # Calculate recent price action
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            price_range = recent_high - recent_low
            
            # Check proximity to support/resistance (TUNED: more flexible)
            near_support = current_price < (recent_low + price_range * 0.35)  # Was 0.3
            near_resistance = current_price > (recent_high - price_range * 0.35)  # Was 0.3
            
            # Trend analysis (TUNED: more flexible)
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            
            strong_uptrend = (sma_20 > sma_50 * 1.03) and (current_price > sma_20 * 1.01)  # Was 1.05 and 1.02
            strong_downtrend = (sma_20 < sma_50 * 0.97) and (current_price < sma_20 * 0.99)  # Was 0.95 and 0.98
            
            # Reversal signals (TUNED: more sensitive)
            recent_candles = df.tail(3)  # Look at fewer candles
            reversal_signals = False
            
            # Check for doji, hammer, shooting star patterns
            for _, candle in recent_candles.iterrows():
                body_size = abs(candle['close'] - candle['open'])
                candle_range = candle['high'] - candle['low']
                
                if body_size < candle_range * 0.4:  # More flexible (was 0.3)
                    reversal_signals = True
                    break
            
            # Volume analysis for distribution/accumulation (TUNED: more flexible)
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['volume'].tail(3).mean()  # Shorter period
            high_volume = recent_volume > volume_ma * 1.3  # Was 1.5
            
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
                'bounce_potential': True,  # Default to allowing signals
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
        """Check if we're in a very strong uptrend (TUNED: more strict definition)"""
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
            
            # Very strong uptrend conditions (TUNED: more strict to allow more shorts)
            conditions = [
                sma_20 > sma_50 * 1.08,  # TUNED: More strict (was 1.05)
                ema_12 > ema_26 * 1.05,  # TUNED: More strict (was 1.03)
                latest['close'] > sma_20 * 1.05,  # TUNED: More strict (was 1.02)
                recent['close'].iloc[-1] > recent['close'].iloc[-10] * 1.08,  # TUNED: More strict (was 1.05)
                recent['volume'].mean() > df['volume'].tail(50).mean() * 1.3  # TUNED: More strict (was 1.2)
            ]
            
            return sum(conditions) >= 4  # Need 4/5 conditions
            
        except Exception as e:
            self.logger.error(f"Very strong uptrend check error: {e}")
            return False
    
    def create_buy_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                         volume_entry: Dict, confluence_zones: List[Dict],
                         tech_score: int, trend_score: int, stoch_score: int, structure_score: int) -> Dict:
        """TUNED: Create BUY signal with flexible entry logic"""
        try:
            # Entry price logic for BUY
            entry_candidates = [current_price]
            
            # Use support levels from confluence zones  
            support_zones = [zone for zone in confluence_zones
                           if zone['zone_type'] == 'support' and 
                           zone['price'] < current_price * 1.03 and  # TUNED: More flexible (was 1.02)
                           zone['price'] > current_price * 0.93]     # TUNED: More flexible (was 0.95)
            
            if support_zones:
                entry_candidates.append(support_zones[0]['price'])
            
            # Use volume-based entry if it's a support level
            if (volume_entry.get('confidence', 0) > 0.5 and  # TUNED: Lower confidence req (was 0.6)
                volume_entry.get('entry_price', current_price) < current_price):
                entry_candidates.append(volume_entry['entry_price'])
            
            # Use lower Bollinger Band as support
            bb_lower = latest.get('bb_lower', current_price * 0.98)
            if bb_lower < current_price and bb_lower > current_price * 0.93:  # TUNED: More flexible
                entry_candidates.append(bb_lower)
            
            # Choose optimal entry
            optimal_entry = min(entry_candidates)
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            # Order type logic for BUY (TUNED: more market orders)
            if distance_pct > 0.008 or latest.get('stoch_rsi_k', 50) < 30:  # TUNED: Lower threshold (was 0.01)
                order_type = 'limit'
            else:
                order_type = 'market'
                optimal_entry = current_price
            
            # Stop loss and take profit for BUY
            atr = latest.get('atr', current_price * 0.02)
            stop_loss = optimal_entry - (1.8 * atr)      # TUNED: Tighter stop (was 2.0)
            take_profit_1 = optimal_entry + (2.8 * atr)  # TUNED: Closer TP1 (was 3.0)
            take_profit_2 = optimal_entry + (5.5 * atr)  # TUNED: Closer TP2 (was 6.0)
            
            # Risk-reward calculation for BUY
            risk_amount = optimal_entry - stop_loss
            reward_amount = take_profit_2 - optimal_entry
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Confidence calculation for BUY (TUNED: higher base confidence)
            base_confidence = 55  # TUNED: Higher base (was 50)
            
            # Score-based confidence
            base_confidence += tech_score * 3.5    # TUNED: Slightly lower weight (was 4)
            base_confidence += trend_score * 3.5   # TUNED: Slightly lower weight (was 4)
            base_confidence += stoch_score * 3.5   # TUNED: Slightly lower weight (was 4)
            base_confidence += structure_score * 3 # Same weight
            
            # Bonus for good support levels
            if len(support_zones) > 0:
                base_confidence += 6  # TUNED: Lower bonus (was 8)
            if volume_entry.get('confidence', 0) > 0.6:
                base_confidence += 4  # TUNED: Lower bonus (was 6)
            
            confidence = min(88, base_confidence)  # TUNED: Lower cap (was 90)
            
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
                'signal_type': 'tuned_technical_buy',
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
        """TUNED: Create SELL signal with flexible entry logic"""
        try:
            # Entry price logic for SELL
            entry_candidates = [current_price]
            
            # Use resistance levels from confluence zones  
            resistance_zones = [zone for zone in confluence_zones
                              if zone['zone_type'] == 'resistance' and 
                              zone['price'] > current_price * 0.97 and  # TUNED: More flexible (was 0.98)
                              zone['price'] < current_price * 1.07]     # TUNED: More flexible (was 1.05)
            
            if resistance_zones:
                entry_candidates.append(resistance_zones[0]['price'])
            
            # Use volume-based entry if it's a resistance level
            if (volume_entry.get('confidence', 0) > 0.5 and  # TUNED: Lower confidence req (was 0.6)
                volume_entry.get('entry_price', current_price) > current_price):
                entry_candidates.append(volume_entry['entry_price'])
            
            # Use upper Bollinger Band as resistance
            bb_upper = latest.get('bb_upper', current_price * 1.02)
            if bb_upper > current_price and bb_upper < current_price * 1.07:  # TUNED: More flexible
                entry_candidates.append(bb_upper)
            
            # Choose optimal entry
            optimal_entry = max(entry_candidates)
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            # Order type logic for SELL (TUNED: more market orders)
            if distance_pct > 0.008 or latest.get('stoch_rsi_k', 50) > 75:  # TUNED: Lower threshold (was 0.01)
                order_type = 'limit'
            else:
                order_type = 'market'
                optimal_entry = current_price
            
            # Stop loss and take profit for SELL
            atr = latest.get('atr', current_price * 0.02)
            stop_loss = optimal_entry + (1.8 * atr)      # TUNED: Tighter stop (was 2.0)
            take_profit_1 = optimal_entry - (2.8 * atr)  # TUNED: Closer TP1 (was 3.0)
            take_profit_2 = optimal_entry - (5.5 * atr)  # TUNED: Closer TP2 (was 6.0)
            
            # Risk-reward calculation for SELL
            risk_amount = stop_loss - optimal_entry
            reward_amount = optimal_entry - take_profit_2
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            # Confidence calculation for SELL (TUNED: higher base confidence)
            base_confidence = 52  # TUNED: Higher base (was 50)
            
            # Score-based confidence
            base_confidence += tech_score * 3.5    # TUNED: Slightly lower weight (was 4)
            base_confidence += trend_score * 3.5   # TUNED: Slightly lower weight (was 4)
            base_confidence += stoch_score * 3.5   # TUNED: Slightly lower weight (was 4)
            base_confidence += structure_score * 3 # Same weight
            
            # Bonus for good resistance levels
            if len(resistance_zones) > 0:
                base_confidence += 6  # TUNED: Lower bonus (was 8)
            if volume_entry.get('confidence', 0) > 0.6:
                base_confidence += 4  # TUNED: Lower bonus (was 6)
            
            confidence = min(88, base_confidence)  # TUNED: Lower cap (was 90)
            
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
                'signal_type': 'tuned_technical_sell',
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
        """Signal quality validation with tuned thresholds"""
        try:
            if not signal:
                return None
            
            symbol = signal['symbol']
            side = signal['side']
            confidence = signal['confidence']
            risk_reward = signal['risk_reward_ratio']
            latest = df.iloc[-1]
            current_price = signal['current_price']
            
            # Minimum confidence thresholds (TUNED: more accessible)
            min_confidence = 45 if side == 'buy' else 47  # TUNED: Lower thresholds (was 50/55)
            if confidence < min_confidence:
                self.logger.debug(f"Signal filtered: {symbol} confidence {confidence}% below minimum {min_confidence}%")
                return None
            
            # Minimum risk-reward ratio (TUNED: more accessible)
            min_rr = 1.8 if side == 'buy' else 1.9  # TUNED: Lower thresholds (was 2.0/2.2)
            if risk_reward < min_rr:
                self.logger.debug(f"Signal filtered: {symbol} R/R {risk_reward:.1f} below minimum {min_rr}")
                return None
            
            # Market structure validation (TUNED: more lenient)
            if side == 'sell':
                # Don't short unless reasonably near resistance
                if not market_structure['near_resistance'] and market_structure.get('resistance_distance', 0.15) > 0.15:  # TUNED: More lenient
                    self.logger.debug(f"SELL signal filtered: {symbol} not near resistance")
                    return None
                
                # Don't short in very strong uptrends only
                if self.is_very_strong_uptrend(df):
                    self.logger.debug(f"SELL signal filtered: {symbol} in very strong uptrend")
                    return None
            
            elif side == 'buy':
                # Don't buy if too far from support (TUNED: more lenient)
                if market_structure.get('support_distance', 0.15) > 0.15:  # TUNED: More lenient
                    self.logger.debug(f"BUY signal filtered: {symbol} too far from support")
                    return None
            
            # Volume validation (TUNED: more lenient)
            volume_ratio = df.iloc[-1].get('volume_ratio', 1)
            min_volume = 0.8 if side == 'buy' else 1.1  # TUNED: Lower requirements (was 1.0/1.3)
            if volume_ratio < min_volume:
                self.logger.debug(f"Signal filtered: {symbol} volume {volume_ratio:.1f} below minimum {min_volume}")
                return None
            
            # Volatility check (TUNED: more lenient)
            atr_pct = signal.get('atr_percentage', latest.get('atr', current_price * 0.02) / current_price)
            if atr_pct > 0.15:  # TUNED: Higher tolerance (was 0.12)
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
                
                # ===== TUNED SCORING SYSTEM =====
                
                # 1. SIGNAL CONFIDENCE (35% weight)
                confidence_score = confidence / 100
                
                # 2. RISK-REWARD RATIO (25% weight)  
                analysis = signal.get('analysis', {})
                risk_assessment = analysis.get('risk_assessment', {})
                rr_ratio = risk_assessment.get('risk_reward_ratio', signal.get('risk_reward_ratio', 1))
                rr_score = min(1.0, rr_ratio / 3.0)  # TUNED: Lower target (was 3.5)
                
                # 3. MARKET STRUCTURE ALIGNMENT (20% weight)
                entry_methods = signal.get('entry_methods', {})
                structure_score = 0
                if 'structure_score' in entry_methods:
                    structure_score = min(1.0, entry_methods['structure_score'] / 2.5)  # TUNED: Lower requirement
                elif 'confluence_zones' in entry_methods:
                    structure_score = min(1.0, entry_methods['confluence_zones'] / 1.5)  # TUNED: Lower requirement
                
                # 4. VOLUME QUALITY (10% weight)
                volume_score = min(1.0, signal['volume_24h'] / 20_000_000)  # TUNED: Lower requirement (was 30M)
                
                # 5. MTF CONFIRMATION (15% weight)
                mtf_analysis = signal.get('mtf_analysis', {})
                confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
                total_timeframes = len(getattr(self.config, 'confirmation_timeframes', [])) or 3
                
                if total_timeframes > 0:
                    mtf_score = confirmed_count / total_timeframes
                    mtf_bonus = mtf_score * 0.15
                else:
                    mtf_bonus = 0
                
                # 6. DISTANCE PENALTY (3% weight)
                distance = abs(signal['entry_price'] - signal['current_price']) / signal['current_price']
                distance_score = max(0, 1 - distance * 6)  # TUNED: Less penalty (was 8)
                
                # 7. ORDER TYPE (2% weight)
                order_type_score = 1.0 if signal['order_type'] == 'market' else 0.97  # TUNED: Less penalty
                
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
                
                # ===== TUNED PRIORITY SYSTEM =====
                mtf_status = signal.get('mtf_status', 'NONE')
                
                # Base priority on confidence and R/R (TUNED: more accessible)
                if confidence >= 70 and rr_ratio >= 3.0:
                    base_priority = 1000  # Exceptional signals
                elif confidence >= 60 and rr_ratio >= 2.5:  # TUNED: Lower requirements
                    base_priority = 500   # High quality signals
                elif confidence >= 50 and rr_ratio >= 2.0:  # TUNED: Lower requirements
                    base_priority = 250   # Good signals (increased)
                elif confidence >= 45 and rr_ratio >= 1.8:  # TUNED: Added tier
                    base_priority = 100   # Decent signals
                else:
                    base_priority = 50    # Marginal signals
                
                # MTF modifier (balanced)
                mtf_modifier = {
                    'STRONG': 1.4,   # 40% boost
                    'PARTIAL': 1.15,  # 15% boost
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
            
            # self.logger.info(f"📊 Ranked {len(opportunities)} opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return signals

    def assess_risk(self, signal: Dict, df: pd.DataFrame, market_data: Dict) -> Dict:
        """Enhanced risk assessment with tuned approach"""
        try:
            latest = df.iloc[-1]
            current_price = signal['current_price']
            
            # Volatility risk (TUNED: more lenient)
            atr = latest.get('atr', current_price * 0.02)
            volatility = atr / current_price
            
            # Distance risk (TUNED: more lenient)
            distance = abs(signal['entry_price'] - current_price) / current_price
            distance_risk = min(1.0, distance * 6)  # TUNED: Reduced from 8
            
            # Condition-based risk (TUNED: more lenient thresholds)
            rsi = latest.get('rsi', 50)
            bb_position = latest.get('bb_position', 0.5)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Risk factors (TUNED: more lenient thresholds)
            extreme_rsi = rsi < 10 or rsi > 90  # TUNED: More extreme (was 15/85)
            extreme_bb = bb_position < 0.02 or bb_position > 0.98  # TUNED: More extreme
            low_volume = volume_ratio < 0.6  # TUNED: Lower threshold (was 0.7)
            
            condition_risk = 0
            if extreme_rsi:
                condition_risk += 0.10  # TUNED: Reduced from 0.15
            if extreme_bb:
                condition_risk += 0.08  # TUNED: Reduced from 0.10
            if low_volume:
                condition_risk += 0.06  # TUNED: Reduced from 0.08
            
            # Side-specific risk (TUNED: reduced penalty for shorts)
            side_risk = 0.03 if signal['side'] == 'sell' else 0.01  # TUNED: Reduced
            
            # MTF risk reduction
            mtf_analysis = signal.get('mtf_analysis', {})
            confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
            total_timeframes = len(getattr(self.config, 'confirmation_timeframes', [])) or 3
            mtf_risk_reduction = (confirmed_count / total_timeframes) * 0.25 if total_timeframes > 0 else 0  # TUNED: Increased
            
            # Calculate total risk score
            base_risk = (volatility * 2.0 + distance_risk * 2.0 + condition_risk + side_risk)  # TUNED: Reduced multipliers
            total_risk = max(0.05, min(1.0, base_risk - mtf_risk_reduction))  # TUNED: Lower minimum
            
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
            
            # Risk level classification (TUNED: more lenient)
            if total_risk > 0.90:  # TUNED: Increased from 0.85
                risk_level = 'Very High'
            elif total_risk > 0.70:  # TUNED: Increased from 0.65
                risk_level = 'High'
            elif total_risk > 0.50:  # TUNED: Increased from 0.45
                risk_level = 'Medium'
            elif total_risk > 0.30:  # TUNED: Increased from 0.25
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
            return {'total_risk_score': 0.3, 'risk_level': 'Medium'}  # TUNED: Lower default risk

    def assess_signal_risk(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Alias for assess_risk method for backward compatibility"""
        return self.assess_risk(signal, df, {})

    def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol_data: Dict, 
                                   volume_profile: Dict, fibonacci_data: Dict, 
                                   confluence_zones: List[Dict]) -> Optional[Dict]:
        """Comprehensive symbol analysis with tuned approach"""
        try:
            if df.empty or len(df) < 15:  # TUNED: Lower minimum data requirement (was 20)
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
            if 'volume' not in df.columns or len(df) < 15:  # TUNED: Lower requirement
                return {'pattern': 'insufficient_data', 'strength': 0}
            
            recent_15 = df.tail(15)  # TUNED: Shorter period
            
            # Volume trend
            volume_ma_5 = recent_15['volume'].rolling(5).mean().iloc[-1]
            volume_ma_15 = df['volume'].rolling(15).mean().iloc[-1]  # TUNED: Shorter period
            
            # Buying vs selling pressure
            up_volume = recent_15[recent_15['close'] > recent_15['open']]['volume'].sum()
            down_volume = recent_15[recent_15['close'] < recent_15['open']]['volume'].sum()
            total_volume = up_volume + down_volume
            
            buying_pressure = up_volume / total_volume if total_volume > 0 else 0.5
            
            # Volume pattern classification (TUNED: more lenient)
            if volume_ma_5 > volume_ma_15 * 1.25:  # TUNED: Lower threshold
                pattern = 'surge'
            elif volume_ma_5 > volume_ma_15 * 1.08:  # TUNED: Lower threshold
                pattern = 'increasing'
            elif volume_ma_5 < volume_ma_15 * 0.75:  # TUNED: More lenient
                pattern = 'declining'
            else:
                pattern = 'stable'
            
            return {
                'pattern': pattern,
                'buying_pressure': buying_pressure,
                'volume_ma_ratio': volume_ma_5 / volume_ma_15 if volume_ma_15 > 0 else 1,
                'strength': min(1.0, volume_ma_5 / volume_ma_15) if volume_ma_15 > 0 else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Volume pattern analysis error: {e}")
            return {'pattern': 'unknown', 'strength': 0.5}

    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Calculate trend strength using multiple indicators"""
        try:
            if len(df) < 30:  # TUNED: Lower requirement (was 50)
                return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'low'}
            
            latest = df.iloc[-1]
            recent_30 = df.tail(30)  # TUNED: Shorter period
            
            # Price trend
            price_change_15 = (latest['close'] - recent_30.iloc[-15]['close']) / recent_30.iloc[-15]['close']  # TUNED: Shorter period
            price_change_30 = (latest['close'] - recent_30.iloc[0]['close']) / recent_30.iloc[0]['close']
            
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
            bullish_candles = len(recent_30[recent_30['close'] > recent_30['open']])
            consistency = bullish_candles / len(recent_30)
            
            # Overall trend strength
            strength = (abs(price_change_15) + abs(ma_alignment_score) / 4 + consistency) / 3
            strength = min(1.0, strength)
            
            # Direction determination (TUNED: more lenient thresholds)
            if price_change_15 > 0.015 and ma_alignment_score > 0:  # TUNED: Lower threshold
                direction = 'strong_bullish'
            elif price_change_15 > 0 and ma_alignment_score >= 0:
                direction = 'bullish'
            elif price_change_15 < -0.015 and ma_alignment_score < 0:  # TUNED: Lower threshold
                direction = 'strong_bearish'
            elif price_change_15 < 0 and ma_alignment_score <= 0:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            consistency_level = 'high' if consistency > 0.65 else 'medium' if consistency > 0.35 else 'low'  # TUNED: More lenient
            
            return {
                'strength': strength,
                'direction': direction,
                'consistency': consistency_level,
                'price_change_15': price_change_15,
                'ma_alignment_score': ma_alignment_score
            }
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'low'}

    def analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze recent price action patterns"""
        try:
            if len(df) < 8:  # TUNED: Lower requirement (was 10)
                return {'pattern': 'insufficient_data', 'strength': 0}
            
            recent_8 = df.tail(8)  # TUNED: Shorter period
            latest = df.iloc[-1]
            
            # Candlestick patterns
            body_size = abs(latest['close'] - latest['open']) / latest['open']
            upper_shadow = latest['high'] - max(latest['close'], latest['open'])
            lower_shadow = min(latest['close'], latest['open']) - latest['low']
            
            # Pattern identification
            patterns = []
            
            # Doji-like (TUNED: more lenient)
            if body_size < 0.002:  # TUNED: More lenient
                patterns.append('doji')
            
            # Hammer/Shooting star (TUNED: more lenient)
            if lower_shadow > body_size * 1.5 and upper_shadow < body_size:  # TUNED: More lenient
                patterns.append('hammer')
            elif upper_shadow > body_size * 1.5 and lower_shadow < body_size:  # TUNED: More lenient
                patterns.append('shooting_star')
            
            # Support/Resistance test
            recent_lows = recent_8['low'].min()
            recent_highs = recent_8['high'].max()
            
            if latest['low'] <= recent_lows * 1.002:  # TUNED: More lenient
                patterns.append('support_test')
            if latest['high'] >= recent_highs * 0.998:  # TUNED: More lenient
                patterns.append('resistance_test')
            
            # Price momentum
            closes = recent_8['close'].values
            momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            
            return {
                'patterns': patterns,
                'momentum': momentum,
                'body_size': body_size,
                'shadow_ratio': (upper_shadow + lower_shadow) / body_size if body_size > 0 else 0,
                'strength': min(1.0, abs(momentum) * 8 + body_size * 40)  # TUNED: More lenient scoring
            }
            
        except Exception as e:
            self.logger.error(f"Price action analysis error: {e}")
            return {'pattern': 'unknown', 'strength': 0.5}

    def assess_market_conditions(self, df: pd.DataFrame, symbol_data: Dict) -> Dict:
        """Assess overall market conditions for the symbol"""
        try:
            latest = df.iloc[-1]
            
            # Liquidity assessment (TUNED: more lenient)
            volume_24h = symbol_data.get('volume_24h', 0)
            price_change_24h = symbol_data.get('price_change_24h', 0)
            
            # Market cap and volume relationship (TUNED: lower thresholds)
            if volume_24h > 5_000_000:  # TUNED: Lower threshold
                liquidity = 'high'
            elif volume_24h > 500_000:  # TUNED: Lower threshold
                liquidity = 'medium'
            else:
                liquidity = 'low'
            
            # Volatility assessment (TUNED: more lenient)
            atr_pct = latest.get('atr', latest['close'] * 0.02) / latest['close']
            if atr_pct > 0.08:  # TUNED: Higher threshold
                volatility_level = 'high'
            elif atr_pct > 0.04:  # TUNED: Higher threshold
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
            
            # Market sentiment (TUNED: more lenient thresholds)
            if price_change_24h > 4:  # TUNED: Lower threshold
                sentiment = 'very_bullish'
            elif price_change_24h > 1.5:  # TUNED: Lower threshold
                sentiment = 'bullish'
            elif price_change_24h < -4:  # TUNED: Lower threshold
                sentiment = 'very_bearish'
            elif price_change_24h < -1.5:  # TUNED: Lower threshold
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            # Trading conditions
            conditions = []
            if liquidity in ['high', 'medium']:  # TUNED: Accept medium liquidity
                conditions.append('good_liquidity')
            if volatility_level in ['medium', 'high']:
                conditions.append('sufficient_volatility')
            if abs(price_change_24h) > 0.8:  # TUNED: Lower threshold
                conditions.append('active_movement')
            
            return {
                'liquidity': liquidity,
                'volatility_level': volatility_level,
                'sentiment': sentiment,
                'price_change_24h': price_change_24h,
                'volume_24h': volume_24h,
                'trading_conditions': conditions,
                'favorable_for_trading': len(conditions) >= 1  # TUNED: Lower requirement
            }
            
        except Exception as e:
            self.logger.error(f"Market conditions assessment error: {e}")
            return {'liquidity': 'unknown', 'volatility_level': 'unknown', 'sentiment': 'neutral'}

    def filter_signals_by_quality(self, signals: List[Dict], max_signals: int = 25) -> List[Dict]:
        """Filter signals by quality metrics with tuned approach"""
        try:
            if not signals:
                return []
            
            # Sort by confidence and risk-reward
            quality_signals = []
            
            for signal in signals:
                confidence = signal.get('confidence', 0)
                rr_ratio = signal.get('risk_reward_ratio', 0)
                volume_24h = signal.get('volume_24h', 0)
                
                # Quality score calculation (TUNED: more accessible)
                quality_score = (
                    confidence * 0.4 +  # 40% weight on confidence
                    min(100, rr_ratio * 18) * 0.3 +  # 30% weight on R/R (TUNED: more lenient)
                    min(100, volume_24h / 800_000) * 0.2 +  # 20% weight on volume (TUNED: lower threshold)
                    (100 if signal.get('order_type') == 'market' else 92) * 0.1  # 10% weight on execution (TUNED: less penalty)
                )
                
                signal['quality_score'] = quality_score
                
                # Quality filters (TUNED: more lenient)
                if (confidence >= 40 and rr_ratio >= 1.6 and volume_24h >= 300_000):  # TUNED: Much lower thresholds
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
            
            # Confidence distribution (TUNED: adjusted thresholds)
            high_confidence = len([op for op in opportunities if op['confidence'] >= 65])  # TUNED: Lower threshold
            medium_confidence = len([op for op in opportunities if 45 <= op['confidence'] < 65])  # TUNED: Lower threshold
            low_confidence = len([op for op in opportunities if op['confidence'] < 45])
            
            # Risk-reward distribution (TUNED: adjusted thresholds)
            excellent_rr = len([op for op in opportunities if op.get('risk_reward_ratio', 0) >= 2.5])  # TUNED: Lower threshold
            good_rr = len([op for op in opportunities if 1.8 <= op.get('risk_reward_ratio', 0) < 2.5])  # TUNED: Lower threshold
            fair_rr = len([op for op in opportunities if op.get('risk_reward_ratio', 0) < 1.8])
            
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
        """Generate trading recommendation based on signal quality (TUNED)"""
        try:
            if not opportunities:
                return "No signals found. Wait for better market conditions."
            
            total_ops = len(opportunities)
            high_quality = len([op for op in opportunities 
                              if op['confidence'] >= 60 and op.get('risk_reward_ratio', 0) >= 2.2])  # TUNED: Lower thresholds
            
            if high_quality >= 2:  # TUNED: Lower requirement
                return f"Excellent conditions: {high_quality} high-quality signals available."
            elif high_quality >= 1:
                return f"Good conditions: {high_quality} quality signal(s) found."
            elif total_ops >= 3:  # TUNED: Lower requirement
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