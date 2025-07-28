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
        """Fixed signal generation with proper conditions"""
        try:
            latest = df.iloc[-1]
            current_price = symbol_data['current_price']
            symbol = symbol_data['symbol']
            
            # Get indicator values with proper validation
            rsi = latest.get('rsi')
            if rsi is None:
                self.logger.warning(f"Missing RSI data for {symbol}")
                return None
                
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            stoch_rsi_k = latest.get('stoch_rsi_k')
            stoch_rsi_d = latest.get('stoch_rsi_d')
            
            # Ensure we have critical indicators
            if stoch_rsi_k is None or stoch_rsi_d is None:
                self.logger.warning(f"Missing Stochastic RSI data for {symbol}")
                return None
            
            # Market structure analysis - more strict requirements
            market_structure = self.analyze_market_structure(df, current_price)
            if not self.has_sufficient_market_data(market_structure):
                self.logger.debug(f"Insufficient market structure data for {symbol}")
                return None
            
            signal = None
            
            # ===== IMPROVED BUY SIGNAL CONDITIONS =====
            buy_primary_conditions = [
                rsi < 35,  # More conservative than 30
                stoch_rsi_k < 30,  # Must be oversold
                stoch_rsi_k > stoch_rsi_d,  # Stoch RSI K crossing above D
                macd > macd_signal * 0.98,  # MACD not too bearish
            ]
            
            buy_confirmation_conditions = [
                market_structure.get('near_support', False),  # Must be near support
                not market_structure.get('strong_downtrend', False),
                latest.get('volume_ratio', 1) > 1.5,  # Higher volume requirement
                current_price > latest.get('sma_200', current_price * 0.95) * 0.98,  # Above major support
            ]
            
            buy_trend_conditions = [
                latest.get('ema_12', current_price) > latest.get('ema_26', current_price) * 0.995,
                latest.get('bb_position', 0.5) < 0.4,  # In lower part of BB
            ]
            
            # Count conditions - STRICTER REQUIREMENTS
            buy_primary_score = sum(buy_primary_conditions)
            buy_confirm_score = sum(buy_confirmation_conditions)
            buy_trend_score = sum(buy_trend_conditions)
            
            # Need ALL primary conditions + most confirmation conditions
            if (buy_primary_score >= 2 and buy_confirm_score >= 2 and buy_trend_score >= 1):
                signal = self.create_buy_signal(
                    symbol_data, current_price, latest, volume_entry, 
                    confluence_zones, buy_primary_score, buy_confirm_score, buy_trend_score
                )
            
            # ===== IMPROVED SELL SIGNAL CONDITIONS =====
            if not signal:  # Only if no BUY signal
                sell_primary_conditions = [
                    rsi > 70,  # More conservative than 75
                    stoch_rsi_k > 75,  # Must be overbought
                    stoch_rsi_k < stoch_rsi_d,  # Stoch RSI K crossing below D
                    macd < macd_signal * 1.02,  # MACD turning bearish
                ]
                
                sell_confirmation_conditions = [
                    market_structure.get('near_resistance', False),  # Must be near resistance
                    not market_structure.get('strong_uptrend', False),
                    latest.get('volume_ratio', 1) > 1.8,  # Higher volume for sells
                    current_price < latest.get('sma_200', current_price * 1.05) * 1.05,  # Below major resistance
                ]
                
                sell_trend_conditions = [
                    latest.get('ema_12', current_price) < latest.get('ema_26', current_price) * 1.005,
                    latest.get('bb_position', 0.5) > 0.6,  # In upper part of BB
                ]
                
                # Count conditions - STRICTER REQUIREMENTS
                sell_primary_score = sum(sell_primary_conditions)
                sell_confirm_score = sum(sell_confirmation_conditions)
                sell_trend_score = sum(sell_trend_conditions)
                
                # Need ALL primary conditions + most confirmation conditions
                if (sell_primary_score >= 2 and sell_confirm_score >= 2 and sell_trend_score >= 1):
                    signal = self.create_sell_signal(
                        symbol_data, current_price, latest, volume_entry, 
                        confluence_zones, sell_primary_score, sell_confirm_score, sell_trend_score
                    )

            # Log signal debug
            # print(f"\n=== DEBUG: {symbol} ===")
            # print(f"RSI: {latest.get('rsi', 'Missing')}")
            # print(f"Stoch RSI K: {latest.get('stoch_rsi_k', 'Missing')}")
            # print(f"Stoch RSI D: {latest.get('stoch_rsi_d', 'Missing')}")
            # print(f"MACD: {latest.get('macd', 'Missing')}")
            # print(f"MACD Signal: {latest.get('macd_signal', 'Missing')}")
            # print(f"Volume Ratio: {latest.get('volume_ratio', 'Missing')}")
            # print(f"BB Position: {latest.get('bb_position', 'Missing')}")
            
            # # Check buy conditions
            # rsi = latest.get('rsi', 50)
            # stoch_k = latest.get('stoch_rsi_k', 50)
            
            # print(f"\nBuy Conditions Check:")
            # print(f"RSI < 35: {rsi < 35} (RSI: {rsi})")
            # print(f"Stoch RSI K < 30: {stoch_k < 30} (K: {stoch_k})")
            
            # print(f"\nSell Conditions Check:")
            # print(f"RSI > 70: {rsi > 70} (RSI: {rsi})")
            # print(f"Stoch RSI K > 75: {stoch_k > 75} (K: {stoch_k})")               
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol_data.get('symbol', 'unknown')}: {e}")
            return None
        
    def has_sufficient_market_data(self, market_structure: Dict) -> bool:
        """Ensure we have enough data to make trading decisions"""
        required_keys = ['near_support', 'near_resistance', 'strong_uptrend', 'strong_downtrend']
        
        # Check if we have real market structure data (not defaults)
        if all(key in market_structure for key in required_keys):
            # Additional check: if all values are default-like, reject
            if (not market_structure['near_support'] and 
                not market_structure['near_resistance'] and
                not market_structure['strong_uptrend'] and 
                not market_structure['strong_downtrend']):
                return False
            return True
        return False
           
    def analyze_market_structure(self, df: pd.DataFrame, current_price: float) -> Dict:
        """More robust market structure analysis"""
        try:
            if len(df) < 50:  # Need more data for reliable analysis
                return self.get_insufficient_data_structure()
            
            # Calculate with sufficient lookback
            recent_high = df['high'].tail(30).max()
            recent_low = df['low'].tail(30).min()
            price_range = recent_high - recent_low
            
            if price_range == 0:
                return self.get_insufficient_data_structure()
            
            # More strict proximity requirements
            near_support = current_price < (recent_low + price_range * 0.25)  # Stricter
            near_resistance = current_price > (recent_high - price_range * 0.25)  # Stricter
            
            # Trend analysis with more data
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            # More strict trend definitions
            strong_uptrend = (sma_20 > sma_50 * 1.02) and (current_price > sma_20 * 1.015)
            strong_downtrend = (sma_20 < sma_50 * 0.98) and (current_price < sma_20 * 0.985)
            
            # Volume confirmation
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['volume'].tail(5).mean()
            
            return {
                'near_support': near_support,
                'near_resistance': near_resistance,
                'strong_uptrend': strong_uptrend,
                'strong_downtrend': strong_downtrend,
                'bounce_potential': near_support and not strong_downtrend and recent_volume > volume_ma,
                'distribution_signs': near_resistance and recent_volume > volume_ma * 1.5,
                'price_range_position': (current_price - recent_low) / price_range,
                'has_sufficient_data': True
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return self.get_insufficient_data_structure()
    
    def get_insufficient_data_structure(self) -> Dict:
        """Return structure indicating insufficient data - prevents signal generation"""
        return {
            'near_support': False,
            'near_resistance': False,
            'strong_uptrend': False,
            'strong_downtrend': False,
            'bounce_potential': False,
            'distribution_signs': False,
            'price_range_position': 0.5,
            'has_sufficient_data': False
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
                         tech_score: int, confirm_score: int, trend_score: int) -> Optional[Dict]:
        """Create BUY signal with adaptive entry logic based on market structure and confluence"""
        try:
            entry_candidates = []

            # Always consider current price
            entry_candidates.append({
                'price': current_price,
                'score': 0.5,
                'reason': 'current_price'
            })

            # Score all support zones
            for zone in confluence_zones:
                if zone['zone_type'] == 'support':
                    distance = abs(zone['price'] - current_price) / current_price
                    score = 1.0 - distance
                    if confirm_score >= 2:
                        score += 0.15
                    if zone.get('confluence', 0) > 1:
                        score += 0.1 * zone['confluence']
                    if confirm_score >= 2 and distance > 0.01:
                        score += 0.05
                    entry_candidates.append({
                        'price': zone['price'],
                        'score': score,
                        'reason': f"support_{zone.get('label', '')}"
                    })

            # Volume-based entry
            if (volume_entry.get('confidence', 0) > 0.5 and
                volume_entry.get('entry_price', current_price) < current_price):
                score = 0.7
                if confirm_score >= 2:
                    score += 0.1
                entry_candidates.append({
                    'price': volume_entry['entry_price'],
                    'score': score,
                    'reason': 'volume_profile'
                })

            # Lower Bollinger Band
            bb_lower = latest.get('bb_lower', current_price * 0.98)
            if bb_lower < current_price:
                score = 0.6
                if confirm_score >= 2:
                    score += 0.05
                entry_candidates.append({
                    'price': bb_lower,
                    'score': score,
                    'reason': 'bb_lower'
                })

            # If trending, prefer dynamic levels (e.g., SMA20)
            if confirm_score >= 2 and trend_score >= 2:
                sma_20 = latest.get('sma_20', current_price)
                entry_candidates.append({
                    'price': sma_20,
                    'score': 0.8,
                    'reason': 'sma_20'
                })

            # Select the candidate with the highest score
            best_entry = max(entry_candidates, key=lambda x: x['score'])
            optimal_entry = best_entry['price']

            distance_pct = abs(optimal_entry - current_price) / current_price

            # Order type logic for BUY
            if distance_pct > 0.01 or latest.get('stoch_rsi_k', 50) < 30:
                order_type = 'limit'
            else:
                order_type = 'market'
                optimal_entry = current_price

            # Stop loss and take profit for BUY
            resistances = [zone['price'] for zone in confluence_zones if zone['zone_type'] == 'resistance' and zone['price'] > optimal_entry]
            if not resistances:
                self.logger.info(f"No resistance found above entry for {symbol_data['symbol']}, skipping BUY signal.")
                return None
            else:
                atr = latest.get('atr', current_price * 0.02)
                stop_loss = optimal_entry - (3.0 * atr)
                take_profit_1 = min(resistances)
                further_resistances = [r for r in resistances if r > take_profit_1]
                take_profit_2 = max(further_resistances) if further_resistances else take_profit_1

                # Risk-reward calculation for BUY
                risk_amount = optimal_entry - stop_loss
                reward_amount = take_profit_2 - optimal_entry
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

                # Confidence calculation for BUY
                base_confidence = 55
                base_confidence += tech_score * 3.5
                base_confidence += trend_score * 3.5
                base_confidence += confirm_score * 3.5
                if any(c['reason'].startswith('support') for c in entry_candidates):
                    base_confidence += 6
                if volume_entry.get('confidence', 0) > 0.6:
                    base_confidence += 4
                confidence = min(88, base_confidence)

                return {
                    'symbol': symbol_data['symbol'],
                    'side': 'Buy',
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
                        'confirm_score': confirm_score,
                        'support_zones_count': len([z for z in confluence_zones if z['zone_type'] == 'support']),
                        'volume_confidence': volume_entry.get('confidence', 0),
                        'entry_reason': best_entry['reason']
                    }
                }

        except Exception as e:
            self.logger.error(f"BUY signal creation error: {e}")
            return None

    def create_sell_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                          volume_entry: Dict, confluence_zones: List[Dict],
                          tech_score: int, confirm_score: int, trend_score: int) -> Optional[Dict]:
        """Create SELL signal with adaptive entry logic based on market structure and confluence"""
        try:
            entry_candidates = []

            # Always consider current price
            entry_candidates.append({
                'price': current_price,
                'score': 0.5,
                'reason': 'current_price'
            })

            # Score all resistance zones
            for zone in confluence_zones:
                if zone['zone_type'] == 'resistance':
                    distance = abs(zone['price'] - current_price) / current_price
                    score = 1.0 - distance
                    if confirm_score >= 2:
                        score += 0.15
                    if zone.get('confluence', 0) > 1:
                        score += 0.1 * zone['confluence']
                    if confirm_score >= 2 and distance > 0.01:
                        score += 0.05
                    entry_candidates.append({
                        'price': zone['price'],
                        'score': score,
                        'reason': f"resistance_{zone.get('label', '')}"
                    })

            # Volume-based entry
            if (volume_entry.get('confidence', 0) > 0.5 and
                volume_entry.get('entry_price', current_price) > current_price):
                score = 0.7
                if confirm_score >= 2:
                    score += 0.1
                entry_candidates.append({
                    'price': volume_entry['entry_price'],
                    'score': score,
                    'reason': 'volume_profile'
                })

            # Upper Bollinger Band
            bb_upper = latest.get('bb_upper', current_price * 1.02)
            if bb_upper > current_price:
                score = 0.6
                if confirm_score >= 2:
                    score += 0.05
                entry_candidates.append({
                    'price': bb_upper,
                    'score': score,
                    'reason': 'bb_upper'
                })

            # If trending, prefer dynamic levels (e.g., SMA20)
            if confirm_score >= 2 and trend_score >= 2:
                sma_20 = latest.get('sma_20', current_price)
                entry_candidates.append({
                    'price': sma_20,
                    'score': 0.8,
                    'reason': 'sma_20'
                })

            # Select the candidate with the highest score
            best_entry = max(entry_candidates, key=lambda x: x['score'])
            optimal_entry = best_entry['price']

            distance_pct = abs(optimal_entry - current_price) / current_price

            # Order type logic for SELL
            if distance_pct > 0.01 or latest.get('stoch_rsi_k', 50) > 75:
                order_type = 'limit'
            else:
                order_type = 'market'
                optimal_entry = current_price

            supports = [zone['price'] for zone in confluence_zones if zone['zone_type'] == 'support' and zone['price'] < optimal_entry]
            if not supports:
                self.logger.info(f"No support found below entry for {symbol_data['symbol']}, skipping SELL signal.")
                return None
            else:
                atr = latest.get('atr', current_price * 0.02)
                stop_loss = optimal_entry + (3.0 * atr)
                take_profit_1 = max(supports)
                further_supports = [s for s in supports if s < take_profit_1]
                take_profit_2 = min(further_supports) if further_supports else take_profit_1

                # Risk-reward calculation for SELL
                risk_amount = stop_loss - optimal_entry
                reward_amount = optimal_entry - take_profit_2
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

                # Confidence calculation for SELL
                base_confidence = 52
                base_confidence += tech_score * 3.5
                base_confidence += trend_score * 3.5
                base_confidence += confirm_score * 3.5
                if any(c['reason'].startswith('resistance') for c in entry_candidates):
                    base_confidence += 6
                if volume_entry.get('confidence', 0) > 0.6:
                    base_confidence += 4
                confidence = min(88, base_confidence)

                return {
                    'symbol': symbol_data['symbol'],
                    'side': 'Sell',
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
                        'confirm_score': confirm_score,
                        'resistance_zones_count': len([z for z in confluence_zones if z['zone_type'] == 'resistance']),
                        'volume_confidence': volume_entry.get('confidence', 0),
                        'entry_reason': best_entry['reason']
                    }
                }

        except Exception as e:
            self.logger.error(f"SELL signal creation error: {e}")
            return None
        
    def validate_signal_quality(self, signal: Dict, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """Stricter signal validation"""
        try:
            if not signal or not market_structure.get('has_sufficient_data', False):
                return None
            
            # Higher minimum thresholds
            min_confidence = 55 if signal['side'] == 'buy' else 58
            min_rr = 2.0 if signal['side'] == 'buy' else 2.2
            
            if signal['confidence'] < min_confidence:
                self.logger.debug(f"Signal filtered: {signal['symbol']} confidence too low")
                return None
            
            if signal['risk_reward_ratio'] < min_rr:
                self.logger.debug(f"Signal filtered: {signal['symbol']} R/R too low")
                return None
            
            # Additional stochastic RSI validation
            latest = df.iloc[-1]
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            
            if signal['side'] == 'buy' and stoch_rsi_k > 40:
                self.logger.debug(f"BUY signal filtered: {signal['symbol']} Stoch RSI not oversold enough")
                return None
                
            if signal['side'] == 'sell' and stoch_rsi_k < 65:
                self.logger.debug(f"SELL signal filtered: {signal['symbol']} Stoch RSI not overbought enough")
                return None
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return None

    def rank_opportunities_with_mtf(self, signals: List[Dict], dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """
        Enhanced ranking that balances signal quality, accessibility, and likelihood of hitting profit before loss.
        Optionally pass a dict of DataFrames keyed by symbol for profit likelihood calculation.
        """
        try:
            opportunities = []
            
            for signal in signals:
                # Get signal quality metrics
                confidence = signal['confidence']
                original_confidence = signal.get('original_confidence', confidence)
                mtf_boost = confidence - original_confidence

                # 1. SIGNAL CONFIDENCE (35% weight)
                confidence_score = confidence / 100

                # 2. RISK-REWARD RATIO (25% weight)
                analysis = signal.get('analysis', {})
                risk_assessment = analysis.get('risk_assessment', {})
                rr_ratio = risk_assessment.get('risk_reward_ratio', signal.get('risk_reward_ratio', 1))
                rr_score = min(1.0, rr_ratio / 3.0)

                # 3. MARKET STRUCTURE ALIGNMENT (20% weight)
                entry_methods = signal.get('entry_methods', {})
                structure_score = 0
                if 'structure_score' in entry_methods:
                    structure_score = min(1.0, entry_methods['structure_score'] / 2.5)
                elif 'confluence_zones' in entry_methods:
                    structure_score = min(1.0, entry_methods['confluence_zones'] / 1.5)

                # 4. VOLUME QUALITY (10% weight)
                volume_score = min(1.0, signal['volume_24h'] / 20_000_000)

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
                distance_score = max(0, 1 - distance * 6)

                # 7. ORDER TYPE (2% weight)
                order_type_score = 1.0 if signal['order_type'] == 'market' else 0.97

                # 8. PROFIT LIKELIHOOD (NEW, 10% weight, subtracted from volume/structure if needed)
                profit_likelihood = 0.5  # Default neutral
                latest = None
                if dfs and signal['symbol'] in dfs:
                    latest = dfs[signal['symbol']].iloc[-1]
                elif 'analysis' in signal and 'technical_summary' in signal['analysis']:
                    # Try to get latest from analysis if available
                    latest = signal['analysis']['technical_summary']
                if latest is not None:
                    profit_likelihood = self.estimate_profit_likelihood(signal, latest)
                # Weight: 10%
                
                # ===== CALCULATE TOTAL SCORE =====
                total_score = (
                    confidence_score * 0.30 +      # 30% - Signal quality
                    rr_score * 0.22 +              # 22% - Risk management
                    structure_score * 0.16 +       # 16% - Market structure
                    volume_score * 0.08 +          # 8% - Volume support
                    mtf_bonus +                    # 15% - MTF confirmation (additive)
                    distance_score * 0.03 +        # 3% - Entry distance
                    order_type_score * 0.01 +      # 1% - Order type
                    profit_likelihood * 0.15       # 15% - Likelihood of hitting profit before loss
                )

                # ===== TUNED PRIORITY SYSTEM =====
                mtf_status = signal.get('mtf_status', 'NONE')
                if confidence >= 70 and rr_ratio >= 3.0 and profit_likelihood > 0.7:
                    base_priority = 1200  # Exceptional signals
                elif confidence >= 60 and rr_ratio >= 2.5 and profit_likelihood > 0.6:
                    base_priority = 600   # High quality signals
                elif confidence >= 50 and rr_ratio >= 2.0 and profit_likelihood > 0.5:
                    base_priority = 300   # Good signals
                elif confidence >= 45 and rr_ratio >= 1.8:
                    base_priority = 120   # Decent signals
                else:
                    base_priority = 60    # Marginal signals

                mtf_modifier = {
                    'STRONG': 1.4,
                    'PARTIAL': 1.15,
                    'NONE': 1.0,
                    'DISABLED': 1.0
                }.get(mtf_status, 1.0)

                final_priority = int(base_priority * mtf_modifier)

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
                    'profit_likelihood': profit_likelihood,
                    'original_confidence': original_confidence,
                    'mtf_boost': mtf_boost,
                    'confirmed_timeframes': confirmed_timeframes,
                    'conflicting_timeframes': conflicting_timeframes
                })

            # Sort by priority first, then by score, then by profit likelihood
            opportunities.sort(key=lambda x: (x['priority'], x['score'], x['profit_likelihood']), reverse=True)
            return opportunities[:self.config.charts_per_batch]

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

    def estimate_profit_likelihood(self, signal: Dict, latest: pd.Series) -> float:
        """
        Estimate the likelihood of hitting take profit before stop loss.
        Returns a score between 0 and 1 (higher = more likely to hit profit first).
        """
        entry = signal['entry_price']
        stop = signal['stop_loss']
        tp = signal['take_profit_1']  # or use auto_close_profit_at if you want
        atr = latest.get('atr', entry * 0.02)
        side = signal['side'].lower()

        # Distance to profit/loss
        if side == 'buy':
            dist_profit = tp - entry
            dist_loss = entry - stop
            direction_factor = 1 if latest.get('trend', {}).get('direction', '') in ['bullish', 'strong_bullish'] else 0.8
        else:
            dist_profit = entry - tp
            dist_loss = stop - entry
            direction_factor = 1 if latest.get('trend', {}).get('direction', '') in ['bearish', 'strong_bearish'] else 0.8

        # Normalize by ATR (volatility)
        profit_moves = dist_profit / atr if atr > 0 else 0
        loss_moves = dist_loss / atr if atr > 0 else 0

        # Simple likelihood: closer TP, further SL, and trend in favor
        if profit_moves + loss_moves == 0:
            return 0.5  # Neutral if no movement

        base_likelihood = profit_moves / (profit_moves + loss_moves)
        likelihood = base_likelihood * direction_factor

        # Clamp between 0 and 1
        return max(0, min(1, likelihood))
    
# Additional utility functions for debugging

def debug_signal_conditions(df: pd.DataFrame, symbol: str):
    """Debug function to check why signals are being generated"""
    latest = df.iloc[-1]
    
    print(f"\n=== DEBUG: {symbol} ===")
    print(f"RSI: {latest.get('rsi', 'Missing')}")
    print(f"Stoch RSI K: {latest.get('stoch_rsi_k', 'Missing')}")
    print(f"Stoch RSI D: {latest.get('stoch_rsi_d', 'Missing')}")
    print(f"MACD: {latest.get('macd', 'Missing')}")
    print(f"MACD Signal: {latest.get('macd_signal', 'Missing')}")
    print(f"Volume Ratio: {latest.get('volume_ratio', 'Missing')}")
    print(f"BB Position: {latest.get('bb_position', 'Missing')}")
    
    # Check buy conditions
    rsi = latest.get('rsi', 50)
    stoch_k = latest.get('stoch_rsi_k', 50)
    
    print(f"\nBuy Conditions Check:")
    print(f"RSI < 35: {rsi < 35} (RSI: {rsi})")
    print(f"Stoch RSI K < 30: {stoch_k < 30} (K: {stoch_k})")
    
    print(f"\nSell Conditions Check:")
    print(f"RSI > 70: {rsi > 70} (RSI: {rsi})")
    print(f"Stoch RSI K > 75: {stoch_k > 75} (K: {stoch_k})")

