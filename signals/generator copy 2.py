"""
Signal generation and ranking for the Enhanced Bybit Trading System.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from config.config import EnhancedSystemConfig


class SignalGenerator:
    """Signal generation and ranking system"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_enhanced_signal(self, df: pd.DataFrame, symbol_data: Dict, 
                            volume_entry: Dict, confluence_zones: List[Dict]) -> Optional[Dict]:
        """Generate enhanced trading signal using all analysis methods - FIXED LOGIC"""
        try:
            latest = df.iloc[-1]
            current_price = symbol_data['current_price']
            
            # Get all indicator values
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            ema_12 = latest.get('ema_12', current_price)
            ema_26 = latest.get('ema_26', current_price)
            volume_ratio = latest.get('volume_ratio', 1)
            bb_position = latest.get('bb_position', 0.5)
            
            # Stochastic RSI values
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            stoch_rsi_d = latest.get('stoch_rsi_d', 50)
            
            # Ichimoku signals
            ichimoku_bullish = latest.get('ichimoku_bullish', False)
            ichimoku_bearish = latest.get('ichimoku_bearish', False)
            
            signal = None
            
            # ===== FIXED: BUY CONDITIONS =====
            # Primary trend and momentum conditions
            primary_buy_conditions = [
                30 < rsi < 75,                    # RSI not overbought, some room to go up
                macd > macd_signal,               # MACD bullish crossover
                ema_12 > ema_26,                  # Short EMA above long EMA (uptrend)
                current_price > latest.get('sma_20', current_price),  # Price above SMA20
                volume_ratio > 1.0,               # Above average volume
                bb_position < 0.8                 # Not too close to upper BB (room to grow)
            ]
            
            # ===== FIXED: Stochastic RSI Analysis =====
            # Stoch RSI signals for BUY
            stoch_rsi_oversold = stoch_rsi_k < 30          # Oversold condition
            stoch_rsi_bullish_cross = stoch_rsi_k > stoch_rsi_d  # %K above %D (bullish)
            stoch_rsi_recovering = stoch_rsi_k > 20        # Not in deep oversold
            
            # Stoch RSI BUY signals
            stoch_rsi_buy_signals = [
                stoch_rsi_oversold,               # In oversold territory (bullish setup)
                stoch_rsi_recovering,             # Not too deep (can recover)
                stoch_rsi_bullish_cross           # Bullish crossover
            ]
            
            primary_buy_score = sum(primary_buy_conditions)
            stoch_rsi_buy_score = sum(stoch_rsi_buy_signals)
            
            # Enhanced BUY signal generation
            if primary_buy_score >= 4:  # Need at least 4/6 primary conditions
                # Build entry candidates
                entry_candidates = [current_price]
                
                if volume_entry['confidence'] > 0.5:
                    entry_candidates.append(volume_entry['entry_price'])
                
                buy_zones = [zone for zone in confluence_zones 
                        if zone['zone_type'] == 'support' and zone['price'] < current_price]
                if buy_zones:
                    entry_candidates.append(buy_zones[0]['price'])
                
                # Add Ichimoku support levels if available
                if latest.get('ichimoku_kijun', 0) > 0 and latest['ichimoku_kijun'] < current_price:
                    entry_candidates.append(latest['ichimoku_kijun'])
                
                optimal_entry = np.mean(entry_candidates)
                distance_pct = abs(optimal_entry - current_price) / current_price
                
                # Order type logic - FIXED
                if distance_pct > 0.005 or stoch_rsi_oversold:  # Use limit if oversold or far entry
                    order_type = 'limit'
                else:
                    order_type = 'market'
                    optimal_entry = current_price
                
                # Calculate stops and targets
                atr = latest.get('atr', current_price * 0.02)
                stop_loss = optimal_entry - (2.5 * atr)
                take_profit_1 = optimal_entry + (4 * atr)
                take_profit_2 = optimal_entry + (7 * atr)
                
                risk_amount = optimal_entry - stop_loss
                reward_amount = take_profit_2 - optimal_entry
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                
                # ===== FIXED: Confidence Calculation =====
                base_confidence = 60
                
                # Primary condition bonuses
                if volume_entry['confidence'] > 0.7:
                    base_confidence += 10
                if len(buy_zones) > 0:
                    base_confidence += 8
                if rsi < 60:  # Not overbought
                    base_confidence += 5
                if volume_ratio > 2:  # High volume
                    base_confidence += 5
                
                # ===== FIXED: Stochastic RSI Bonus Logic =====
                if stoch_rsi_oversold and stoch_rsi_bullish_cross:
                    base_confidence += 12  # Strong oversold bounce setup
                elif stoch_rsi_oversold:
                    base_confidence += 8   # Oversold setup
                elif stoch_rsi_bullish_cross:
                    base_confidence += 6   # Bullish crossover
                
                # Ichimoku bonus
                if ichimoku_bullish:
                    base_confidence += 8
                
                signal = {
                    'symbol': symbol_data['symbol'],
                    'side': 'buy',
                    'order_type': order_type,
                    'entry_price': optimal_entry,
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'risk_reward_ratio': risk_reward_ratio,
                    'confidence': min(95, base_confidence),
                    'signal_type': 'enhanced_multi_method',
                    'volume_24h': symbol_data['volume_24h'],
                    'price_change_24h': symbol_data['price_change_24h'],
                    'entry_methods': {
                        'volume_profile': volume_entry,
                        'confluence_zones': len(buy_zones),
                        'technical_analysis': True,
                        'stoch_rsi_oversold': stoch_rsi_oversold,
                        'stoch_rsi_bullish_cross': stoch_rsi_bullish_cross,
                        'ichimoku': ichimoku_bullish
                    }
                }
            
            # ===== FIXED: SELL CONDITIONS =====
            # Primary trend and momentum conditions for SELL
            primary_sell_conditions = [
                25 < rsi < 70,                    # RSI not oversold, some room to go down
                macd < macd_signal,               # MACD bearish crossover
                ema_12 < ema_26,                  # Short EMA below long EMA (downtrend)
                current_price < latest.get('sma_20', current_price),  # Price below SMA20
                volume_ratio > 1.0,               # Above average volume
                bb_position > 0.2                 # Not too close to lower BB
            ]
            
            # ===== FIXED: Stochastic RSI for SELL =====
            stoch_rsi_overbought = stoch_rsi_k > 70        # Overbought condition
            stoch_rsi_bearish_cross = stoch_rsi_k < stoch_rsi_d  # %K below %D (bearish)
            stoch_rsi_topping = stoch_rsi_k < 80           # Not in extreme overbought
            
            # Stoch RSI SELL signals
            stoch_rsi_sell_signals = [
                stoch_rsi_overbought,             # In overbought territory (bearish setup)
                stoch_rsi_topping,                # Not too extreme (can fall)
                stoch_rsi_bearish_cross           # Bearish crossover
            ]
            
            primary_sell_score = sum(primary_sell_conditions)
            stoch_rsi_sell_score = sum(stoch_rsi_sell_signals)
            
            # Enhanced SELL signal generation - only if no BUY signal
            if not signal and primary_sell_score >= 4:  # Need at least 4/6 primary conditions
                # Build entry candidates
                entry_candidates = [current_price]
                
                sell_zones = [zone for zone in confluence_zones 
                            if zone['zone_type'] == 'resistance' and zone['price'] > current_price]
                if sell_zones:
                    entry_candidates.append(sell_zones[0]['price'])
                
                if latest.get('ichimoku_kijun', 0) > 0 and latest['ichimoku_kijun'] > current_price:
                    entry_candidates.append(latest['ichimoku_kijun'])
                
                optimal_entry = np.mean(entry_candidates)
                distance_pct = abs(optimal_entry - current_price) / current_price
                
                # Order type logic - FIXED
                if distance_pct > 0.005 or stoch_rsi_overbought:  # Use limit if overbought or far entry
                    order_type = 'limit'
                else:
                    order_type = 'market'
                    optimal_entry = current_price
                
                atr = latest.get('atr', current_price * 0.02)
                stop_loss = optimal_entry + (2.5 * atr)
                take_profit_1 = optimal_entry - (4 * atr)
                take_profit_2 = optimal_entry - (7 * atr)
                
                risk_amount = stop_loss - optimal_entry
                reward_amount = optimal_entry - take_profit_2
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                
                # ===== FIXED: SELL Confidence Calculation =====
                base_confidence = 60
                
                # Primary condition bonuses
                if len(sell_zones) > 0:
                    base_confidence += 8
                if rsi > 40:  # Not oversold
                    base_confidence += 5
                if volume_ratio > 2:  # High volume
                    base_confidence += 5
                
                # ===== FIXED: Stochastic RSI SELL Bonus Logic =====
                if stoch_rsi_overbought and stoch_rsi_bearish_cross:
                    base_confidence += 12  # Strong overbought reversal setup
                elif stoch_rsi_overbought:
                    base_confidence += 8   # Overbought setup
                elif stoch_rsi_bearish_cross:
                    base_confidence += 6   # Bearish crossover
                
                # Ichimoku bonus
                if ichimoku_bearish:
                    base_confidence += 8
                
                signal = {
                    'symbol': symbol_data['symbol'],
                    'side': 'sell',
                    'order_type': order_type,
                    'entry_price': optimal_entry,
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'risk_reward_ratio': risk_reward_ratio,
                    'confidence': min(95, base_confidence),
                    'signal_type': 'enhanced_multi_method',
                    'volume_24h': symbol_data['volume_24h'],
                    'price_change_24h': symbol_data['price_change_24h'],
                    'entry_methods': {
                        'volume_profile': volume_entry,
                        'confluence_zones': len(sell_zones),
                        'technical_analysis': True,
                        'stoch_rsi_overbought': stoch_rsi_overbought,
                        'stoch_rsi_bearish_cross': stoch_rsi_bearish_cross,
                        'ichimoku': ichimoku_bearish
                    }
                }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return None

    def rank_opportunities_with_mtf(self, signals: List[Dict]) -> List[Dict]:
        """Rank trading opportunities with MTF confirmation priority"""
        try:
            opportunities = []
            
            for signal in signals:
                # Calculate opportunity score with MTF boost
                confidence_score = signal['confidence'] / 100
                original_confidence = signal.get('original_confidence', signal['confidence'])
                mtf_boost = signal['confidence'] - original_confidence
                
                # Volume score
                volume_score = min(1.0, signal['volume_24h'] / 100_000_000)
                
                # Risk-reward score
                analysis = signal.get('analysis', {})
                risk_assessment = analysis.get('risk_assessment', {})
                rr_ratio = risk_assessment.get('risk_reward_ratio', 1)
                rr_score = min(1.0, rr_ratio / 3.0)
                
                # Distance score
                distance = abs(signal['entry_price'] - signal['current_price']) / signal['current_price']
                distance_score = max(0, 1 - distance * 20)
                
                # Order type preference
                order_type_score = 1.0 if signal['order_type'] == 'market' else 0.8
                
                # MTF bonus score
                mtf_analysis = signal.get('mtf_analysis', {})
                confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
                total_timeframes = len(self.config.confirmation_timeframes)
                
                if total_timeframes > 0:
                    mtf_score = confirmed_count / total_timeframes
                    mtf_bonus = mtf_score * 0.3  # Up to 30% bonus for full confirmation
                else:
                    mtf_score = 0
                    mtf_bonus = 0
                
                # Calculate total score with MTF consideration
                total_score = (
                    confidence_score * 0.25 +
                    volume_score * 0.15 +
                    rr_score * 0.15 +
                    distance_score * 0.1 +
                    order_type_score * 0.1 +
                    mtf_bonus * 1.0
                )
                
                # Get MTF confirmation details
                confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
                conflicting_timeframes = mtf_analysis.get('conflicting_timeframes', [])
                
                opportunities.append({
                    'rank': 0,
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
                    'chart_file': signal.get('chart_file', 'Not available'),
                    'signal_type': signal.get('signal_type', 'enhanced'),
                    'technical_summary': analysis.get('technical_summary', {}),
                    'risk_level': risk_assessment.get('risk_level', 'Unknown'),
                    # MTF specific fields
                    'mtf_status': signal.get('mtf_status', 'DISABLED'),
                    'mtf_confirmed': confirmed_timeframes,
                    'mtf_conflicting': conflicting_timeframes,
                    'mtf_confirmation_count': confirmed_count,
                    'mtf_total_timeframes': total_timeframes,
                    'mtf_confirmation_strength': mtf_analysis.get('confirmation_strength', 0),
                    'priority_boost': signal.get('priority_boost', 0)
                })
            
            # Sort by MTF priority first, then total score
            opportunities.sort(key=lambda x: (x['priority_boost'], x['total_score']), reverse=True)
            
            # Assign ranks
            for i, opp in enumerate(opportunities):
                opp['rank'] = i + 1
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"MTF opportunity ranking error: {e}")
            return []
    
    def create_technical_summary(self, df: pd.DataFrame) -> Dict:
        """Create comprehensive technical summary"""
        try:
            latest = df.iloc[-1]
            
            # Trend Analysis
            sma_20 = latest.get('sma_20', 0)
            sma_50 = latest.get('sma_50', 0)
            ema_12 = latest.get('ema_12', 0)
            ema_26 = latest.get('ema_26', 0)
            
            trend_score = 0
            if latest['close'] > sma_20 > sma_50:
                trend_score += 2
            if ema_12 > ema_26:
                trend_score += 1
            if latest.get('macd', 0) > latest.get('macd_signal', 0):
                trend_score += 1
            
            trend_strength = 'Strong Bullish' if trend_score >= 3 else 'Bullish' if trend_score == 2 else 'Neutral' if trend_score == 1 else 'Bearish'
            
            # Momentum Analysis
            rsi = latest.get('rsi', 50)
            momentum_status = 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'
            
            # Volatility Analysis
            bb_width = latest.get('bb_width', 0)
            volatility_status = 'High' if bb_width > 0.04 else 'Low' if bb_width < 0.02 else 'Normal'
            
            # Volume Analysis
            volume_ratio = latest.get('volume_ratio', 1)
            volume_status = 'High' if volume_ratio > 1.5 else 'Low' if volume_ratio < 0.8 else 'Normal'
            
            # Stochastic RSI summary
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            stoch_rsi_d = latest.get('stoch_rsi_d', 50)
            stoch_rsi_status = 'Overbought' if stoch_rsi_k > 80 else 'Oversold' if stoch_rsi_k < 20 else 'Neutral'
            stoch_rsi_trend = 'Bullish' if stoch_rsi_k > stoch_rsi_d else 'Bearish' if stoch_rsi_k < stoch_rsi_d else 'Neutral'
            
            # Ichimoku summary
            ichimoku_status = 'Bullish' if latest.get('ichimoku_bullish', False) else \
                            'Bearish' if latest.get('ichimoku_bearish', False) else 'Neutral'
            
            return {
                'trend': {
                    'direction': trend_strength,
                    'score': trend_score,
                    'sma_alignment': sma_20 > sma_50,
                    'ema_alignment': ema_12 > ema_26
                },
                'momentum': {
                    'rsi': rsi,
                    'status': momentum_status,
                    'macd_bullish': latest.get('macd', 0) > latest.get('macd_signal', 0),
                    'stoch_rsi_k': stoch_rsi_k,
                    'stoch_rsi_d': stoch_rsi_d,
                    'stoch_rsi_status': stoch_rsi_status,
                    'stoch_rsi_trend': stoch_rsi_trend
                },
                'volatility': {
                    'status': volatility_status,
                    'bb_width': bb_width,
                    'atr_pct': latest.get('atr', 0) / latest['close'] * 100 if latest['close'] > 0 else 0
                },
                'volume': {
                    'status': volume_status,
                    'ratio': volume_ratio,
                    'spike_detected': volume_ratio > 2.0
                },
                'ichimoku': {
                    'status': ichimoku_status,
                    'price_above_cloud': latest['close'] > max(latest.get('ichimoku_span_a', 0), 
                                                            latest.get('ichimoku_span_b', 0)),
                    'tenkan_above_kijun': latest.get('ichimoku_tenkan', 0) > latest.get('ichimoku_kijun', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Technical summary error: {e}")
            return {}
    
    def assess_signal_risk(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Assess risk for the trading signal"""
        try:
            latest = df.iloc[-1]
            
            # Price volatility risk
            returns = df['close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(24)
            
            # Distance risk
            distance_risk = abs(signal['entry_price'] - signal['current_price']) / signal['current_price']
            
            # Market condition risk
            rsi = latest.get('rsi', 50)
            bb_position = latest.get('bb_position', 0.5)
            
            condition_risk = 0
            if rsi > 80 or rsi < 20:
                condition_risk += 0.3
            if bb_position > 0.9 or bb_position < 0.1:
                condition_risk += 0.2
            
            # Volume risk
            volume_ratio = latest.get('volume_ratio', 1)
            volume_risk = 0.1 if volume_ratio < 0.5 else 0
            
            # MTF risk adjustment
            mtf_analysis = signal.get('mtf_analysis', {})
            mtf_risk_reduction = len(mtf_analysis.get('confirmed_timeframes', [])) * 0.1
            
            # Calculate total risk score
            base_risk = volatility * 2 + distance_risk * 5 + condition_risk + volume_risk
            total_risk = max(0.1, min(1.0, base_risk - mtf_risk_reduction))
            
            # Risk-reward ratio
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
            
            return {
                'total_risk_score': total_risk,
                'volatility_risk': volatility,
                'distance_risk': distance_risk,
                'condition_risk': condition_risk,
                'volume_risk': volume_risk,
                'mtf_risk_reduction': mtf_risk_reduction,
                'risk_reward_ratio': risk_reward_ratio,
                'risk_level': 'High' if total_risk > 0.7 else 'Medium' if total_risk > 0.4 else 'Low'
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return {'total_risk_score': 0.5, 'risk_level': 'Unknown'}