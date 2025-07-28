"""
Enhanced Signal Generation for the Bybit Trading System.
CONTEXT-AWARE REASONING: Adaptive thresholds and pullback strategies
- Dynamic signal thresholds based on market context
- "Wait for pullback" logic with specific levels
- Multi-scenario analysis (immediate vs pullback vs breakout)
- Confidence modulation based on trend strength
- Weighted scoring system replacing rigid AND conditions
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from config.config import EnhancedSystemConfig


class SignalGenerator:
    """Enhanced signal generation with context-aware reasoning"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_enhanced_signal(self, df: pd.DataFrame, symbol_data: Dict, 
                                volume_entry: Dict, confluence_zones: List[Dict]) -> Optional[Dict]:
        """Context-aware signal generation with pullback strategies"""
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
            
            # Market structure analysis
            market_structure = self.analyze_market_structure(df, current_price)
            if not self.has_sufficient_market_data(market_structure):
                self.logger.debug(f"Insufficient market structure data for {symbol}")
                return None
            
            # ===== CONTEXT-AWARE ANALYSIS =====
            trend_context = self.analyze_trend_context(df, latest)
            pullback_analysis = self.analyze_pullback_opportunity(df, current_price, confluence_zones)
            scenario_analysis = self.generate_scenario_analysis(df, current_price, latest, confluence_zones)
            
            # ===== DYNAMIC THRESHOLD CALCULATION =====
            buy_thresholds = self.calculate_dynamic_buy_thresholds(trend_context, market_structure)
            sell_thresholds = self.calculate_dynamic_sell_thresholds(trend_context, market_structure)
            
            # ===== SIGNAL GENERATION WITH WEIGHTED SCORING =====
            signal = None
            
            # Check for BUY opportunities
            buy_analysis = self.analyze_buy_opportunity(
                latest, current_price, buy_thresholds, market_structure, 
                trend_context, pullback_analysis, scenario_analysis
            )
            
            if buy_analysis['should_signal']:
                signal = self.create_context_aware_buy_signal(
                    symbol_data, current_price, latest, volume_entry, 
                    confluence_zones, buy_analysis, pullback_analysis, scenario_analysis
                )
            
            # Check for SELL opportunities (only if no BUY signal)
            if not signal:
                sell_analysis = self.analyze_sell_opportunity(
                    latest, current_price, sell_thresholds, market_structure, 
                    trend_context, pullback_analysis, scenario_analysis
                )
                
                if sell_analysis['should_signal']:
                    signal = self.create_context_aware_sell_signal(
                        symbol_data, current_price, latest, volume_entry, 
                        confluence_zones, sell_analysis, pullback_analysis, scenario_analysis
                    )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol_data.get('symbol', 'unknown')}: {e}")
            return None

    def analyze_trend_context(self, df: pd.DataFrame, latest: pd.Series) -> Dict:
        """Analyze the current trend context for dynamic threshold adjustment"""
        try:
            if len(df) < 50:
                return {'strength': 'weak', 'direction': 'neutral', 'momentum': 'low'}
            
            # Multiple timeframe trend analysis
            short_trend = self.get_trend_direction(df.tail(20))
            medium_trend = self.get_trend_direction(df.tail(50))
            
            # Moving averages alignment
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            ema_12 = latest.get('ema_12', latest['close'])
            ema_26 = latest.get('ema_26', latest['close'])
            
            # Trend strength calculation
            price_above_sma20 = latest['close'] > sma_20
            price_above_sma50 = latest['close'] > sma_50
            sma20_above_sma50 = sma_20 > sma_50
            ema12_above_ema26 = ema_12 > ema_26
            
            bullish_signals = sum([price_above_sma20, price_above_sma50, sma20_above_sma50, ema12_above_ema26])
            
            # Determine trend strength and direction
            if bullish_signals >= 3:
                if short_trend == medium_trend == 'bullish':
                    strength = 'very_strong'
                else:
                    strength = 'strong'
                direction = 'bullish'
            elif bullish_signals <= 1:
                if short_trend == medium_trend == 'bearish':
                    strength = 'very_strong'
                else:
                    strength = 'strong'
                direction = 'bearish'
            else:
                strength = 'moderate'
                direction = 'neutral'
            
            # Momentum analysis
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            
            if rsi > 60 and macd > macd_signal:
                momentum = 'strong_bullish'
            elif rsi < 40 and macd < macd_signal:
                momentum = 'strong_bearish'
            elif rsi > 50 and macd > macd_signal:
                momentum = 'bullish'
            elif rsi < 50 and macd < macd_signal:
                momentum = 'bearish'
            else:
                momentum = 'neutral'
            
            return {
                'strength': strength,
                'direction': direction,
                'momentum': momentum,
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'bullish_signals': bullish_signals,
                'ma_alignment_bullish': sma20_above_sma50 and ema12_above_ema26
            }
            
        except Exception as e:
            self.logger.error(f"Trend context analysis error: {e}")
            return {'strength': 'weak', 'direction': 'neutral', 'momentum': 'low'}

    def get_trend_direction(self, df_subset: pd.DataFrame) -> str:
        """Determine trend direction for a given timeframe"""
        if len(df_subset) < 5:
            return 'neutral'
        
        start_price = df_subset.iloc[0]['close']
        end_price = df_subset.iloc[-1]['close']
        change_pct = (end_price - start_price) / start_price
        
        if change_pct > 0.03:
            return 'strong_bullish'
        elif change_pct > 0.01:
            return 'bullish'
        elif change_pct < -0.03:
            return 'strong_bearish'
        elif change_pct < -0.01:
            return 'bearish'
        else:
            return 'neutral'

    def analyze_pullback_opportunity(self, df: pd.DataFrame, current_price: float, 
                                   confluence_zones: List[Dict]) -> Dict:
        """Analyze if we should wait for a pullback or take immediate action"""
        try:
            if len(df) < 20:
                return {'should_wait': False, 'pullback_levels': [], 'confidence': 'low'}
            
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            price_range = recent_high - recent_low
            
            # Current position in range
            range_position = (current_price - recent_low) / price_range if price_range > 0 else 0.5
            
            # Identify key pullback levels
            pullback_levels = []
            
            # Support levels from confluence zones
            for zone in confluence_zones:
                if zone['zone_type'] == 'support' and zone['price'] < current_price:
                    distance_pct = (current_price - zone['price']) / current_price
                    if 0.01 <= distance_pct <= 0.05:  # 1-5% pullback
                        pullback_levels.append({
                            'price': zone['price'],
                            'type': 'support',
                            'confidence': zone.get('confluence', 1),
                            'distance_pct': distance_pct
                        })
            
            # Moving average levels
            latest = df.iloc[-1]
            sma_20 = latest.get('sma_20')
            if sma_20 and sma_20 < current_price:
                distance_pct = (current_price - sma_20) / current_price
                if 0.005 <= distance_pct <= 0.03:  # 0.5-3% pullback
                    pullback_levels.append({
                        'price': sma_20,
                        'type': 'sma_20',
                        'confidence': 2,
                        'distance_pct': distance_pct
                    })
            
            # Fibonacci retracement levels (if we have recent swing high/low)
            if len(df) >= 30:
                swing_high = df['high'].tail(30).max()
                swing_low = df['low'].tail(30).min()
                if swing_high > swing_low:
                    fib_618 = swing_high - (swing_high - swing_low) * 0.618
                    fib_382 = swing_high - (swing_high - swing_low) * 0.382
                    
                    for fib_level, fib_name in [(fib_618, 'fib_618'), (fib_382, 'fib_382')]:
                        if fib_level < current_price:
                            distance_pct = (current_price - fib_level) / current_price
                            if 0.01 <= distance_pct <= 0.04:
                                pullback_levels.append({
                                    'price': fib_level,
                                    'type': fib_name,
                                    'confidence': 3 if fib_name == 'fib_618' else 2,
                                    'distance_pct': distance_pct
                                })
            
            # Determine if we should wait for pullback
            should_wait = False
            confidence = 'low'
            
            if range_position > 0.8 and len(pullback_levels) > 0:  # Near top of range with good pullback levels
                should_wait = True
                confidence = 'high'
            elif range_position > 0.7 and len([p for p in pullback_levels if p['confidence'] >= 2]) > 0:
                should_wait = True
                confidence = 'medium'
            
            # Sort pullback levels by confidence and proximity
            pullback_levels.sort(key=lambda x: (x['confidence'], -x['distance_pct']), reverse=True)
            
            return {
                'should_wait': should_wait,
                'pullback_levels': pullback_levels[:3],  # Top 3 levels
                'confidence': confidence,
                'range_position': range_position,
                'recent_high': recent_high,
                'recent_low': recent_low
            }
            
        except Exception as e:
            self.logger.error(f"Pullback analysis error: {e}")
            return {'should_wait': False, 'pullback_levels': [], 'confidence': 'low'}

    def generate_scenario_analysis(self, df: pd.DataFrame, current_price: float, 
                                 latest: pd.Series, confluence_zones: List[Dict]) -> Dict:
        """Generate multiple trading scenarios like my manual analysis"""
        try:
            scenarios = {}
            
            # Get key levels
            resistances = [z['price'] for z in confluence_zones if z['zone_type'] == 'resistance' and z['price'] > current_price]
            supports = [z['price'] for z in confluence_zones if z['zone_type'] == 'support' and z['price'] < current_price]
            
            nearest_resistance = min(resistances) if resistances else current_price * 1.05
            nearest_support = max(supports) if supports else current_price * 0.95
            
            # Scenario 1: Immediate action (current levels)
            scenarios['immediate'] = {
                'action': 'market_entry',
                'entry_price': current_price,
                'stop_loss': nearest_support * 0.995,
                'take_profit': nearest_resistance * 0.995,
                'confidence': 'medium',
                'risk_reward': abs(nearest_resistance - current_price) / abs(current_price - nearest_support) if nearest_support != current_price else 1
            }
            
            # Scenario 2: Wait for pullback
            pullback_analysis = self.analyze_pullback_opportunity(df, current_price, confluence_zones)
            if pullback_analysis['pullback_levels']:
                best_pullback = pullback_analysis['pullback_levels'][0]
                scenarios['pullback'] = {
                    'action': 'wait_for_pullback',
                    'entry_price': best_pullback['price'],
                    'stop_loss': best_pullback['price'] * 0.985,
                    'take_profit': nearest_resistance * 0.995,
                    'confidence': 'high',
                    'pullback_type': best_pullback['type'],
                    'wait_distance_pct': best_pullback['distance_pct']
                }
            
            # Scenario 3: Breakout continuation
            if nearest_resistance:
                breakout_entry = nearest_resistance * 1.002
                next_resistance = None
                for r in sorted(resistances):
                    if r > nearest_resistance * 1.01:
                        next_resistance = r
                        break
                
                if not next_resistance:
                    next_resistance = nearest_resistance * 1.08
                
                scenarios['breakout'] = {
                    'action': 'breakout_entry',
                    'entry_price': breakout_entry,
                    'stop_loss': nearest_resistance * 0.995,
                    'take_profit': next_resistance * 0.99,
                    'confidence': 'medium',
                    'breakout_level': nearest_resistance
                }
            
            # Scenario 4: Breakdown/reversal
            if nearest_support:
                scenarios['breakdown'] = {
                    'action': 'short_on_breakdown',
                    'entry_price': nearest_support * 0.998,
                    'stop_loss': nearest_support * 1.01,
                    'take_profit': nearest_support * 0.95,
                    'confidence': 'low',
                    'breakdown_level': nearest_support
                }
            
            return scenarios
            
        except Exception as e:
            self.logger.error(f"Scenario analysis error: {e}")
            return {}

    def calculate_dynamic_buy_thresholds(self, trend_context: Dict, market_structure: Dict) -> Dict:
        """Calculate dynamic RSI and other thresholds for BUY signals based on context"""
        base_rsi_threshold = 35
        base_stoch_threshold = 30
        
        # Adjust based on trend strength
        if trend_context['strength'] == 'very_strong' and trend_context['direction'] == 'bullish':
            # In very strong uptrends, can buy at higher RSI levels
            rsi_threshold = min(55, base_rsi_threshold + 20)
            stoch_threshold = min(50, base_stoch_threshold + 20)
        elif trend_context['strength'] == 'strong' and trend_context['direction'] == 'bullish':
            rsi_threshold = min(45, base_rsi_threshold + 10)
            stoch_threshold = min(40, base_stoch_threshold + 10)
        elif trend_context['direction'] == 'bearish':
            # In bearish trends, need deeper oversold conditions
            rsi_threshold = max(25, base_rsi_threshold - 10)
            stoch_threshold = max(20, base_stoch_threshold - 10)
        else:
            rsi_threshold = base_rsi_threshold
            stoch_threshold = base_stoch_threshold
        
        # Adjust based on market structure
        if market_structure.get('near_support', False):
            rsi_threshold += 5
            stoch_threshold += 5
        
        return {
            'rsi_threshold': rsi_threshold,
            'stoch_threshold': stoch_threshold,
            'volume_threshold': 1.2 if trend_context['strength'] == 'very_strong' else 1.5,
            'trend_alignment_required': trend_context['strength'] in ['strong', 'very_strong']
        }

    def calculate_dynamic_sell_thresholds(self, trend_context: Dict, market_structure: Dict) -> Dict:
        """Calculate dynamic RSI and other thresholds for SELL signals based on context"""
        base_rsi_threshold = 70
        base_stoch_threshold = 75
        
        # Adjust based on trend strength
        if trend_context['strength'] == 'very_strong' and trend_context['direction'] == 'bearish':
            # In very strong downtrends, can sell at lower RSI levels
            rsi_threshold = max(45, base_rsi_threshold - 25)
            stoch_threshold = max(50, base_stoch_threshold - 25)
        elif trend_context['strength'] == 'strong' and trend_context['direction'] == 'bearish':
            rsi_threshold = max(60, base_rsi_threshold - 10)
            stoch_threshold = max(65, base_stoch_threshold - 10)
        elif trend_context['direction'] == 'bullish':
            # In bullish trends, need deeper overbought conditions
            rsi_threshold = min(80, base_rsi_threshold + 10)
            stoch_threshold = min(85, base_stoch_threshold + 10)
        else:
            rsi_threshold = base_rsi_threshold
            stoch_threshold = base_stoch_threshold
        
        # Adjust based on market structure
        if market_structure.get('near_resistance', False):
            rsi_threshold -= 5
            stoch_threshold -= 5
        
        return {
            'rsi_threshold': rsi_threshold,
            'stoch_threshold': stoch_threshold,
            'volume_threshold': 1.5 if trend_context['strength'] == 'very_strong' else 1.8,
            'trend_alignment_required': trend_context['strength'] in ['strong', 'very_strong']
        }

    def analyze_buy_opportunity(self, latest: pd.Series, current_price: float, 
                              thresholds: Dict, market_structure: Dict, 
                              trend_context: Dict, pullback_analysis: Dict, 
                              scenario_analysis: Dict) -> Dict:
        """Analyze BUY opportunity with weighted scoring"""
        try:
            rsi = latest.get('rsi', 50)
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            stoch_rsi_d = latest.get('stoch_rsi_d', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Weighted scoring system
            scores = {}
            
            # Technical indicators (40% weight)
            rsi_score = max(0, (thresholds['rsi_threshold'] - rsi) / thresholds['rsi_threshold']) if rsi <= thresholds['rsi_threshold'] else 0
            stoch_score = max(0, (thresholds['stoch_threshold'] - stoch_rsi_k) / thresholds['stoch_threshold']) if stoch_rsi_k <= thresholds['stoch_threshold'] else 0
            stoch_cross_score = 1 if stoch_rsi_k > stoch_rsi_d else 0
            macd_score = 1 if macd > macd_signal * 0.98 else 0
            
            scores['technical'] = (rsi_score * 0.4 + stoch_score * 0.3 + stoch_cross_score * 0.2 + macd_score * 0.1)
            
            # Market structure (25% weight)
            structure_score = 0
            if market_structure.get('near_support', False):
                structure_score += 0.4
            if not market_structure.get('strong_downtrend', False):
                structure_score += 0.3
            if market_structure.get('bounce_potential', False):
                structure_score += 0.3
            
            scores['structure'] = structure_score
            
            # Trend alignment (20% weight)
            trend_score = 0
            if trend_context['direction'] in ['bullish', 'strong_bullish']:
                trend_score += 0.6
            elif trend_context['direction'] == 'neutral':
                trend_score += 0.3
            
            if trend_context['momentum'] in ['bullish', 'strong_bullish']:
                trend_score += 0.4
            
            scores['trend'] = min(1.0, trend_score)
            
            # Volume confirmation (10% weight)
            volume_score = min(1.0, volume_ratio / thresholds['volume_threshold'])
            scores['volume'] = volume_score
            
            # Pullback opportunity (5% weight)
            pullback_score = 0
            if pullback_analysis['should_wait']:
                pullback_score = 0.3  # Slight penalty for not being at optimal level
            else:
                pullback_score = 1.0  # Good current level
            
            scores['pullback'] = pullback_score
            
            # Calculate total weighted score
            total_score = (
                scores['technical'] * 0.40 +
                scores['structure'] * 0.25 +
                scores['trend'] * 0.20 +
                scores['volume'] * 0.10 +
                scores['pullback'] * 0.05
            )
            
            # Determine signal strength
            should_signal = False
            signal_strength = 'none'
            
            if total_score >= 0.75:
                should_signal = True
                signal_strength = 'strong'
            elif total_score >= 0.60:
                should_signal = True
                signal_strength = 'moderate'
            elif total_score >= 0.45:
                should_signal = True
                signal_strength = 'weak'
            
            # Override logic for extreme conditions
            if rsi < 20 and market_structure.get('near_support', False):
                should_signal = True
                signal_strength = 'strong'
                total_score = max(total_score, 0.8)
            
            return {
                'should_signal': should_signal,
                'signal_strength': signal_strength,
                'total_score': total_score,
                'component_scores': scores,
                'recommended_action': self.get_recommended_buy_action(total_score, pullback_analysis, scenario_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Buy opportunity analysis error: {e}")
            return {'should_signal': False, 'signal_strength': 'none', 'total_score': 0}

    def analyze_sell_opportunity(self, latest: pd.Series, current_price: float, 
                               thresholds: Dict, market_structure: Dict, 
                               trend_context: Dict, pullback_analysis: Dict, 
                               scenario_analysis: Dict) -> Dict:
        """Analyze SELL opportunity with weighted scoring"""
        try:
            rsi = latest.get('rsi', 50)
            stoch_rsi_k = latest.get('stoch_rsi_k', 50)
            stoch_rsi_d = latest.get('stoch_rsi_d', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Weighted scoring system
            scores = {}
            
            # Technical indicators (40% weight)
            rsi_score = max(0, (rsi - thresholds['rsi_threshold']) / (100 - thresholds['rsi_threshold'])) if rsi >= thresholds['rsi_threshold'] else 0
            stoch_score = max(0, (stoch_rsi_k - thresholds['stoch_threshold']) / (100 - thresholds['stoch_threshold'])) if stoch_rsi_k >= thresholds['stoch_threshold'] else 0
            stoch_cross_score = 1 if stoch_rsi_k < stoch_rsi_d else 0
            macd_score = 1 if macd < macd_signal * 1.02 else 0
            
            scores['technical'] = (rsi_score * 0.4 + stoch_score * 0.3 + stoch_cross_score * 0.2 + macd_score * 0.1)
            
            # Market structure (25% weight)
            structure_score = 0
            if market_structure.get('near_resistance', False):
                structure_score += 0.4
            if not market_structure.get('strong_uptrend', False):
                structure_score += 0.3
            if market_structure.get('distribution_signs', False):
                structure_score += 0.3
            
            scores['structure'] = structure_score
            
            # Trend alignment (20% weight)
            trend_score = 0
            if trend_context['direction'] in ['bearish', 'strong_bearish']:
                trend_score += 0.6
            elif trend_context['direction'] == 'neutral':
                trend_score += 0.3
            
            if trend_context['momentum'] in ['bearish', 'strong_bearish']:
                trend_score += 0.4
            
            scores['trend'] = min(1.0, trend_score)
            
            # Volume confirmation (10% weight)
            volume_score = min(1.0, volume_ratio / thresholds['volume_threshold'])
            scores['volume'] = volume_score
            
            # Check if in strong uptrend (reduces sell signals)
            uptrend_penalty = 0
            if trend_context['direction'] in ['bullish', 'strong_bullish'] and trend_context['strength'] == 'very_strong':
                uptrend_penalty = 0.3
            
            scores['uptrend_penalty'] = uptrend_penalty
            
            # Calculate total weighted score
            total_score = (
                scores['technical'] * 0.40 +
                scores['structure'] * 0.25 +
                scores['trend'] * 0.20 +
                scores['volume'] * 0.10
            ) - uptrend_penalty
            
            total_score = max(0, total_score)  # Ensure non-negative
            
            # Determine signal strength
            should_signal = False
            signal_strength = 'none'
            
            if total_score >= 0.75:
                should_signal = True
                signal_strength = 'strong'
            elif total_score >= 0.60:
                should_signal = True
                signal_strength = 'moderate'
            elif total_score >= 0.45:
                should_signal = True
                signal_strength = 'weak'
            
            # Override logic for extreme conditions
            if rsi > 85 and market_structure.get('near_resistance', False):
                should_signal = True
                signal_strength = 'strong'
                total_score = max(total_score, 0.8)
            
            return {
                'should_signal': should_signal,
                'signal_strength': signal_strength,
                'total_score': total_score,
                'component_scores': scores,
                'recommended_action': self.get_recommended_sell_action(total_score, pullback_analysis, scenario_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Sell opportunity analysis error: {e}")
            return {'should_signal': False, 'signal_strength': 'none', 'total_score': 0}

    def get_recommended_buy_action(self, score: float, pullback_analysis: Dict, scenario_analysis: Dict) -> str:
        """Get recommended action for BUY signals"""
        if pullback_analysis['should_wait'] and score < 0.8:
            return 'wait_for_pullback'
        elif score >= 0.75:
            return 'immediate_entry'
        elif score >= 0.60:
            return 'cautious_entry'
        else:
            return 'wait_for_better_setup'

    def get_recommended_sell_action(self, score: float, pullback_analysis: Dict, scenario_analysis: Dict) -> str:
        """Get recommended action for SELL signals"""
        if score >= 0.75:
            return 'immediate_entry'
        elif score >= 0.60:
            return 'cautious_entry'
        else:
            return 'wait_for_better_setup'

    def create_context_aware_buy_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                                      volume_entry: Dict, confluence_zones: List[Dict],
                                      buy_analysis: Dict, pullback_analysis: Dict, 
                                      scenario_analysis: Dict) -> Optional[Dict]:
        """Create BUY signal with context-aware entry logic"""
        try:
            entry_candidates = []
            recommended_action = buy_analysis['recommended_action']
            
            # Determine best entry strategy based on analysis
            if recommended_action == 'wait_for_pullback' and pullback_analysis['pullback_levels']:
                # Prefer pullback levels
                for level in pullback_analysis['pullback_levels']:
                    entry_candidates.append({
                        'price': level['price'], 'score': 0.9 + (level['confidence'] * 0.1),
                        'reason': f"pullback_{level['type']}",
                        'strategy': 'wait_for_pullback'
                    })
                
                # Also add current price as backup
                entry_candidates.append({
                    'price': current_price,
                    'score': 0.4,
                    'reason': 'current_price_backup',
                    'strategy': 'immediate_if_needed'
                })
            else:
                # Immediate entry scenarios
                entry_candidates.append({
                    'price': current_price,
                    'score': 0.8,
                    'reason': 'current_price',
                    'strategy': 'immediate'
                })

                # Score all support zones
                for zone in confluence_zones:
                    if zone['zone_type'] == 'support' and zone['price'] <= current_price:
                        distance = abs(zone['price'] - current_price) / current_price
                        if distance <= 0.02:  # Within 2%
                            score = 0.85 + (zone.get('confluence', 1) * 0.05)
                            entry_candidates.append({
                                'price': zone['price'],
                                'score': score,
                                'reason': f"support_{zone.get('label', '')}",
                                'strategy': 'limit_order'
                            })

                # Volume-based entry
                if volume_entry.get('confidence', 0) > 0.5:
                    vol_price = volume_entry.get('entry_price', current_price)
                    if vol_price <= current_price:
                        score = 0.7 + (volume_entry['confidence'] * 0.1)
                        entry_candidates.append({
                            'price': vol_price,
                            'score': score,
                            'reason': 'volume_profile',
                            'strategy': 'limit_order'
                        })

                # Bollinger Band lower
                bb_lower = latest.get('bb_lower', current_price * 0.98)
                if bb_lower <= current_price:
                    entry_candidates.append({
                        'price': bb_lower,
                        'score': 0.65,
                        'reason': 'bb_lower',
                        'strategy': 'limit_order'
                    })

            # Select best entry
            if not entry_candidates:
                return None

            best_entry = max(entry_candidates, key=lambda x: x['score'])
            optimal_entry = best_entry['price']
            entry_strategy = best_entry['strategy']

            # Determine order type based on strategy and distance
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            if entry_strategy == 'wait_for_pullback':
                order_type = 'limit'
                # Add waiting instruction
                signal_notes = f"WAIT for pullback to ${optimal_entry:.4f} ({best_entry['reason']})"
            elif distance_pct > 0.008 or latest.get('stoch_rsi_k', 50) < 35:
                order_type = 'limit'
                signal_notes = f"Limit order at ${optimal_entry:.4f}"
            else:
                order_type = 'market'
                optimal_entry = current_price
                signal_notes = "Market entry recommended"

            # Risk management
            resistances = [zone['price'] for zone in confluence_zones 
                          if zone['zone_type'] == 'resistance' and zone['price'] > optimal_entry]
            
            if not resistances:
                self.logger.debug(f"No resistance found above entry for {symbol_data['symbol']}, skipping BUY signal.")
                return None

            atr = latest.get('atr', current_price * 0.02)
            
            # Dynamic stop loss based on context
            if entry_strategy == 'wait_for_pullback':
                stop_loss = optimal_entry - (2.5 * atr)  # Tighter stop for pullback entries
            else:
                stop_loss = optimal_entry - (3.0 * atr)  # Standard stop

            # Take profit levels
            take_profit_1 = min(resistances)
            further_resistances = [r for r in resistances if r > take_profit_1]
            take_profit_2 = max(further_resistances) if further_resistances else take_profit_1 * 1.05  # Default to 5% above first TP

            # Risk-reward calculation
            risk_amount = optimal_entry - stop_loss
            reward_amount = take_profit_1 - optimal_entry
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

            # Context-aware confidence calculation
            base_confidence = 50
            base_confidence += buy_analysis['total_score'] * 30  # Up to 30 points from analysis
            
            # Bonus for different strategies
            if entry_strategy == 'wait_for_pullback':
                base_confidence += 8
            elif best_entry['reason'].startswith('support'):
                base_confidence += 6
            elif volume_entry.get('confidence', 0) > 0.6:
                base_confidence += 4

            # Trend context bonus
            if buy_analysis['component_scores'].get('trend', 0) > 0.7:
                base_confidence += 5

            confidence = min(90, base_confidence)

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
                'signal_type': 'context_aware_buy',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'signal_notes': signal_notes,
                'entry_strategy': entry_strategy,
                'recommended_action': buy_analysis['recommended_action'],
                'analysis_details': {
                    'total_score': buy_analysis['total_score'],
                    'signal_strength': buy_analysis['signal_strength'],
                    'component_scores': buy_analysis['component_scores'],
                    'entry_reason': best_entry['reason'],
                    'pullback_opportunity': pullback_analysis['should_wait'],
                    'pullback_levels': pullback_analysis['pullback_levels'][:2] if pullback_analysis['pullback_levels'] else []
                },
                'scenarios': {
                    'current_scenario': entry_strategy,
                    'alternative_scenarios': self.format_alternative_scenarios(scenario_analysis, 'buy')
                }
            }

        except Exception as e:
            self.logger.error(f"Context-aware BUY signal creation error: {e}")
            return None

    def create_context_aware_sell_signal(self, symbol_data: Dict, current_price: float, latest: pd.Series,
                                       volume_entry: Dict, confluence_zones: List[Dict],
                                       sell_analysis: Dict, pullback_analysis: Dict, 
                                       scenario_analysis: Dict) -> Optional[Dict]:
        """Create SELL signal with context-aware entry logic"""
        try:
            entry_candidates = []
            recommended_action = sell_analysis['recommended_action']

            # Always consider current price
            entry_candidates.append({
                'price': current_price,
                'score': 0.7,
                'reason': 'current_price',
                'strategy': 'immediate'
            })

            # Score all resistance zones
            for zone in confluence_zones:
                if zone['zone_type'] == 'resistance' and zone['price'] >= current_price:
                    distance = abs(zone['price'] - current_price) / current_price
                    if distance <= 0.02:  # Within 2%
                        score = 0.85 + (zone.get('confluence', 1) * 0.05)
                        entry_candidates.append({
                            'price': zone['price'],
                            'score': score,
                            'reason': f"resistance_{zone.get('label', '')}",
                            'strategy': 'limit_order'
                        })

            # Volume-based entry
            if volume_entry.get('confidence', 0) > 0.5:
                vol_price = volume_entry.get('entry_price', current_price)
                if vol_price >= current_price:
                    score = 0.7 + (volume_entry['confidence'] * 0.1)
                    entry_candidates.append({
                        'price': vol_price,
                        'score': score,
                        'reason': 'volume_profile',
                        'strategy': 'limit_order'
                    })

            # Bollinger Band upper
            bb_upper = latest.get('bb_upper', current_price * 1.02)
            if bb_upper >= current_price:
                entry_candidates.append({
                    'price': bb_upper,
                    'score': 0.65,
                    'reason': 'bb_upper',
                    'strategy': 'limit_order'
                })

            # Select best entry
            best_entry = max(entry_candidates, key=lambda x: x['score'])
            optimal_entry = best_entry['price']
            entry_strategy = best_entry['strategy']

            # Determine order type
            distance_pct = abs(optimal_entry - current_price) / current_price
            
            if distance_pct > 0.008 or latest.get('stoch_rsi_k', 50) > 75:
                order_type = 'limit'
                signal_notes = f"Limit order at ${optimal_entry:.4f}"
            else:
                order_type = 'market'
                optimal_entry = current_price
                signal_notes = "Market entry recommended"

            # Risk management for SELL
            supports = [zone['price'] for zone in confluence_zones 
                       if zone['zone_type'] == 'support' and zone['price'] < optimal_entry]
            
            if not supports:
                self.logger.debug(f"No support found below entry for {symbol_data['symbol']}, skipping SELL signal.")
                return None

            atr = latest.get('atr', current_price * 0.02)
            stop_loss = optimal_entry + (3.0 * atr)
            take_profit_1 = max(supports)
            further_supports = [s for s in supports if s < take_profit_1]
            take_profit_2 = min(further_supports) if further_supports else take_profit_1 * 0.95

            # Risk-reward calculation
            risk_amount = stop_loss - optimal_entry
            reward_amount = optimal_entry - take_profit_1
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

            # Context-aware confidence calculation
            base_confidence = 48  # Slightly lower base for sells
            base_confidence += sell_analysis['total_score'] * 30
            
            if best_entry['reason'].startswith('resistance'):
                base_confidence += 6
            if volume_entry.get('confidence', 0) > 0.6:
                base_confidence += 4
            if sell_analysis['component_scores'].get('trend', 0) > 0.7:
                base_confidence += 5

            confidence = min(88, base_confidence)

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
                'signal_type': 'context_aware_sell',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'signal_notes': signal_notes,
                'entry_strategy': entry_strategy,
                'recommended_action': sell_analysis['recommended_action'],
                'analysis_details': {
                    'total_score': sell_analysis['total_score'],
                    'signal_strength': sell_analysis['signal_strength'],
                    'component_scores': sell_analysis['component_scores'],
                    'entry_reason': best_entry['reason']
                },
                'scenarios': {
                    'current_scenario': entry_strategy,
                    'alternative_scenarios': self.format_alternative_scenarios(scenario_analysis, 'sell')
                }
            }

        except Exception as e:
            self.logger.error(f"Context-aware SELL signal creation error: {e}")
            return None

    def format_alternative_scenarios(self, scenario_analysis: Dict, side: str) -> List[Dict]:
        """Format alternative scenarios for the signal output"""
        alternatives = []
        
        for scenario_name, scenario in scenario_analysis.items():
            if scenario_name == 'immediate':
                continue  # Skip immediate as it's the main scenario
                
            if side == 'buy':
                if scenario_name == 'pullback':
                    alternatives.append({
                        'name': 'Wait for Pullback',
                        'description': f"Wait for pullback to ${scenario['entry_price']:.4f} ({scenario.get('pullback_type', 'support')})",
                        'confidence': 'High' if scenario['confidence'] == 'high' else 'Medium'
                    })
                elif scenario_name == 'breakout':
                    alternatives.append({
                        'name': 'Breakout Play',
                        'description': f"Enter on breakout above ${scenario.get('breakout_level', 0):.4f}",
                        'confidence': 'Medium'
                    })
            else:  # sell
                if scenario_name == 'breakdown':
                    alternatives.append({
                        'name': 'Breakdown Short',
                        'description': f"Short on breakdown below ${scenario.get('breakdown_level', 0):.4f}",
                        'confidence': 'Low'
                    })
        
        return alternatives

    # Keep all existing methods for backward compatibility
    
    def validate_signal_quality(self, signal: Dict, df: pd.DataFrame, market_structure: Dict) -> Optional[Dict]:
        """Enhanced signal validation with context awareness"""
        try:
            if not signal or not market_structure.get('has_sufficient_data', False):
                return None
            
            # Context-aware minimum thresholds
            analysis_details = signal.get('analysis_details', {})
            signal_strength = analysis_details.get('signal_strength', 'weak')
            
            # Adjust thresholds based on signal strength and market conditions
            if signal_strength == 'strong':
                min_confidence = 50 if signal['side'] == 'buy' else 52
                min_rr = 1.8 if signal['side'] == 'buy' else 2.0
            elif signal_strength == 'moderate':
                min_confidence = 55 if signal['side'] == 'buy' else 58
                min_rr = 2.0 if signal['side'] == 'buy' else 2.2
            else:  # weak
                min_confidence = 60 if signal['side'] == 'buy' else 62
                min_rr = 2.2 if signal['side'] == 'buy' else 2.4
            
            if signal['confidence'] < min_confidence:
                self.logger.debug(f"Signal filtered: {signal['symbol']} confidence too low ({signal['confidence']} < {min_confidence})")
                return None
            
            if signal['risk_reward_ratio'] < min_rr:
                self.logger.debug(f"Signal filtered: {signal['symbol']} R/R too low ({signal['risk_reward_ratio']:.2f} < {min_rr})")
                return None
            
            # Additional validation for wait-for-pullback signals
            if signal.get('entry_strategy') == 'wait_for_pullback':
                pullback_levels = analysis_details.get('pullback_levels', [])
                if not pullback_levels:
                    self.logger.debug(f"Pullback signal filtered: {signal['symbol']} no valid pullback levels")
                    return None
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
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
        """Enhanced market structure analysis with more context"""
        try:
            if len(df) < 50:
                return self.get_insufficient_data_structure()
            
            # Calculate with sufficient lookback
            recent_high = df['high'].tail(30).max()
            recent_low = df['low'].tail(30).min()
            price_range = recent_high - recent_low
            
            if price_range == 0:
                return self.get_insufficient_data_structure()
            
            # More nuanced proximity requirements
            range_position = (current_price - recent_low) / price_range
            near_support = range_position < 0.35  # More flexible
            near_resistance = range_position > 0.65  # More flexible
            
            # Enhanced trend analysis
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            
            # Multi-timeframe trend check
            short_trend = (current_price > sma_20) and (sma_20 > sma_50 * 1.01)
            medium_trend = sma_20 > sma_50 * 1.02
            
            strong_uptrend = short_trend and medium_trend and (current_price > sma_20 * 1.01)
            strong_downtrend = (current_price < sma_20) and (sma_20 < sma_50 * 0.99) and (current_price < sma_20 * 0.99)
            
            # Volume confirmation
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['volume'].tail(5).mean()
            
            # Enhanced bounce/distribution detection
            bounce_potential = (near_support and not strong_downtrend and 
                              recent_volume > volume_ma and range_position < 0.4)
            distribution_signs = (near_resistance and recent_volume > volume_ma * 1.3 and
                                range_position > 0.6)
            
            return {
                'near_support': near_support,
                'near_resistance': near_resistance,
                'strong_uptrend': strong_uptrend,
                'strong_downtrend': strong_downtrend,
                'bounce_potential': bounce_potential,
                'distribution_signs': distribution_signs,
                'price_range_position': range_position,
                'trend_alignment': {
                    'short_term_bullish': short_trend,
                    'medium_term_bullish': medium_trend,
                    'volume_supporting': recent_volume > volume_ma
                },
                'has_sufficient_data': True
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return self.get_insufficient_data_structure()
    
    def get_insufficient_data_structure(self) -> Dict:
        """Return structure indicating insufficient data"""
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

    # Keep all existing methods unchanged for backward compatibility
    def rank_opportunities_with_mtf(self, signals: List[Dict], dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """MTF-Priority Ranking System - Fixed to prioritize confirmed signals"""
        try:
            opportunities = []
            
            for signal in signals:
                # === MTF CONFIRMATION ANALYSIS (HIGHEST PRIORITY) ===
                mtf_analysis = signal.get('mtf_analysis', {})
                mtf_status = signal.get('mtf_status', 'NONE')
                
                # MTF Score and Multiplier (MOST IMPORTANT)
                if mtf_status == 'STRONG':
                    mtf_score = 1.0
                    mtf_priority_tier = 3000  # Top tier
                    mtf_multiplier = 1.5      # 50% boost
                elif mtf_status == 'PARTIAL':
                    confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
                    mtf_score = len(confirmed_timeframes) / len(self.config.confirmation_timeframes)  # Assuming 3 timeframes
                    mtf_priority_tier = 2000  # Second tier
                    mtf_multiplier = 1.0 + (mtf_score * 0.3)  # Up to 30% boost
                else:  # NONE
                    mtf_score = 0.0
                    mtf_priority_tier = 1000  # Third tier
                    mtf_multiplier = 0.7      # 30% PENALTY for no confirmation
                
                # === CORE SIGNAL METRICS ===
                confidence = signal['confidence']
                confidence_score = confidence / 100
                rr_ratio = signal.get('risk_reward_ratio', 1)
                rr_score = min(1.0, rr_ratio / 3.0)
                
                # === SIGNAL STRENGTH ===
                analysis_details = signal.get('analysis_details', {})
                signal_strength = analysis_details.get('signal_strength', 'weak')
                
                strength_multipliers = {
                    'strong': 1.3,
                    'moderate': 1.1,
                    'weak': 0.9
                }
                strength_multiplier = strength_multipliers[signal_strength]
                
                # === MARKET QUALITY FACTORS ===
                volume_24h = signal.get('volume_24h', 0)
                
                # Volume scoring (liquidity is crucial)
                if volume_24h >= 10_000_000:
                    volume_score = 1.0
                    volume_bonus = 200  # Priority bonus
                elif volume_24h >= 5_000_000:
                    volume_score = 0.8
                    volume_bonus = 100
                elif volume_24h >= 1_000_000:
                    volume_score = 0.6
                    volume_bonus = 50
                else:
                    volume_score = 0.3
                    volume_bonus = -100  # Priority penalty
                
                # === ENTRY STRATEGY QUALITY ===
                entry_strategy = signal.get('entry_strategy', 'immediate')
                strategy_scores = {
                    'wait_for_pullback': 0.9,  # Best entries
                    'limit_order': 0.8,
                    'immediate': 0.7
                }
                strategy_score = strategy_scores.get(entry_strategy, 0.7)
                
                # === TECHNICAL COMPONENT SCORING ===
                component_scores = analysis_details.get('component_scores', {})
                technical_score = component_scores.get('technical', 0.5)
                structure_score = component_scores.get('structure', 0.5)
                trend_score = component_scores.get('trend', 0.5)
                
                # === RISK PENALTIES ===
                risk_penalty = 1.0
                
                # Heavy penalty for weak signals without MTF confirmation
                if mtf_status == 'NONE' and signal_strength == 'weak':
                    risk_penalty = 0.5  # 50% penalty
                elif mtf_status == 'NONE' and confidence < 60:
                    risk_penalty = 0.6  # 40% penalty
                elif mtf_status == 'PARTIAL' and confidence < 50:
                    risk_penalty = 0.8  # 20% penalty
                
                # === CALCULATE BASE SCORE ===
                base_score = (
                    confidence_score * 0.25 +
                    rr_score * 0.20 +
                    technical_score * 0.18 +
                    structure_score * 0.15 +
                    trend_score * 0.12 +
                    strategy_score * 0.10
                )
                
                # === APPLY ALL MULTIPLIERS ===
                final_score = (
                    base_score *
                    mtf_multiplier *      # MOST IMPORTANT
                    strength_multiplier *
                    risk_penalty
                ) + (volume_score * 0.05)  # Small volume bonus to final score
                
                # Cap the score
                final_score = max(0, min(1.0, final_score))
                
                # === PRIORITY CALCULATION (MTF-FIRST) ===
                # Start with MTF tier, then add other factors
                base_priority = mtf_priority_tier
                
                # Add confidence-based priority within tier
                confidence_priority = int(confidence * 8)  # 0-800 range
                
                # Add R/R priority
                rr_priority = int(min(rr_ratio, 4.0) * 50)  # 0-200 range
                
                # Add volume bonus/penalty
                priority_adjustment = volume_bonus + confidence_priority + rr_priority
                
                final_priority = base_priority + priority_adjustment
                
                # === SPECIAL CASES ===
                # Boost for exceptional non-MTF signals (rare cases)
                if (mtf_status == 'NONE' and confidence >= 85 and 
                    rr_ratio >= 3.0 and signal_strength == 'strong'):
                    final_priority += 500  # Still lower than PARTIAL, but competitive
                    final_score *= 1.1
                
                # === QUALITY GATES ===
                quality_flags = []
                
                # Critical quality issues
                if mtf_status == 'NONE' and confidence < 55:
                    quality_flags.append('low_confidence_no_mtf')
                if volume_24h < 500_000:
                    quality_flags.append('low_liquidity')
                if rr_ratio < 1.5:
                    quality_flags.append('poor_risk_reward')
                
                # === STORE DETAILED RANKING INFO ===
                ranking_details = {
                    'mtf_status': mtf_status,
                    'mtf_score': mtf_score,
                    'mtf_priority_tier': mtf_priority_tier,
                    'mtf_multiplier': mtf_multiplier,
                    'base_score': base_score,
                    'final_score': final_score,
                    'strength_multiplier': strength_multiplier,
                    'risk_penalty': risk_penalty,
                    'volume_score': volume_score,
                    'quality_flags': quality_flags,
                    'priority_breakdown': {
                        'mtf_tier': mtf_priority_tier,
                        'confidence_bonus': confidence_priority,
                        'rr_bonus': rr_priority,
                        'volume_bonus': volume_bonus
                    }
                }
                
                opportunities.append({
                    **signal,
                    'score': final_score,
                    'priority': final_priority,
                    'ranking_details': ranking_details,
                    'quality_grade': self._calculate_quality_grade(signal, ranking_details)
                })

            # === SORT BY MTF-PRIORITY RANKING ===
            opportunities.sort(key=lambda x: (x['priority'], x['score']), reverse=True)
            
            # === APPLY QUALITY FILTERS ===
            filtered_opportunities = []
            for opp in opportunities:
                quality_flags = opp['ranking_details']['quality_flags']
                
                # Allow high-quality signals regardless of MTF
                if (opp['confidence'] >= 80 and opp['risk_reward_ratio'] >= 2.5 and 
                    opp['volume_24h'] >= 1_000_000):
                    filtered_opportunities.append(opp)
                # Require MTF for medium-quality signals
                elif opp.get('mtf_status') in ['STRONG', 'PARTIAL']:
                    filtered_opportunities.append(opp)
                # Very strict criteria for no-MTF signals
                elif (opp.get('mtf_status') == 'NONE' and opp['confidence'] >= 75 and 
                    opp['risk_reward_ratio'] >= 2.8 and len(quality_flags) == 0):
                    filtered_opportunities.append(opp)
                else:
                    self.logger.debug(f"Filtered {opp['symbol']}: MTF={opp.get('mtf_status')}, "
                                    f"Conf={opp['confidence']}, Flags={quality_flags}")
            
            return filtered_opportunities[:self.config.charts_per_batch]

        except Exception as e:
            self.logger.error(f"MTF-priority ranking error: {e}")
            return signals

    # def rank_opportunities_with_mtf(self, signals: List[Dict], dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
    #     """
    #     MTF-Priority Ranking System - Enhanced to prioritize higher timeframe confirmation
        
    #     Key Principles:
    #     1. Higher timeframes (2h, 4h, 6h) carry exponentially more weight
    #     2. Signals without higher timeframe confirmation are heavily penalized
    #     3. Lower timeframe-only signals are considered high-risk/low-probability
    #     4. Quality over quantity - better to have fewer high-conviction trades
    #     """
    #     try:
    #         opportunities = []
            
    #         # Define timeframe hierarchy with exponential importance weights
    #         timeframe_weights = {
    #             '15m': 1.0,    # Base weight
    #             '30m': 1.5,    # Primary timeframe gets moderate boost
    #             '2h': 4.0,     # Higher timeframes get exponential weight
    #             '4h': 6.0,
    #             '6h': 8.0,
    #             '1d': 10.0     # Daily gets maximum weight
    #         }
            
    #         # Define minimum higher timeframe requirements
    #         higher_timeframes = ['2h', '4h', '6h', '1d']
    #         lower_timeframes = ['15m', '30m']
            
    #         for signal in signals:
    #             # === MTF ANALYSIS WITH TIMEFRAME HIERARCHY ===
    #             mtf_analysis = signal.get('mtf_analysis', {})
    #             mtf_status = signal.get('mtf_status', 'NONE')
    #             confirmed_timeframes = mtf_analysis.get('confirmed_timeframes', [])
    #             neutral_timeframes = mtf_analysis.get('neutral_timeframes', [])
    #             opposing_timeframes = mtf_analysis.get('opposing_timeframes', [])
                
    #             # Calculate weighted MTF score based on timeframe importance
    #             total_weight = sum(timeframe_weights.get(tf, 1.0) for tf in timeframe_weights.keys())
    #             confirmed_weight = sum(timeframe_weights.get(tf, 1.0) for tf in confirmed_timeframes)
    #             opposing_weight = sum(timeframe_weights.get(tf, 1.0) for tf in opposing_timeframes)
                
    #             # MTF Quality Score (0-1)
    #             mtf_quality_score = confirmed_weight / total_weight if total_weight > 0 else 0
                
    #             # === HIGHER TIMEFRAME CONVICTION ANALYSIS ===
    #             higher_tf_confirmed = [tf for tf in confirmed_timeframes if tf in higher_timeframes]
    #             higher_tf_opposing = [tf for tf in opposing_timeframes if tf in higher_timeframes]
    #             lower_tf_confirmed = [tf for tf in confirmed_timeframes if tf in lower_timeframes]
                
    #             # Calculate higher timeframe conviction
    #             higher_tf_conviction = len(higher_tf_confirmed) / len(higher_timeframes)
    #             lower_tf_only = len(confirmed_timeframes) > 0 and len(higher_tf_confirmed) == 0
                
    #             # === MTF CLASSIFICATION AND TIER ASSIGNMENT ===
    #             if len(higher_tf_confirmed) >= 2:  # At least 2 higher TFs confirmed
    #                 mtf_tier = 'PREMIUM'
    #                 base_priority = 5000
    #                 mtf_multiplier = 2.0
    #                 conviction_level = 'VERY_HIGH'
    #             elif len(higher_tf_confirmed) >= 1:  # At least 1 higher TF confirmed
    #                 mtf_tier = 'HIGH_QUALITY'
    #                 base_priority = 3500
    #                 mtf_multiplier = 1.6
    #                 conviction_level = 'HIGH'
    #             elif len(higher_tf_confirmed) == 0 and len(higher_tf_opposing) == 0:  # Higher TFs neutral
    #                 if len(lower_tf_confirmed) >= 2:
    #                     mtf_tier = 'SPECULATIVE'
    #                     base_priority = 1500
    #                     mtf_multiplier = 0.8
    #                     conviction_level = 'MEDIUM'
    #                 else:
    #                     mtf_tier = 'LOW_CONVICTION'
    #                     base_priority = 800
    #                     mtf_multiplier = 0.5
    #                     conviction_level = 'LOW'
    #             else:  # Higher TFs opposing
    #                 mtf_tier = 'COUNTER_TREND'
    #                 base_priority = 200
    #                 mtf_multiplier = 0.3
    #                 conviction_level = 'VERY_LOW'
                
    #             # === SPECIAL PENALTIES FOR PROBLEMATIC PATTERNS ===
    #             quality_penalties = []
                
    #             # Heavy penalty for lower timeframe only signals
    #             if lower_tf_only:
    #                 mtf_multiplier *= 0.4  # 60% penalty
    #                 quality_penalties.append('lower_tf_only')
                
    #             # Penalty for opposing higher timeframes
    #             if len(higher_tf_opposing) > 0:
    #                 penalty = 0.7 ** len(higher_tf_opposing)  # Exponential penalty
    #                 mtf_multiplier *= penalty
    #                 quality_penalties.append(f'opposing_higher_tf_{len(higher_tf_opposing)}')
                
    #             # Bonus for higher timeframe alignment
    #             if len(higher_tf_confirmed) >= 2:
    #                 mtf_multiplier *= 1.3  # 30% bonus for strong alignment
                
    #             # === CORE SIGNAL METRICS ===
    #             confidence = signal['confidence']
    #             confidence_score = confidence / 100
    #             rr_ratio = signal.get('risk_reward_ratio', 1)
    #             rr_score = min(1.0, rr_ratio / 3.0)
                
    #             # === SIGNAL STRENGTH WITH MTF CONTEXT ===
    #             analysis_details = signal.get('analysis_details', {})
    #             signal_strength = analysis_details.get('signal_strength', 'weak')
                
    #             # Adjust signal strength based on MTF context
    #             if mtf_tier in ['PREMIUM', 'HIGH_QUALITY']:
    #                 if signal_strength == 'weak':
    #                     signal_strength = 'moderate'  # Upgrade weak signals with good MTF
    #                 elif signal_strength == 'moderate':
    #                     signal_strength = 'strong'  # Upgrade moderate signals
    #             elif mtf_tier in ['COUNTER_TREND', 'LOW_CONVICTION']:
    #                 if signal_strength == 'strong':
    #                     signal_strength = 'moderate'  # Downgrade strong signals with poor MTF
    #                 elif signal_strength == 'moderate':
    #                     signal_strength = 'weak'  # Downgrade moderate signals
                
    #             strength_multipliers = {
    #                 'strong': 1.4,
    #                 'moderate': 1.1,
    #                 'weak': 0.8
    #             }
    #             strength_multiplier = strength_multipliers[signal_strength]
                
    #             # === MARKET QUALITY FACTORS ===
    #             volume_24h = signal.get('volume_24h', 0)
                
    #             # Volume scoring with higher standards for lower MTF tiers
    #             if mtf_tier in ['PREMIUM', 'HIGH_QUALITY']:
    #                 volume_thresholds = [5_000_000, 2_000_000, 1_000_000]
    #             else:
    #                 volume_thresholds = [15_000_000, 8_000_000, 3_000_000]  # Higher requirements
                
    #             if volume_24h >= volume_thresholds[0]:
    #                 volume_score = 1.0
    #                 volume_priority_bonus = 300
    #             elif volume_24h >= volume_thresholds[1]:
    #                 volume_score = 0.8
    #                 volume_priority_bonus = 150
    #             elif volume_24h >= volume_thresholds[2]:
    #                 volume_score = 0.6
    #                 volume_priority_bonus = 50
    #             else:
    #                 volume_score = 0.3
    #                 volume_priority_bonus = -200
    #                 if mtf_tier in ['SPECULATIVE', 'LOW_CONVICTION', 'COUNTER_TREND']:
    #                     volume_priority_bonus = -500  # Heavy penalty for low MTF + low volume
                
    #             # === RISK ASSESSMENT WITH MTF CONTEXT ===
    #             risk_penalty = 1.0
                
    #             # Risk penalties based on MTF tier
    #             if mtf_tier == 'COUNTER_TREND':
    #                 risk_penalty = 0.2  # 80% penalty - very dangerous
    #             elif mtf_tier == 'LOW_CONVICTION':
    #                 risk_penalty = 0.4  # 60% penalty
    #             elif mtf_tier == 'SPECULATIVE':
    #                 risk_penalty = 0.7  # 30% penalty
    #             elif lower_tf_only and confidence < 65:
    #                 risk_penalty = 0.5  # 50% penalty for weak lower TF only signals
                
    #             # Additional risk factors
    #             if len(quality_penalties) >= 2:
    #                 risk_penalty *= 0.6  # Multiple quality issues
                
    #             # === ENTRY STRATEGY EVALUATION ===
    #             entry_strategy = signal.get('entry_strategy', 'immediate')
    #             strategy_scores = {
    #                 'wait_for_pullback': 0.9,
    #                 'limit_order': 0.8,
    #                 'immediate': 0.7
    #             }
    #             strategy_score = strategy_scores.get(entry_strategy, 0.7)
                
    #             # Strategy adjustments based on MTF
    #             if mtf_tier in ['PREMIUM', 'HIGH_QUALITY'] and entry_strategy == 'immediate':
    #                 strategy_score += 0.1  # Immediate entry OK for high conviction
    #             elif mtf_tier in ['SPECULATIVE', 'LOW_CONVICTION'] and entry_strategy != 'wait_for_pullback':
    #                 strategy_score -= 0.15  # Require patience for low conviction
                
    #             # === TECHNICAL COMPONENT SCORING ===
    #             component_scores = analysis_details.get('component_scores', {})
    #             technical_score = component_scores.get('technical', 0.5)
    #             structure_score = component_scores.get('structure', 0.5)
    #             trend_score = component_scores.get('trend', 0.5)
                
    #             # === CALCULATE BASE SCORE ===
    #             base_score = (
    #                 confidence_score * 0.20 +    # Reduced weight, MTF is more important
    #                 rr_score * 0.15 +
    #                 technical_score * 0.18 +
    #                 structure_score * 0.15 +
    #                 trend_score * 0.12 +
    #                 strategy_score * 0.10 +
    #                 mtf_quality_score * 0.10     # Direct MTF contribution
    #             )
                
    #             # === APPLY ALL MULTIPLIERS ===
    #             final_score = (
    #                 base_score *
    #                 mtf_multiplier *      # MOST IMPORTANT FACTOR
    #                 strength_multiplier *
    #                 risk_penalty
    #             ) + (volume_score * 0.03)  # Small volume bonus
                
    #             # Cap the score
    #             final_score = max(0, min(1.0, final_score))
                
    #             # === PRIORITY CALCULATION WITH MTF HIERARCHY ===
    #             # Start with MTF tier priority
    #             priority_adjustments = 0
                
    #             # Confidence adjustment (smaller range for MTF-focused ranking)
    #             priority_adjustments += int(confidence * 4)  # 0-400 range
                
    #             # R/R adjustment
    #             priority_adjustments += int(min(rr_ratio, 4.0) * 30)  # 0-120 range
                
    #             # Volume adjustment
    #             priority_adjustments += volume_priority_bonus
                
    #             # Higher timeframe conviction bonus
    #             if len(higher_tf_confirmed) >= 2:
    #                 priority_adjustments += 500
    #             elif len(higher_tf_confirmed) == 1:
    #                 priority_adjustments += 200
                
    #             # Final priority
    #             final_priority = base_priority + priority_adjustments
                
    #             # === QUALITY GATES ===
    #             quality_flags = list(quality_penalties)
                
    #             # Add quality flags
    #             if confidence < 50:
    #                 quality_flags.append('low_confidence')
    #             if rr_ratio < 1.8:
    #                 quality_flags.append('poor_risk_reward')
    #             if volume_24h < 500_000:
    #                 quality_flags.append('very_low_liquidity')
    #             if mtf_tier in ['COUNTER_TREND', 'LOW_CONVICTION']:
    #                 quality_flags.append('poor_mtf_structure')
                
    #             # === STORE ENHANCED RANKING INFO ===
    #             ranking_details = {
    #                 'mtf_tier': mtf_tier,
    #                 'conviction_level': conviction_level,
    #                 'mtf_quality_score': mtf_quality_score,
    #                 'higher_tf_confirmed': higher_tf_confirmed,
    #                 'higher_tf_opposing': higher_tf_opposing,
    #                 'lower_tf_only': lower_tf_only,
    #                 'higher_tf_conviction': higher_tf_conviction,
    #                 'base_score': base_score,
    #                 'final_score': final_score,
    #                 'mtf_multiplier': mtf_multiplier,
    #                 'strength_multiplier': strength_multiplier,
    #                 'risk_penalty': risk_penalty,
    #                 'volume_score': volume_score,
    #                 'quality_flags': quality_flags,
    #                 'priority_breakdown': {
    #                     'mtf_base': base_priority,
    #                     'confidence_bonus': int(confidence * 4),
    #                     'rr_bonus': int(min(rr_ratio, 4.0) * 30),
    #                     'volume_bonus': volume_priority_bonus,
    #                     'higher_tf_bonus': 500 if len(higher_tf_confirmed) >= 2 else (200 if len(higher_tf_confirmed) == 1 else 0)
    #                 }
    #             }
                
    #             opportunities.append({
    #                 **signal,
    #                 'score': final_score,
    #                 'priority': final_priority,
    #                 'ranking_details': ranking_details,
    #                 'quality_grade': self._calculate_enhanced_quality_grade(signal, ranking_details)
    #             })

    #         # === SORT BY MTF-PRIORITY HIERARCHY ===
    #         opportunities.sort(key=lambda x: (x['priority'], x['score']), reverse=True)
            
    #         # === APPLY STRICT QUALITY FILTERS ===
    #         filtered_opportunities = []
    #         premium_count = 0
    #         high_quality_count = 0
            
    #         for opp in opportunities:
    #             mtf_tier = opp['ranking_details']['mtf_tier']
    #             quality_flags = opp['ranking_details']['quality_flags']
    #             conviction_level = opp['ranking_details']['conviction_level']
                
    #             # Always accept PREMIUM signals (unless critical flaws)
    #             if mtf_tier == 'PREMIUM':
    #                 if len(quality_flags) <= 1:  # Allow minor flaws
    #                     filtered_opportunities.append(opp)
    #                     premium_count += 1
    #                     continue
                
    #             # Accept HIGH_QUALITY signals with good metrics
    #             elif mtf_tier == 'HIGH_QUALITY':
    #                 if (opp['confidence'] >= 55 and opp['risk_reward_ratio'] >= 2.0 and 
    #                     len(quality_flags) <= 2):
    #                     filtered_opportunities.append(opp)
    #                     high_quality_count += 1
    #                     continue
                
    #             # Very strict criteria for SPECULATIVE signals
    #             elif mtf_tier == 'SPECULATIVE':
    #                 if (opp['confidence'] >= 70 and opp['risk_reward_ratio'] >= 2.5 and 
    #                     opp['volume_24h'] >= 2_000_000 and len(quality_flags) == 0):
    #                     filtered_opportunities.append(opp)
    #                     continue
                
    #             # Almost never accept LOW_CONVICTION or COUNTER_TREND
    #             elif mtf_tier == 'LOW_CONVICTION':
    #                 if (opp['confidence'] >= 80 and opp['risk_reward_ratio'] >= 3.0 and 
    #                     opp['volume_24h'] >= 5_000_000 and len(quality_flags) == 0):
    #                     filtered_opportunities.append(opp)
    #                     continue
                
    #             # Log filtered signals for debugging
    #             self.logger.debug(f"Filtered {opp['symbol']}: MTF_Tier={mtf_tier}, "
    #                             f"Conviction={conviction_level}, Conf={opp['confidence']}, "
    #                             f"Flags={quality_flags}")
            
    #         # === FINAL PORTFOLIO BALANCING ===
    #         # Ensure we don't have too many speculative trades
    #         final_list = []
    #         speculative_count = 0
    #         max_speculative = 2  # Maximum speculative trades allowed
            
    #         for opp in filtered_opportunities:
    #             mtf_tier = opp['ranking_details']['mtf_tier']
                
    #             if mtf_tier in ['PREMIUM', 'HIGH_QUALITY']:
    #                 final_list.append(opp)
    #             elif mtf_tier == 'SPECULATIVE' and speculative_count < max_speculative:
    #                 final_list.append(opp)
    #                 speculative_count += 1
    #             elif mtf_tier in ['LOW_CONVICTION', 'COUNTER_TREND']:
    #                 # Only allow if we have very few other opportunities
    #                 if len(final_list) < 2:
    #                     final_list.append(opp)
            
    #         # Log ranking summary
    #         self.logger.info(f"MTF Ranking Summary: {len(final_list)} signals selected - "
    #                         f"Premium: {premium_count}, High Quality: {high_quality_count}, "
    #                         f"Speculative: {speculative_count}")
            
    #         return final_list[:self.config.charts_per_batch]

    #     except Exception as e:
    #         self.logger.error(f"Enhanced MTF ranking error: {e}")
    #         return signals

    def _calculate_enhanced_quality_grade(self, signal: Dict, ranking_details: Dict) -> str:
        """Calculate enhanced quality grade with MTF focus"""
        mtf_tier = ranking_details['mtf_tier']
        conviction_level = ranking_details['conviction_level']
        confidence = signal['confidence']
        rr_ratio = signal.get('risk_reward_ratio', 0)
        volume_24h = signal.get('volume_24h', 0)
        quality_flags = ranking_details['quality_flags']
        higher_tf_confirmed = len(ranking_details['higher_tf_confirmed'])
        
        # Start with MTF tier base grade
        tier_grades = {
            'PREMIUM': 95,
            'HIGH_QUALITY': 85,
            'SPECULATIVE': 65,
            'LOW_CONVICTION': 45,
            'COUNTER_TREND': 25
        }
        base_grade = tier_grades.get(mtf_tier, 50)
        
        # Higher timeframe confirmation bonus
        base_grade += higher_tf_confirmed * 5
        
        # Traditional metrics adjustments (smaller impact)
        if confidence >= 75:
            base_grade += 5
        elif confidence <= 50:
            base_grade -= 8
        
        if rr_ratio >= 2.8:
            base_grade += 4
        elif rr_ratio <= 1.8:
            base_grade -= 6
        
        if volume_24h >= 5_000_000:
            base_grade += 3
        elif volume_24h <= 500_000:
            base_grade -= 8
        
        # Quality flags penalty
        base_grade -= len(quality_flags) * 6
        
        # Ensure reasonable bounds
        base_grade = max(15, min(100, base_grade))
        
        # Convert to letter grade
        if base_grade >= 95:
            return 'A+'
        elif base_grade >= 90:
            return 'A'
        elif base_grade >= 85:
            return 'A-'
        elif base_grade >= 80:
            return 'B+'
        elif base_grade >= 75:
            return 'B'
        elif base_grade >= 70:
            return 'B-'
        elif base_grade >= 65:
            return 'C+'
        elif base_grade >= 60:
            return 'C'
        elif base_grade >= 55:
            return 'C-'
        elif base_grade >= 50:
            return 'D+'
        elif base_grade >= 45:
            return 'D'
        else:
            return 'F'

    def _calculate_quality_grade(self, signal: Dict, ranking_details: Dict) -> str:
        """Calculate overall quality grade A-F"""
        mtf_status = ranking_details['mtf_status']
        confidence = signal['confidence']
        rr_ratio = signal.get('risk_reward_ratio', 0)
        volume_24h = signal.get('volume_24h', 0)
        quality_flags = ranking_details['quality_flags']
        
        # Start with base grade from MTF
        if mtf_status == 'STRONG':
            base_grade = 85
        elif mtf_status == 'PARTIAL':
            base_grade = 70
        else:
            base_grade = 50
        
        # Adjust for other factors
        if confidence >= 80:
            base_grade += 10
        elif confidence <= 50:
            base_grade -= 15
        
        if rr_ratio >= 3.0:
            base_grade += 8
        elif rr_ratio <= 1.8:
            base_grade -= 10
        
        if volume_24h >= 5_000_000:
            base_grade += 5
        elif volume_24h <= 500_000:
            base_grade -= 10
        
        # Penalty for quality flags
        base_grade -= len(quality_flags) * 8
        
        # Convert to letter grade
        if base_grade >= 90:
            return 'A+'
        elif base_grade >= 85:
            return 'A'
        elif base_grade >= 80:
            return 'A-'
        elif base_grade >= 75:
            return 'B+'
        elif base_grade >= 70:
            return 'B'
        elif base_grade >= 65:
            return 'B-'
        elif base_grade >= 60:
            return 'C+'
        elif base_grade >= 55:
            return 'C'
        else:
            return 'D'

    def explain_ranking_logic(self, opportunities: List[Dict]) -> str:
        """Explain the ranking logic for debugging"""
        if not opportunities:
            return "No opportunities to rank"
        
        explanation = "=== MTF-PRIORITY RANKING LOGIC ===\n\n"
        
        # Group by MTF status
        strong_mtf = [o for o in opportunities if o.get('mtf_status') == 'STRONG']
        partial_mtf = [o for o in opportunities if o.get('mtf_status') == 'PARTIAL']
        no_mtf = [o for o in opportunities if o.get('mtf_status') == 'NONE']
        
        explanation += f"STRONG MTF (Tier 3000+): {len(strong_mtf)} signals\n"
        explanation += f"PARTIAL MTF (Tier 2000+): {len(partial_mtf)} signals\n"
        explanation += f"NO MTF (Tier 1000+): {len(no_mtf)} signals\n\n"
        
        explanation += "Top 3 Rankings:\n"
        for i, opp in enumerate(opportunities[:3], 1):
            details = opp.get('ranking_details', {})
            explanation += f"{i}. {opp['symbol']}: "
            explanation += f"MTF={opp.get('mtf_status', 'NONE')}, "
            explanation += f"Priority={opp['priority']}, "
            explanation += f"Grade={opp.get('quality_grade', 'N/A')}, "
            explanation += f"Score={opp['score']:.3f}\n"
        
        return explanation

    def _identify_winning_factors(self, signal: Dict, final_score: float, scoring_breakdown: Dict) -> List[str]:
        """Identify key factors that make this a winning signal"""
        factors = []
        
        # High confidence
        if signal['confidence'] >= 70:
            factors.append("High Confidence")
        
        # Excellent R/R
        if signal.get('risk_reward_ratio', 0) >= 2.5:
            factors.append("Excellent Risk/Reward")
        
        # Strong technical setup
        if scoring_breakdown['component_scores']['technical'] > 0.7:
            factors.append("Strong Technical Setup")
        
        # Good market structure
        if scoring_breakdown['component_scores']['structure'] > 0.6:
            factors.append("Favorable Market Structure")
        
        # Trend alignment
        if scoring_breakdown['component_scores']['trend'] > 0.65:
            factors.append("Trend Aligned")
        
        # High volume
        if signal.get('volume_24h', 0) > 5_000_000:
            factors.append("High Liquidity")
        
        # Pullback opportunity
        if signal.get('entry_strategy') == 'wait_for_pullback':
            factors.append("Optimal Entry Level")
        
        # MTF confirmation
        if scoring_breakdown['mtf_multiplier'] > 1.15:
            factors.append("Multi-Timeframe Confirmed")
        
        # Strong signal
        analysis_details = signal.get('analysis_details', {})
        if analysis_details.get('signal_strength') == 'strong':
            factors.append("Strong Signal Strength")
        
        return factors

    def _apply_winning_filters(self, opportunities: List[Dict]) -> List[Dict]:
        """Apply final filters to ensure only winning setups"""
        filtered_opportunities = []
        
        for opp in opportunities:
            # Must meet minimum winning criteria
            winning_criteria = 0
            
            if opp['confidence'] >= 55:
                winning_criteria += 1
            if opp.get('risk_reward_ratio', 0) >= 1.8:
                winning_criteria += 1
            if opp['score'] >= 0.5:
                winning_criteria += 1
            if opp.get('volume_24h', 0) >= 500_000:
                winning_criteria += 1
            
            # At least 3 out of 4 criteria must be met
            if winning_criteria >= 3:
                filtered_opportunities.append(opp)
            else:
                self.logger.debug(f"Filtered out {opp['symbol']}: only {winning_criteria}/4 winning criteria met")
        
        return filtered_opportunities

    def get_ranking_explanation(self, signal: Dict) -> str:
        """Explain why this signal was ranked highly"""
        if 'scoring_breakdown' not in signal:
            return "Standard ranking applied"
        
        breakdown = signal['scoring_breakdown']
        factors = signal.get('winning_factors', [])
        
        explanation = f"Score: {breakdown['final_score']:.3f} | Priority: {signal['priority']}\n"
        explanation += f"Key Strengths: {', '.join(factors[:3])}\n"
        
        # Show top contributing factors
        components = breakdown['component_scores']
        top_component = max(components.items(), key=lambda x: x[1])
        explanation += f"Strongest Factor: {top_component[0].title()} ({top_component[1]:.2f})"
        
        return explanation

    # Keep all other existing methods unchanged for compatibility
    def assess_risk(self, signal: Dict, df: pd.DataFrame, market_data: Dict) -> Dict:
        """Enhanced risk assessment with context awareness"""
        try:
            latest = df.iloc[-1]
            current_price = signal['current_price']
            
            # Get analysis details for context
            analysis_details = signal.get('analysis_details', {})
            signal_strength = analysis_details.get('signal_strength', 'weak')
            
            # Base risk factors
            atr = latest.get('atr', current_price * 0.02)
            volatility = atr / current_price
            
            distance = abs(signal['entry_price'] - current_price) / current_price
            distance_risk = min(1.0, distance * 5)  # More lenient for context-aware signals
            
            # Context-adjusted risk assessment
            rsi = latest.get('rsi', 50)
            bb_position = latest.get('bb_position', 0.5)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Risk factors with context adjustment
            extreme_rsi = rsi < 15 or rsi > 85
            extreme_bb = bb_position < 0.05 or bb_position > 0.95
            low_volume = volume_ratio < 0.7
            
            condition_risk = 0
            if extreme_rsi:
                condition_risk += 0.08 if signal_strength == 'strong' else 0.12
            if extreme_bb:
                condition_risk += 0.06 if signal_strength == 'strong' else 0.10
            if low_volume:
                condition_risk += 0.05 if signal_strength == 'strong' else 0.08
            
            # Entry strategy risk adjustment
            entry_strategy = signal.get('entry_strategy', 'immediate')
            strategy_risk = {
                'wait_for_pullback': -0.08,  # Lower risk
                'immediate': 0.0,
                'limit_order': -0.03
            }.get(entry_strategy, 0.0)
            
            # Side-specific risk
            side_risk = 0.02 if signal['side'] == 'sell' else 0.01
            
            # MTF risk reduction
            mtf_analysis = signal.get('mtf_analysis', {})
            confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
            total_timeframes = len(getattr(self.config, 'confirmation_timeframes', [])) or 3
            mtf_risk_reduction = (confirmed_count / total_timeframes) * 0.20 if total_timeframes > 0 else 0
            
            # Calculate total risk with context
            base_risk = (volatility * 1.8 + distance_risk * 1.5 + condition_risk + side_risk + strategy_risk)
            total_risk = max(0.05, min(1.0, base_risk - mtf_risk_reduction))
            
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
            
            # Context-aware risk level
            if total_risk > 0.85:
                risk_level = 'Very High'
            elif total_risk > 0.65:
                risk_level = 'High'
            elif total_risk > 0.45:
                risk_level = 'Medium'
            elif total_risk > 0.25:
                risk_level = 'Low'
            else:
                risk_level = 'Very Low'
            
            return {
                'total_risk_score': total_risk,
                'volatility_risk': volatility,
                'distance_risk': distance_risk,
                'condition_risk': condition_risk,
                'side_risk': side_risk,
                'strategy_risk_adjustment': strategy_risk,
                'mtf_risk_reduction': mtf_risk_reduction,
                'risk_reward_ratio': risk_reward_ratio,
                'risk_level': risk_level,
                'context_factors': {
                    'signal_strength': signal_strength,
                    'entry_strategy': entry_strategy,
                    'extreme_rsi': extreme_rsi,
                    'extreme_bb': extreme_bb,
                    'low_volume': low_volume
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced risk assessment error: {e}")
            return {'total_risk_score': 0.25, 'risk_level': 'Low'}

    # Keep all remaining methods unchanged for backward compatibility
    def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol_data: Dict, 
                                   volume_profile: Dict, fibonacci_data: Dict, 
                                   confluence_zones: List[Dict]) -> Optional[Dict]:
        """Comprehensive symbol analysis - MAIN INTERFACE METHOD"""
        try:
            if df.empty or len(df) < 15:
                self.logger.warning(f"Insufficient data for {symbol_data['symbol']}")
                return None
            
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Generate enhanced signal with context awareness
            volume_entry = volume_profile.get('optimal_entry', {})
            signal = self.generate_enhanced_signal(df, symbol_data, volume_entry, confluence_zones)
            
            if not signal:
                return None
            
            # Enhanced analysis components
            latest = df.iloc[-1]
            
            # All existing analysis components
            technical_summary = self.create_technical_summary(df, latest)
            risk_assessment = self.assess_risk(signal, df, symbol_data)
            volume_analysis = self.analyze_volume_patterns(df)
            trend_strength = self.calculate_trend_strength(df)
            price_action = self.analyze_price_action(df)
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
                'timeframe': '30m'
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis error for {symbol_data.get('symbol', 'Unknown')}: {e}")
            return None

    # Keep all remaining methods unchanged for backward compatibility
    def create_technical_summary(self, df: pd.DataFrame, latest: pd.Series = None) -> Dict:
        """Create technical analysis summary"""
        try:
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
                momentum_score += 1
            if macd > macd_signal:
                momentum_score += 1
            if 20 < stoch_rsi_k < 80:
                momentum_score += 1
            
            momentum_strength = momentum_score / 3.0
            
            # Volatility analysis
            atr = latest.get('atr', latest['close'] * 0.02)
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
            if 'volume' not in df.columns or len(df) < 15:
                return {'pattern': 'insufficient_data', 'strength': 0}
            
            recent_15 = df.tail(15)
            
            # Volume trend
            volume_ma_5 = recent_15['volume'].rolling(5).mean().iloc[-1]
            volume_ma_15 = df['volume'].rolling(15).mean().iloc[-1]
            
            # Buying vs selling pressure
            up_volume = recent_15[recent_15['close'] > recent_15['open']]['volume'].sum()
            down_volume = recent_15[recent_15['close'] < recent_15['open']]['volume'].sum()
            total_volume = up_volume + down_volume
            
            buying_pressure = up_volume / total_volume if total_volume > 0 else 0.5
            
            # Volume pattern classification
            if volume_ma_5 > volume_ma_15 * 1.25:
                pattern = 'surge'
            elif volume_ma_5 > volume_ma_15 * 1.08:
                pattern = 'increasing'
            elif volume_ma_5 < volume_ma_15 * 0.75:
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
            if len(df) < 30:
                return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'low'}
            
            latest = df.iloc[-1]
            recent_30 = df.tail(30)
            
            # Price trend
            price_change_15 = (latest['close'] - recent_30.iloc[-15]['close']) / recent_30.iloc[-15]['close']
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
            
            # Direction determination
            if price_change_15 > 0.015 and ma_alignment_score > 0:
                direction = 'strong_bullish'
            elif price_change_15 > 0 and ma_alignment_score >= 0:
                direction = 'bullish'
            elif price_change_15 < -0.015 and ma_alignment_score < 0:
                direction = 'strong_bearish'
            elif price_change_15 < 0 and ma_alignment_score <= 0:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            consistency_level = 'high' if consistency > 0.65 else 'medium' if consistency > 0.35 else 'low'
            
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
            if len(df) < 8:
                return {'pattern': 'insufficient_data', 'strength': 0}
            
            recent_8 = df.tail(8)
            latest = df.iloc[-1]
            
            # Candlestick patterns
            body_size = abs(latest['close'] - latest['open']) / latest['open']
            upper_shadow = latest['high'] - max(latest['close'], latest['open'])
            lower_shadow = min(latest['close'], latest['open']) - latest['low']
            
            # Pattern identification
            patterns = []
            
            # Doji-like
            if body_size < 0.002:
                patterns.append('doji')
            
            # Hammer/Shooting star
            if lower_shadow > body_size * 1.5 and upper_shadow < body_size:
                patterns.append('hammer')
            elif upper_shadow > body_size * 1.5 and lower_shadow < body_size:
                patterns.append('shooting_star')
            
            # Support/Resistance test
            recent_lows = recent_8['low'].min()
            recent_highs = recent_8['high'].max()
            
            if latest['low'] <= recent_lows * 1.002:
                patterns.append('support_test')
            if latest['high'] >= recent_highs * 0.998:
                patterns.append('resistance_test')
            
            # Price momentum
            closes = recent_8['close'].values
            momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            
            return {
                'patterns': patterns,
                'momentum': momentum,
                'body_size': body_size,
                'shadow_ratio': (upper_shadow + lower_shadow) / body_size if body_size > 0 else 0,
                'strength': min(1.0, abs(momentum) * 8 + body_size * 40)
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
            if volume_24h > 5_000_000:
                liquidity = 'high'
            elif volume_24h > 500_000:
                liquidity = 'medium'
            else:
                liquidity = 'low'
            
            # Volatility assessment
            atr_pct = latest.get('atr', latest['close'] * 0.02) / latest['close']
            if atr_pct > 0.08:
                volatility_level = 'high'
            elif atr_pct > 0.04:
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
            
            # Market sentiment
            if price_change_24h > 4:
                sentiment = 'very_bullish'
            elif price_change_24h > 1.5:
                sentiment = 'bullish'
            elif price_change_24h < -4:
                sentiment = 'very_bearish'
            elif price_change_24h < -1.5:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            # Trading conditions
            conditions = []
            if liquidity in ['high', 'medium']:
                conditions.append('good_liquidity')
            if volatility_level in ['medium', 'high']:
                conditions.append('sufficient_volatility')
            if abs(price_change_24h) > 0.8:
                conditions.append('active_movement')
            
            return {
                'liquidity': liquidity,
                'volatility_level': volatility_level,
                'sentiment': sentiment,
                'price_change_24h': price_change_24h,
                'volume_24h': volume_24h,
                'trading_conditions': conditions,
                'favorable_for_trading': len(conditions) >= 1
            }
            
        except Exception as e:
            self.logger.error(f"Market conditions assessment error: {e}")
            return {'liquidity': 'unknown', 'volatility_level': 'unknown', 'sentiment': 'neutral'}

    def assess_signal_risk(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Alias for assess_risk method for backward compatibility"""
        return self.assess_risk(signal, df, {})

    def filter_signals_by_quality(self, signals: List[Dict], max_signals: int = 25) -> List[Dict]:
        """Enhanced signal filtering with context awareness"""
        try:
            if not signals:
                return []
            
            quality_signals = []
            
            for signal in signals:
                confidence = signal.get('confidence', 0)
                rr_ratio = signal.get('risk_reward_ratio', 0)
                volume_24h = signal.get('volume_24h', 0)
                
                # Get context details
                analysis_details = signal.get('analysis_details', {})
                signal_strength = analysis_details.get('signal_strength', 'weak')
                entry_strategy = signal.get('entry_strategy', 'immediate')
                
                # Context-aware quality thresholds
                if signal_strength == 'strong':
                    min_confidence = 45
                    min_rr = 1.6
                    min_volume = 200_000
                elif signal_strength == 'moderate':
                    min_confidence = 50
                    min_rr = 1.8
                    min_volume = 400_000
                else:  # weak
                    min_confidence = 55
                    min_rr = 2.0
                    min_volume = 600_000
                
                # Bonus for pullback strategies
                if entry_strategy == 'wait_for_pullback':
                    min_confidence -= 3
                    min_rr -= 0.1
                
                # Quality score calculation
                quality_score = (
                    confidence * 0.35 +
                    min(100, rr_ratio * 20) * 0.25 +
                    min(100, volume_24h / 600_000) * 0.15 +
                    (100 if signal.get('order_type') == 'market' else 95) * 0.05
                )
                
                # Context bonuses
                if signal_strength == 'strong':
                    quality_score += 8
                elif signal_strength == 'moderate':
                    quality_score += 4
                
                if entry_strategy == 'wait_for_pullback':
                    quality_score += 5
                
                signal['quality_score'] = quality_score
                
                # Apply filters
                if (confidence >= min_confidence and rr_ratio >= min_rr and volume_24h >= min_volume):
                    quality_signals.append(signal)
            
            # Sort by quality score
            quality_signals.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return quality_signals[:max_signals]
            
        except Exception as e:
            self.logger.error(f"Enhanced signal filtering error: {e}")
            return signals[:max_signals]

    def generate_trading_summary(self, opportunities: List[Dict]) -> Dict:
        """Enhanced trading summary with context awareness"""
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
            
            # Context-aware confidence distribution
            high_confidence = len([op for op in opportunities if op['confidence'] >= 65])
            medium_confidence = len([op for op in opportunities if 50 <= op['confidence'] < 65])
            low_confidence = len([op for op in opportunities if op['confidence'] < 50])
            
            # Risk-reward distribution
            excellent_rr = len([op for op in opportunities if op.get('risk_reward_ratio', 0) >= 2.5])
            good_rr = len([op for op in opportunities if 2.0 <= op.get('risk_reward_ratio', 0) < 2.5])
            fair_rr = len([op for op in opportunities if op.get('risk_reward_ratio', 0) < 2.0])
            
            # Strategy distribution
            pullback_strategies = len([op for op in opportunities if op.get('entry_strategy') == 'wait_for_pullback'])
            immediate_strategies = len([op for op in opportunities if op.get('entry_strategy') == 'immediate'])
            
            # Signal strength distribution
            strong_signals = len([op for op in opportunities 
                                if op.get('analysis_details', {}).get('signal_strength') == 'strong'])
            moderate_signals = len([op for op in opportunities 
                                  if op.get('analysis_details', {}).get('signal_strength') == 'moderate'])
            
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
                'strategy_distribution': {
                    'pullback_strategies': pullback_strategies,
                    'immediate_strategies': immediate_strategies
                },
                'signal_strength_distribution': {
                    'strong': strong_signals,
                    'moderate': moderate_signals,
                    'weak': total_ops - strong_signals - moderate_signals
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
                'recommendation': self.get_enhanced_trading_recommendation(opportunities)
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced trading summary generation error: {e}")
            return {'total_opportunities': len(opportunities), 'error': str(e)}

    def get_enhanced_trading_recommendation(self, opportunities: List[Dict]) -> str:
        """Enhanced trading recommendation with context awareness"""
        try:
            if not opportunities:
                return "No signals found. Wait for better market conditions."
            
            total_ops = len(opportunities)
            
            # Context-aware quality assessment
            strong_signals = len([op for op in opportunities 
                                if op.get('analysis_details', {}).get('signal_strength') == 'strong'])
            
            high_quality = len([op for op in opportunities 
                              if op['confidence'] >= 60 and op.get('risk_reward_ratio', 0) >= 2.2])
            
            pullback_opportunities = len([op for op in opportunities 
                                        if op.get('entry_strategy') == 'wait_for_pullback'])
            
            if strong_signals >= 2 and high_quality >= 2:
                return f"Excellent conditions: {strong_signals} strong signals with {high_quality} high-quality setups."
            elif strong_signals >= 1 and high_quality >= 1:
                return f"Good conditions: {strong_signals} strong signal(s) available."
            elif pullback_opportunities >= 2:
                return f"Patient approach recommended: {pullback_opportunities} pullback opportunities identified."
            elif total_ops >= 3:
                return f"Fair conditions: {total_ops} signals available, use careful position sizing."
            else:
                return f"Limited opportunities: Only {total_ops} signal(s) found. Consider waiting."
                
        except Exception:
            return "Review signals carefully before trading."

    def get_signal_explanation(self, signal: Dict) -> str:
        """Enhanced signal explanation with context awareness"""
        if not signal:
            return "No signal generated"
        
        side = signal['side'].upper()
        symbol = signal['symbol']
        confidence = signal['confidence']
        entry_strategy = signal.get('entry_strategy', 'immediate')
        
        analysis_details = signal.get('analysis_details', {})
        signal_strength = analysis_details.get('signal_strength', 'moderate')
        
        explanation = f"{side} signal for {symbol} ({confidence:.1f}% confidence, {signal_strength} strength)\n"
        
        if entry_strategy == 'wait_for_pullback':
            pullback_levels = analysis_details.get('pullback_levels', [])
            if pullback_levels:
                level = pullback_levels[0]
                explanation += f"Strategy: Wait for pullback to ${level['price']:.4f} ({level['type']})\n"
        else:
            explanation += f"Strategy: {entry_strategy.replace('_', ' ').title()}\n"
        
        if side == 'BUY':
            explanation += "Context: Oversold conditions with strong support nearby"
        else:
            explanation += "Context: Overbought conditions with resistance overhead"
        
        return explanation

    def estimate_profit_likelihood(self, signal: Dict, latest: pd.Series) -> float:
        """Enhanced profit likelihood estimation with context awareness"""
        entry = signal['entry_price']
        stop = signal['stop_loss']
        tp = signal['take_profit_1']
        atr = latest.get('atr', entry * 0.02)
        side = signal['side'].lower()
        
        # Get context details
        analysis_details = signal.get('analysis_details', {})
        signal_strength = analysis_details.get('signal_strength', 'moderate')
        entry_strategy = signal.get('entry_strategy', 'immediate')
        
        # Distance calculations
        if side == 'buy':
            dist_profit = tp - entry
            dist_loss = entry - stop
            trend_direction = analysis_details.get('component_scores', {}).get('trend', 0)
        else:
            dist_profit = entry - tp
            dist_loss = stop - entry
            trend_direction = 1 - analysis_details.get('component_scores', {}).get('trend', 0)
        
        # Normalize by ATR
        profit_moves = dist_profit / atr if atr > 0 else 0
        loss_moves = dist_loss / atr if atr > 0 else 0
        
        if profit_moves + loss_moves == 0:
            return 0.5
        
        base_likelihood = profit_moves / (profit_moves + loss_moves)
        
        # Context adjustments
        context_multiplier = 1.0
        
        # Signal strength bonus
        if signal_strength == 'strong':
            context_multiplier += 0.15
        elif signal_strength == 'moderate':
            context_multiplier += 0.05
        
        # Entry strategy bonus
        if entry_strategy == 'wait_for_pullback':
            context_multiplier += 0.10
        
        # Trend alignment
        context_multiplier += (trend_direction - 0.5) * 0.20
        
        likelihood = base_likelihood * context_multiplier
        
        return max(0.1, min(0.9, likelihood))

# Utility functions for debugging (enhanced)
def debug_signal_conditions(df: pd.DataFrame, symbol: str, generator: SignalGenerator = None):
    """Enhanced debug function with context awareness"""
    latest = df.iloc[-1]
    
    print(f"\n=== ENHANCED DEBUG: {symbol} ===")
    print(f"RSI: {latest.get('rsi', 'Missing')}")
    print(f"Stoch RSI K: {latest.get('stoch_rsi_k', 'Missing')}")
    print(f"Stoch RSI D: {latest.get('stoch_rsi_d', 'Missing')}")
    print(f"MACD: {latest.get('macd', 'Missing')}")
    print(f"MACD Signal: {latest.get('macd_signal', 'Missing')}")
    print(f"Volume Ratio: {latest.get('volume_ratio', 'Missing')}")
    print(f"BB Position: {latest.get('bb_position', 'Missing')}")
    
    if generator:
        # Analyze trend context
        trend_context = generator.analyze_trend_context(df, latest)
        print(f"\nTrend Context:")
        print(f"Direction: {trend_context['direction']}")
        print(f"Strength: {trend_context['strength']}")
        print(f"Momentum: {trend_context['momentum']}")
        
        # Get dynamic thresholds
        market_structure = {'near_support': False, 'near_resistance': False}
        buy_thresholds = generator.calculate_dynamic_buy_thresholds(trend_context, market_structure)
        sell_thresholds = generator.calculate_dynamic_sell_thresholds(trend_context, market_structure)
        
        print(f"\nDynamic Thresholds:")
        print(f"BUY RSI Threshold: {buy_thresholds['rsi_threshold']}")
        print(f"SELL RSI Threshold: {sell_thresholds['rsi_threshold']}")
    
    # Check traditional conditions
    rsi = latest.get('rsi', 50)
    stoch_k = latest.get('stoch_rsi_k', 50)
    
    print(f"\nTraditional Conditions:")
    print(f"RSI < 35: {rsi < 35} (RSI: {rsi})")
    print(f"RSI > 70: {rsi > 70}")
    print(f"Stoch RSI K < 30: {stoch_k < 30} (K: {stoch_k})")
    print(f"Stoch RSI K > 75: {stoch_k > 75}")
    
    if generator:
        print(f"\nContext-Aware Conditions:")
        print(f"BUY RSI < {buy_thresholds['rsi_threshold']}: {rsi < buy_thresholds['rsi_threshold']}")
        print(f"SELL RSI > {sell_thresholds['rsi_threshold']}: {rsi > sell_thresholds['rsi_threshold']}")