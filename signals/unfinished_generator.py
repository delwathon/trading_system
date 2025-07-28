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
                self.logger.info(f"No resistance found above entry for {symbol_data['symbol']}, skipping BUY signal.")
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
            take_profit_2 = max(further_resistances) if further_resistances else take_profit_1 * 1.03

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
                self.logger.info(f"No support found below entry for {symbol_data['symbol']}, skipping SELL signal.")
                return None

            atr = latest.get('atr', current_price * 0.02)
            stop_loss = optimal_entry + (3.0 * atr)
            take_profit_1 = max(supports)
            further_supports = [s for s in supports if s < take_profit_1]
            take_profit_2 = min(further_supports) if further_supports else take_profit_1 * 0.97

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
        """Enhanced ranking with context-aware scoring"""
        try:
            opportunities = []
            
            for signal in signals:
                # Get enhanced analysis details
                analysis_details = signal.get('analysis_details', {})
                confidence = signal['confidence']
                
                # Base scoring components
                confidence_score = confidence / 100
                rr_ratio = signal.get('risk_reward_ratio', 1)
                rr_score = min(1.0, rr_ratio / 3.0)
                
                # Context-aware scoring bonuses
                context_bonus = 0
                
                # Signal strength bonus
                signal_strength = analysis_details.get('signal_strength', 'weak')
                if signal_strength == 'strong':
                    context_bonus += 0.15
                elif signal_strength == 'moderate':
                    context_bonus += 0.08
                
                # Entry strategy bonus
                entry_strategy = signal.get('entry_strategy', 'immediate')
                if entry_strategy == 'wait_for_pullback':
                    context_bonus += 0.10  # Premium for patient entries
                
                # Component scores bonus
                component_scores = analysis_details.get('component_scores', {})
                technical_score = component_scores.get('technical', 0)
                structure_score = component_scores.get('structure', 0)
                trend_score = component_scores.get('trend', 0)
                
                structure_bonus = structure_score * 0.12
                trend_bonus = trend_score * 0.10
                
                # Volume and distance scores
                volume_score = min(1.0, signal['volume_24h'] / 15_000_000)
                distance = abs(signal['entry_price'] - signal['current_price']) / signal['current_price']
                distance_score = max(0, 1 - distance * 5)
                
                # MTF analysis (if available)
                mtf_analysis = signal.get('mtf_analysis', {})
                confirmed_count = len(mtf_analysis.get('confirmed_timeframes', []))
                total_timeframes = len(getattr(self.config, 'confirmation_timeframes', [])) or 3
                if total_timeframes > 0:
                    mtf_bonus = (confirmed_count / total_timeframes) * 0.12
                else:
                    mtf_bonus = 0
                
                # Calculate total score with context awareness
                total_score = (
                    confidence_score * 0.30 +
                    rr_score * 0.20 +
                    structure_bonus +
                    trend_bonus +
                    volume_score * 0.08 +
                    distance_score * 0.03 +
                    context_bonus +
                    mtf_bonus
                )
                
                # Enhanced priority calculation
                if confidence >= 75 and rr_ratio >= 2.8 and signal_strength == 'strong':
                    base_priority = 1500
                elif confidence >= 65 and rr_ratio >= 2.3 and signal_strength in ['strong', 'moderate']:
                    base_priority = 800
                elif confidence >= 55 and rr_ratio >= 2.0:
                    base_priority = 400
                elif confidence >= 50 and rr_ratio >= 1.8:
                    base_priority = 200
                else:
                    base_priority = 80
                
                # Entry strategy multiplier
                strategy_multiplier = {
                    'wait_for_pullback': 1.3,
                    'immediate': 1.0,
                    'limit_order': 1.1
                }.get(entry_strategy, 1.0)
                
                final_priority = int(base_priority * strategy_multiplier)
                
                opportunities.append({
                    **signal,
                    'score': total_score,
                    'priority': final_priority,
                    'context_scores': {
                        'confidence_score': confidence_score,
                        'rr_score': rr_score,
                        'structure_bonus': structure_bonus,
                        'trend_bonus': trend_bonus,
                        'context_bonus': context_bonus,
                        'technical_score': technical_score,
                        'volume_score': volume_score,
                        'distance_score': distance_score
                    }
                })

            # Sort by priority, then score
            opportunities.sort(key=lambda x: (x['priority'], x['score']), reverse=True)
            return opportunities[:self.config.charts_per_batch]

        except Exception as e:
            self.logger.error(f"Enhanced ranking error: {e}")
            return signals

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
            technical_summary = self.create_technical_summary(df, latest