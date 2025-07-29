"""
Configurable Multi-Timeframe Level Optimizer for Bybit Trading System
Automatically adapts to ANY timeframe configuration and optimizes entry/exit levels
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.config import EnhancedSystemConfig
from core.exchange import ExchangeManager
from analysis.technical import EnhancedTechnicalAnalysis
from analysis.volume_profile import VolumeProfileAnalyzer
from analysis.fibonacci import FibonacciConfluenceAnalyzer
from signals.generator import SignalGenerator

class TimeframeRole(Enum):
    """Define roles that timeframes can play in analysis"""
    PRIMARY = "primary"
    HIGHER_CONFIRMATION = "higher_confirmation"  # TFs higher than primary
    LOWER_TIMING = "lower_timing"               # TFs lower than primary
    EQUAL_CONFLUENCE = "equal_confluence"       # TFs equal to primary

@dataclass
class TimeframeWeight:
    """Weight configuration for different timeframe roles"""
    primary: float = 1.0
    higher_confirmation: float = 1.5  # Base weight for higher TFs
    lower_timing: float = 0.6         # Lower weight for timing TFs
    equal_confluence: float = 0.8     # Weight for equal TFs

class ConfigurableMultiTimeframeOptimizer:
    """Fully configurable MTF optimizer that adapts to any timeframe setup"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize timeframe categorization
        self.timeframe_categories = self._categorize_timeframes()
        self.timeframe_weights = self._calculate_dynamic_weights()
        
        # Configuration for level optimization
        self.level_optimization_config = {
            'enabled': True,
            'min_timeframes_for_optimization': 2,
            'weight_strategy': 'adaptive',  # 'adaptive', 'linear', 'exponential'
            'proximity_bias_enabled': True,
            'confluence_bonus_enabled': True,
            'validation_strict': True
        }
        
        self._log_configuration()
    
    def _categorize_timeframes(self) -> Dict[str, TimeframeRole]:
        """Automatically categorize timeframes based on their relationship to primary"""
        categories = {}
        primary_minutes = self._timeframe_to_minutes(self.config.timeframe)
        
        # Primary timeframe
        categories[self.config.timeframe] = TimeframeRole.PRIMARY
        
        # Categorize confirmation timeframes
        for tf in getattr(self.config, 'confirmation_timeframes', []):
            tf_minutes = self._timeframe_to_minutes(tf)
            
            if tf_minutes > primary_minutes:
                categories[tf] = TimeframeRole.HIGHER_CONFIRMATION
            elif tf_minutes < primary_minutes:
                categories[tf] = TimeframeRole.LOWER_TIMING
            else:
                categories[tf] = TimeframeRole.EQUAL_CONFLUENCE
        
        return categories
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes for comparison"""
        try:
            if timeframe.endswith('m'):
                return int(timeframe[:-1])
            elif timeframe.endswith('h'):
                return int(timeframe[:-1]) * 60
            elif timeframe.endswith('d'):
                return int(timeframe[:-1]) * 60 * 24
            elif timeframe.endswith('w'):
                return int(timeframe[:-1]) * 60 * 24 * 7
            else:
                # Default mapping for common timeframes
                mapping = {
                    '1': 1, '5': 5, '15': 15, '30': 30,
                    '60': 60, '240': 240, '360': 360, '720': 720, '1440': 1440
                }
                return mapping.get(timeframe, 60)
        except (ValueError, IndexError):
            self.logger.warning(f"Unknown timeframe format: {timeframe}, defaulting to 60m")
            return 60
    
    def _calculate_dynamic_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on timeframe relationships"""
        weights = {}
        base_weights = TimeframeWeight()
        primary_minutes = self._timeframe_to_minutes(self.config.timeframe)
        
        for tf, role in self.timeframe_categories.items():
            tf_minutes = self._timeframe_to_minutes(tf)
            
            if role == TimeframeRole.PRIMARY:
                weights[tf] = base_weights.primary
            
            elif role == TimeframeRole.HIGHER_CONFIRMATION:
                # Higher timeframes get exponentially more weight
                ratio = tf_minutes / primary_minutes
                if ratio >= 6:      # 6h+ when primary is 1h
                    multiplier = 2.5
                elif ratio >= 4:    # 4h+ when primary is 1h  
                    multiplier = 2.0
                elif ratio >= 2:    # 2h+ when primary is 1h
                    multiplier = 1.5
                else:
                    multiplier = 1.3
                
                weights[tf] = base_weights.higher_confirmation * multiplier
            
            elif role == TimeframeRole.LOWER_TIMING:
                # Lower timeframes get reduced weight, more reduction for smaller TFs
                ratio = primary_minutes / tf_minutes
                if ratio >= 4:      # 15m when primary is 1h
                    multiplier = 0.4
                elif ratio >= 2:    # 30m when primary is 1h
                    multiplier = 0.6
                else:
                    multiplier = 0.8
                
                weights[tf] = base_weights.lower_timing * multiplier
            
            else:  # EQUAL_CONFLUENCE
                weights[tf] = base_weights.equal_confluence
        
        return weights
    
    def _log_configuration(self):
        """Log the current MTF optimization configuration"""
        self.logger.info("🎯 CONFIGURABLE MTF LEVEL OPTIMIZER INITIALIZED")
        self.logger.info(f"   Primary Timeframe: {self.config.timeframe}")
        self.logger.info(f"   Confirmation Timeframes: {getattr(self.config, 'confirmation_timeframes', [])}")
        
        # Log timeframe categorization
        higher_tfs = [tf for tf, role in self.timeframe_categories.items() if role == TimeframeRole.HIGHER_CONFIRMATION]
        lower_tfs = [tf for tf, role in self.timeframe_categories.items() if role == TimeframeRole.LOWER_TIMING]
        equal_tfs = [tf for tf, role in self.timeframe_categories.items() if role == TimeframeRole.EQUAL_CONFLUENCE]
        
        self.logger.info(f"   Higher TFs (Trend Confirmation): {higher_tfs}")
        self.logger.info(f"   Lower TFs (Entry Timing): {lower_tfs}")
        self.logger.info(f"   Equal TFs (Confluence): {equal_tfs}")
        
        # Log weights
        self.logger.debug("   Timeframe Weights:")
        for tf, weight in self.timeframe_weights.items():
            role = self.timeframe_categories[tf].value
            self.logger.debug(f"     {tf}: {weight:.1f}x ({role})")
    
    def analyze_symbol_with_adaptive_optimization(self, symbol_data: Dict, primary_signal: Dict) -> Dict:
        """Main method: Analyze symbol with adaptive level optimization"""
        try:
            symbol = symbol_data['symbol']
            primary_side = primary_signal['side']
            
            self.logger.debug(f"🔍 Adaptive MTF analysis for {symbol}")
            
            # Collect signals from all configured timeframes
            timeframe_signals = self._collect_all_timeframe_signals(symbol_data)
            
            # Categorize signals by alignment with primary
            signal_analysis = self._analyze_signal_alignment(timeframe_signals, primary_side)
            
            # Perform level optimization if enough aligned signals exist
            optimization_result = self._optimize_levels_adaptively(
                signal_analysis['aligned_signals'], 
                symbol_data['current_price'], 
                primary_side
            )
            
            # Apply optimization to primary signal if successful
            if optimization_result['optimization_applied']:
                self._apply_optimization_to_signal(primary_signal, optimization_result)
            
            # Calculate MTF confirmation metrics
            mtf_metrics = self._calculate_adaptive_mtf_metrics(signal_analysis, optimization_result)
            
            # Prepare comprehensive result
            result = self._compile_comprehensive_result(
                signal_analysis, optimization_result, mtf_metrics, timeframe_signals
            )
            
            self.logger.debug(f"   ✅ {symbol} MTF analysis complete - Status: {result['mtf_status']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive MTF analysis error for {symbol}: {e}")
            return self._get_fallback_result()
    
    def _collect_all_timeframe_signals(self, symbol_data: Dict) -> Dict[str, Optional[Dict]]:
        """Collect signals from all configured timeframes"""
        signals = {}
        
        # Note: This method needs to be implemented by calling your existing 
        # signal generation logic for each timeframe
        # For now, returning placeholder structure
        
        for tf in self.timeframe_categories.keys():
            try:
                if tf == self.config.timeframe:
                    # Primary signal already exists, don't regenerate
                    continue
                
                # This would call your existing MTF analysis method
                # signal = self._generate_signal_for_timeframe(symbol_data, tf)
                # signals[tf] = signal
                
                # Placeholder for now - implement with your existing logic
                signals[tf] = None
                
            except Exception as e:
                self.logger.debug(f"   Error collecting {tf} signal: {e}")
                signals[tf] = None
        
        return signals
    
    def _analyze_signal_alignment(self, timeframe_signals: Dict, primary_side: str) -> Dict:
        """Analyze how signals align with primary direction"""
        aligned_signals = []
        conflicting_signals = []
        neutral_timeframes = []
        
        # Add primary signal data (assumed to be aligned)
        primary_tf = self.config.timeframe
        aligned_signals.append({
            'timeframe': primary_tf,
            'signal': None,  # Will be filled by caller
            'weight': self.timeframe_weights[primary_tf],
            'role': self.timeframe_categories[primary_tf],
            'confidence_contribution': 1.0
        })
        
        # Analyze other timeframes
        for tf, signal in timeframe_signals.items():
            if signal is None:
                neutral_timeframes.append(tf)
                continue
            
            if signal.get('side') == primary_side:
                aligned_signals.append({
                    'timeframe': tf,
                    'signal': signal,
                    'weight': self.timeframe_weights[tf],
                    'role': self.timeframe_categories[tf],
                    'confidence_contribution': signal.get('confidence', 50) / 100
                })
            else:
                conflicting_signals.append({
                    'timeframe': tf,
                    'signal': signal,
                    'weight': self.timeframe_weights[tf],
                    'role': self.timeframe_categories[tf]
                })
        
        return {
            'aligned_signals': aligned_signals,
            'conflicting_signals': conflicting_signals,
            'neutral_timeframes': neutral_timeframes,
            'alignment_strength': len(aligned_signals) / len(self.timeframe_categories)
        }
    
    def _optimize_levels_adaptively(self, aligned_signals: List[Dict], 
                                  current_price: float, side: str) -> Dict:
        """Adaptively optimize levels based on aligned signals"""
        try:
            if len(aligned_signals) < self.level_optimization_config['min_timeframes_for_optimization']:
                return {
                    'optimization_applied': False,
                    'reason': 'insufficient_aligned_signals',
                    'aligned_count': len(aligned_signals)
                }
            
            # Extract level data from aligned signals
            level_data = self._extract_level_data(aligned_signals)
            
            if not level_data['valid']:
                return {
                    'optimization_applied': False,
                    'reason': 'invalid_level_data',
                    'level_data': level_data
                }
            
            # Calculate optimized levels using configured strategy
            optimized_levels = self._calculate_optimized_levels(
                level_data, current_price, side
            )
            
            # Validate optimized levels
            if not self._validate_optimized_levels(optimized_levels, side):
                return {
                    'optimization_applied': False,
                    'reason': 'validation_failed',
                    'attempted_levels': optimized_levels
                }
            
            return {
                'optimization_applied': True,
                'optimized_levels': optimized_levels,
                'level_data': level_data,
                'optimization_method': self.level_optimization_config['weight_strategy'],
                'timeframes_used': [s['timeframe'] for s in aligned_signals],
                'total_weight': sum(s['weight'] for s in aligned_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Level optimization error: {e}")
            return {
                'optimization_applied': False,
                'reason': 'optimization_error',
                'error': str(e)
            }
    
    def _extract_level_data(self, aligned_signals: List[Dict]) -> Dict:
        """Extract and validate level data from aligned signals"""
        entry_levels = []
        stop_levels = []
        tp1_levels = []
        tp2_levels = []
        
        for signal_data in aligned_signals:
            signal = signal_data.get('signal')
            if not signal:
                continue  # Skip primary signal placeholder
            
            # Validate required fields exist
            required_fields = ['entry_price', 'stop_loss', 'take_profit_1']
            if not all(field in signal for field in required_fields):
                continue
            
            weight = signal_data['weight']
            tf = signal_data['timeframe']
            confidence = signal_data['confidence_contribution']
            
            # Collect weighted levels
            entry_levels.append({
                'price': signal['entry_price'],
                'weight': weight,
                'timeframe': tf,
                'confidence': confidence,
                'role': signal_data['role']
            })
            
            stop_levels.append({
                'price': signal['stop_loss'],
                'weight': weight,
                'timeframe': tf,
                'role': signal_data['role']
            })
            
            tp1_levels.append({
                'price': signal['take_profit_1'],
                'weight': weight,
                'timeframe': tf,
                'role': signal_data['role']
            })
            
            if 'take_profit_2' in signal:
                tp2_levels.append({
                    'price': signal['take_profit_2'],
                    'weight': weight,
                    'timeframe': tf,
                    'role': signal_data['role']
                })
        
        return {
            'valid': len(entry_levels) > 0,
            'entry_levels': entry_levels,
            'stop_levels': stop_levels,
            'tp1_levels': tp1_levels,
            'tp2_levels': tp2_levels,
            'total_signals': len(entry_levels)
        }
    
    def _calculate_optimized_levels(self, level_data: Dict, current_price: float, side: str) -> Dict:
        """Calculate optimized levels using the configured strategy"""
        strategy = self.level_optimization_config['weight_strategy']
        
        if strategy == 'adaptive':
            return self._calculate_adaptive_levels(level_data, current_price, side)
        elif strategy == 'linear':
            return self._calculate_linear_weighted_levels(level_data, current_price, side)
        elif strategy == 'exponential':
            return self._calculate_exponential_weighted_levels(level_data, current_price, side)
        else:
            return self._calculate_adaptive_levels(level_data, current_price, side)
    
    def _calculate_adaptive_levels(self, level_data: Dict, current_price: float, side: str) -> Dict:
        """Calculate levels using adaptive weighting (recommended approach)"""
        try:
            # Entry optimization with proximity bias and role weighting
            optimized_entry = self._optimize_entry_adaptive(
                level_data['entry_levels'], current_price, side
            )
            
            # Stop loss optimization prioritizing higher timeframes
            optimized_stop = self._optimize_stop_adaptive(
                level_data['stop_levels'], optimized_entry, side
            )
            
            # Take profit optimization with confluence detection
            optimized_tp1 = self._optimize_take_profit_adaptive(
                level_data['tp1_levels'], optimized_entry, side, level=1
            )
            
            optimized_tp2 = None
            if level_data['tp2_levels']:
                optimized_tp2 = self._optimize_take_profit_adaptive(
                    level_data['tp2_levels'], optimized_entry, side, level=2
                )
            
            # Calculate risk/reward
            optimized_tp = optimized_tp1 if self.config.default_tp_level == 'take_profit_1' else optimized_tp2

            risk_reward_ratio = self._calculate_risk_reward(
                optimized_entry, optimized_stop, optimized_tp, side
            )
            
            return {
                'entry_price': optimized_entry,
                'stop_loss': optimized_stop,
                'take_profit_1': optimized_tp1,
                'take_profit_2': optimized_tp2,
                'risk_reward_ratio': risk_reward_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Adaptive level calculation error: {e}")
            raise
    
    def _optimize_entry_adaptive(self, entry_levels: List[Dict], current_price: float, side: str) -> float:
        """Optimize entry using adaptive weighting"""
        if not entry_levels:
            return current_price
        
        total_weighted_price = 0
        total_weight = 0
        
        for level in entry_levels:
            # Base weight from timeframe role
            weight = level['weight']
            
            # Proximity bias (prefer entries closer to current price)
            if self.level_optimization_config['proximity_bias_enabled']:
                distance = abs(level['price'] - current_price) / current_price
                proximity_multiplier = max(0.5, 1 - distance * 20)  # Reduce weight for distant entries
                weight *= proximity_multiplier
            
            # Confidence multiplier
            weight *= level.get('confidence', 1.0)
            
            # Role-based adjustment
            if level['role'] == TimeframeRole.HIGHER_CONFIRMATION:
                weight *= 1.2  # Boost higher timeframe entries
            elif level['role'] == TimeframeRole.LOWER_TIMING:
                weight *= 0.8  # Reduce lower timeframe weight
            
            total_weighted_price += level['price'] * weight
            total_weight += weight
        
        return total_weighted_price / total_weight if total_weight > 0 else current_price
    
    def _optimize_stop_adaptive(self, stop_levels: List[Dict], entry_price: float, side: str) -> float:
        """Optimize stop loss using adaptive weighting, prioritizing higher timeframes"""
        if not stop_levels:
            # Fallback: 2% stop
            return entry_price * (0.98 if side == 'buy' else 1.02)
        
        # Filter valid stops based on side
        valid_stops = []
        for level in stop_levels:
            if side == 'buy' and level['price'] < entry_price:
                valid_stops.append(level)
            elif side == 'sell' and level['price'] > entry_price:
                valid_stops.append(level)
        
        if not valid_stops:
            return entry_price * (0.98 if side == 'buy' else 1.02)
        
        # Weight calculation prioritizing higher timeframes
        total_weighted_price = 0
        total_weight = 0
        
        for level in valid_stops:
            weight = level['weight']
            
            # Strong boost for higher timeframe stops
            if level['role'] == TimeframeRole.HIGHER_CONFIRMATION:
                weight *= 2.0
            
            # Distance from entry consideration (prefer not too tight, not too loose)
            distance = abs(level['price'] - entry_price) / entry_price
            optimal_distance = 0.025  # 2.5% ideal stop distance
            distance_score = 1 / (1 + abs(distance - optimal_distance) / optimal_distance)
            weight *= distance_score
            
            total_weighted_price += level['price'] * weight
            total_weight += weight
        
        return total_weighted_price / total_weight
    
    def _optimize_take_profit_adaptive(self, tp_levels: List[Dict], entry_price: float, 
                                     side: str, level: int = 1) -> Optional[float]:
        """Optimize take profit using confluence detection"""
        if not tp_levels:
            # Fallback based on level
            multiplier = 1.04 if level == 1 else 1.08
            return entry_price * multiplier if side == 'buy' else entry_price / multiplier
        
        # Filter valid TPs based on side
        valid_tps = []
        for tp in tp_levels:
            if side == 'buy' and tp['price'] > entry_price:
                valid_tps.append(tp)
            elif side == 'sell' and tp['price'] < entry_price:
                valid_tps.append(tp)
        
        if not valid_tps:
            multiplier = 1.04 if level == 1 else 1.08
            return entry_price * multiplier if side == 'buy' else entry_price / multiplier
        
        # Calculate weights with confluence bonus
        total_weighted_price = 0
        total_weight = 0
        
        for tp in valid_tps:
            weight = tp['weight']
            
            # Confluence bonus (if multiple TPs are close to each other)
            if self.level_optimization_config['confluence_bonus_enabled']:
                confluence_score = 1.0
                for other_tp in valid_tps:
                    if other_tp != tp:
                        price_diff = abs(tp['price'] - other_tp['price']) / entry_price
                        if price_diff < 0.015:  # Within 1.5%
                            confluence_score += 0.3
                
                weight *= confluence_score
            
            # Higher timeframe TP preference
            if tp['role'] == TimeframeRole.HIGHER_CONFIRMATION:
                weight *= 1.5
            
            total_weighted_price += tp['price'] * weight
            total_weight += weight
        
        return total_weighted_price / total_weight
    
    def _calculate_linear_weighted_levels(self, level_data: Dict, current_price: float, side: str) -> Dict:
        """Calculate levels using simple linear weighting"""
        # Simplified linear weighted average
        def weighted_average(levels, price_key='price'):
            if not levels:
                return current_price
            total_weight = sum(l['weight'] for l in levels)
            return sum(l[price_key] * l['weight'] for l in levels) / total_weight
        
        return {
            'entry_price': weighted_average(level_data['entry_levels']),
            'stop_loss': weighted_average(level_data['stop_levels']),
            'take_profit_1': weighted_average(level_data['tp1_levels']),
            'take_profit_2': weighted_average(level_data['tp2_levels']) if level_data['tp2_levels'] else None,
            'risk_reward_ratio': 0  # Calculate separately
        }
    
    def _calculate_exponential_weighted_levels(self, level_data: Dict, current_price: float, side: str) -> Dict:
        """Calculate levels using exponential weighting (higher TFs get much more weight)"""
        def exponential_weighted_average(levels, price_key='price'):
            if not levels:
                return current_price
            
            # Apply exponential scaling to weights
            total_weighted_price = 0
            total_weight = 0
            
            for level in levels:
                # Exponential weight scaling
                exp_weight = level['weight'] ** 1.5
                total_weighted_price += level[price_key] * exp_weight
                total_weight += exp_weight
            
            return total_weighted_price / total_weight
        
        return {
            'entry_price': exponential_weighted_average(level_data['entry_levels']),
            'stop_loss': exponential_weighted_average(level_data['stop_levels']),
            'take_profit_1': exponential_weighted_average(level_data['tp1_levels']),
            'take_profit_2': exponential_weighted_average(level_data['tp2_levels']) if level_data['tp2_levels'] else None,
            'risk_reward_ratio': 0
        }
    
    def _calculate_risk_reward(self, entry: float, stop: float, tp: float, side: str) -> float:
        """Calculate risk/reward ratio"""
        try:
            if side == 'buy':
                risk = entry - stop
                reward = tp - entry
            else:
                risk = stop - entry  
                reward = entry - tp
            
            return reward / risk if risk > 0 else 0
        except Exception:
            return 0
    
    def _validate_optimized_levels(self, levels: Dict, side: str) -> bool:
        """Validate that optimized levels make sense"""
        try:
            entry = levels.get('entry_price')
            stop = levels.get('stop_loss')
            tp1 = levels.get('take_profit_1')
            
            if not all([entry, stop, tp1]):
                return False
            
            # Direction validation
            if side == 'buy':
                if not (stop < entry < tp1):
                    return False
            else:
                if not (tp1 < entry < stop):
                    return False
            
            # Risk/reward validation
            rr = levels.get('risk_reward_ratio', 0)
            if rr < 1.2:  # Minimum acceptable R/R
                return False
            
            # Risk percentage validation
            risk_pct = abs(entry - stop) / entry
            if risk_pct < 0.005 or risk_pct > 0.20:  # 0.5% to 20%
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_optimization_to_signal(self, primary_signal: Dict, optimization_result: Dict):
        """Apply optimization results to the primary signal"""
        if not optimization_result.get('optimization_applied'):
            return
        
        optimized_levels = optimization_result['optimized_levels']
        
        # Store original levels
        original_levels = {
            'entry_price': primary_signal.get('entry_price'),
            'stop_loss': primary_signal.get('stop_loss'),
            'take_profit_1': primary_signal.get('take_profit_1'),
            'take_profit_2': primary_signal.get('take_profit_2'),
            'risk_reward_ratio': primary_signal.get('risk_reward_ratio', 0)
        }
        
        # Update signal with optimized levels
        primary_signal.update(optimized_levels)
        
        # Add optimization metadata
        primary_signal['level_optimization'] = {
            'enabled': True,
            'method': optimization_result['optimization_method'],
            'timeframes_used': optimization_result['timeframes_used'],
            'total_weight': optimization_result['total_weight'],
            'original_levels': original_levels,
            'improvements': {
                'entry_change_pct': abs(optimized_levels['entry_price'] - original_levels['entry_price']) / original_levels['entry_price'] * 100 if original_levels['entry_price'] else 0,
                'rr_improvement': optimized_levels['risk_reward_ratio'] - original_levels['risk_reward_ratio']
            },
            'optimization_config': self.level_optimization_config.copy()
        }
    
    def _calculate_adaptive_mtf_metrics(self, signal_analysis: Dict, optimization_result: Dict) -> Dict:
        """Calculate MTF confirmation metrics adaptively"""
        aligned_count = len(signal_analysis['aligned_signals'])
        conflicting_count = len(signal_analysis['conflicting_signals'])
        total_configured = len(self.timeframe_categories)
        
        # Calculate confirmation strength
        confirmation_strength = aligned_count / total_configured
        
        # Determine MTF status
        if confirmation_strength >= 0.8:
            mtf_status = 'STRONG'
            confidence_boost = 12 + (confirmation_strength - 0.8) * 25  # 12-17 points
        elif confirmation_strength >= 0.6:
            mtf_status = 'PARTIAL'
            confidence_boost = 6 + (confirmation_strength - 0.6) * 30  # 6-12 points
        elif confirmation_strength >= 0.4:
            mtf_status = 'WEAK'
            confidence_boost = 2 + (confirmation_strength - 0.4) * 20  # 2-6 points
        else:
            mtf_status = 'POOR'
            confidence_boost = 0
        
        # Penalty for conflicting signals
        if conflicting_count > 0:
            conflict_penalty = min(8, conflicting_count * 3)
            confidence_boost = max(0, confidence_boost - conflict_penalty)
        
        # Bonus for level optimization
        optimization_bonus = 3 if optimization_result.get('optimization_applied') else 0
        
        return {
            'mtf_status': mtf_status,
            'confirmation_strength': confirmation_strength,
            'mtf_confidence_boost': confidence_boost + optimization_bonus,
            'aligned_count': aligned_count,
            'conflicting_count': conflicting_count,
            'total_configured': total_configured,
            'optimization_bonus': optimization_bonus
        }
    
    def _compile_comprehensive_result(self, signal_analysis: Dict, optimization_result: Dict, 
                                    mtf_metrics: Dict, timeframe_signals: Dict) -> Dict:
        """Compile comprehensive MTF analysis result"""
        return {
            'confirmed_timeframes': [s['timeframe'] for s in signal_analysis['aligned_signals'][1:]],  # Exclude primary
            'conflicting_timeframes': [s['timeframe'] for s in signal_analysis['conflicting_signals']],
            'neutral_timeframes': signal_analysis['neutral_timeframes'],
            'confirmation_strength': mtf_metrics['confirmation_strength'],
            'mtf_confidence_boost': mtf_metrics['mtf_confidence_boost'],
            'mtf_status': mtf_metrics['mtf_status'],
            'timeframe_signals': timeframe_signals,
            'level_optimization_applied': optimization_result.get('optimization_applied', False),
            'optimization_details': optimization_result if optimization_result.get('optimization_applied') else None,
            'timeframe_configuration': {
                'primary_timeframe': self.config.timeframe,
                'confirmation_timeframes': getattr(self.config, 'confirmation_timeframes', []),
                'timeframe_categories': {tf: role.value for tf, role in self.timeframe_categories.items()},
                'timeframe_weights': self.timeframe_weights.copy(),
                'optimization_config': self.level_optimization_config.copy()
            }
        }
    
    def _get_fallback_result(self) -> Dict:
        """Get fallback result when MTF analysis fails"""
        return {
            'confirmed_timeframes': [],
            'conflicting_timeframes': [],
            'neutral_timeframes': getattr(self.config, 'confirmation_timeframes', []),
            'confirmation_strength': 0.0,
            'mtf_confidence_boost': 0.0,
            'mtf_status': 'ERROR',
            'timeframe_signals': {},
            'level_optimization_applied': False,
            'optimization_details': None,
            'timeframe_configuration': {
                'primary_timeframe': self.config.timeframe,
                'confirmation_timeframes': getattr(self.config, 'confirmation_timeframes', []),
                'error': 'MTF analysis failed'
            }
        }
    
    def update_configuration(self, new_config):
        """Update configuration and recalculate weights/categories"""
        self.config = new_config
        self.timeframe_categories = self._categorize_timeframes()
        self.timeframe_weights = self._calculate_dynamic_weights()
        self._log_configuration()
        
        self.logger.info("🔄 MTF Optimizer configuration updated successfully")
    
    def get_configuration_summary(self) -> Dict:
        """Get current configuration summary for logging/debugging"""
        return {
            'primary_timeframe': self.config.timeframe,
            'confirmation_timeframes': getattr(self.config, 'confirmation_timeframes', []),
            'timeframe_categories': {tf: role.value for tf, role in self.timeframe_categories.items()},
            'timeframe_weights': self.timeframe_weights.copy(),
            'level_optimization_enabled': self.level_optimization_config['enabled'],
            'optimization_strategy': self.level_optimization_config['weight_strategy'],
            'min_timeframes_required': self.level_optimization_config['min_timeframes_for_optimization']
        }


# Integration class to connect with existing system
class EnhancedMultiTimeframeAnalyzer:
    """Enhanced MTF analyzer that integrates the configurable optimizer with existing system"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.exchange_manager = ExchangeManager(config)
        self.enhanced_ta = EnhancedTechnicalAnalysis()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.fibonacci_analyzer = FibonacciConfluenceAnalyzer()
        self.signal_generator = SignalGenerator(config)
        
        # Initialize the configurable optimizer
        self.optimizer = ConfigurableMultiTimeframeOptimizer(config, self.logger)
        
        self.logger.info("✅ Enhanced MTF Analyzer with Configurable Optimization initialized")
    
    def analyze_symbol_multi_timeframe(self, symbol_data: Dict, primary_signal: Dict) -> Dict:
        """Main interface method - integrates with existing system.py calls"""
        try:
            symbol = symbol_data['symbol']
            primary_side = primary_signal['side']
            
            self.logger.debug(f"🔍 Enhanced MTF analysis for {symbol}")
            
            # Collect signals from all confirmation timeframes using existing logic
            timeframe_signals = {}
            aligned_signals = [{
                'timeframe': self.config.timeframe,
                'signal': primary_signal,
                'weight': self.optimizer.timeframe_weights.get(self.config.timeframe, 1.0),
                'role': self.optimizer.timeframe_categories.get(self.config.timeframe),
                'confidence_contribution': primary_signal['confidence'] / 100
            }]
            
            confirmed_timeframes = []
            conflicting_timeframes = []
            neutral_timeframes = []
            
            # Analyze each confirmation timeframe
            for timeframe in self.config.confirmation_timeframes:
                try:
                    self.logger.debug(f"   Analyzing {timeframe} timeframe...")
                    
                    # Get signal for this timeframe using existing logic
                    timeframe_signal = self._get_timeframe_signal(symbol_data, timeframe)
                    timeframe_signals[timeframe] = timeframe_signal
                    
                    if timeframe_signal:
                        if timeframe_signal['side'] == primary_side:
                            confirmed_timeframes.append(timeframe)
                            
                            # Add to aligned signals for level optimization
                            aligned_signals.append({
                                'timeframe': timeframe,
                                'signal': timeframe_signal,
                                'weight': self.optimizer.timeframe_weights.get(timeframe, 1.0),
                                'role': self.optimizer.timeframe_categories.get(timeframe),
                                'confidence_contribution': timeframe_signal['confidence'] / 100
                            })
                            
                            self.logger.debug(f"   ✅ {timeframe}: {symbol} {timeframe_signal['side'].upper()} signal confirmed")
                        else:
                            conflicting_timeframes.append(timeframe)
                            self.logger.debug(f"   ⚠️ {timeframe}: {symbol} {timeframe_signal['side'].upper()} signal conflicts")
                    else:
                        neutral_timeframes.append(timeframe)
                        self.logger.debug(f"   ➖ {timeframe}: {symbol} No clear signal")
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"   Error analyzing {timeframe}: {e}")
                    neutral_timeframes.append(timeframe)
            
            # LEVEL OPTIMIZATION using aligned signals
            optimization_result = self._perform_level_optimization(
                aligned_signals, symbol_data['current_price'], primary_side
            )
            
            # Apply optimization to primary signal if successful
            if optimization_result.get('optimization_applied'):
                self._apply_optimization_to_primary_signal(primary_signal, optimization_result)
                self.logger.debug(f"   📊 Levels optimized using {len(aligned_signals)} timeframes")
            
            # Calculate MTF metrics
            total_timeframes = len(self.config.confirmation_timeframes)
            confirmed_count = len(confirmed_timeframes)
            conflicting_count = len(conflicting_timeframes)
            
            confirmation_strength = confirmed_count / total_timeframes if total_timeframes > 0 else 0
            
            # Calculate confidence boost
            mtf_confidence_boost = 0
            if confirmed_count > 0:
                net_confirmation = confirmed_count - conflicting_count
                mtf_confidence_boost = max(0, 
                    (net_confirmation / total_timeframes) * (self.config.mtf_weight_multiplier - 1) * 100
                )
                
                # Bonus for level optimization
                if optimization_result.get('optimization_applied'):
                    mtf_confidence_boost += 3
            
            # Compile comprehensive result
            result = {
                'confirmed_timeframes': confirmed_timeframes,
                'conflicting_timeframes': conflicting_timeframes,
                'neutral_timeframes': neutral_timeframes,
                'confirmation_strength': confirmation_strength,
                'mtf_confidence_boost': mtf_confidence_boost,
                'timeframe_signals': timeframe_signals,
                'level_optimization_applied': optimization_result.get('optimization_applied', False),
                'optimization_details': optimization_result if optimization_result.get('optimization_applied') else None,
                'timeframe_configuration': self.optimizer.get_configuration_summary()
            }
            
            self.logger.debug(f"   📊 Confirmation: {confirmed_count}/{total_timeframes} timeframes")
            self.logger.debug(f"   💪 Strength: {confirmation_strength:.1%}")
            if optimization_result.get('optimization_applied'):
                self.logger.debug(f"   🎯 Level optimization: SUCCESS")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced MTF analysis error for {symbol}: {e}")
            return self._get_fallback_mtf_result()
    
    def _get_timeframe_signal(self, symbol_data: Dict, timeframe: str) -> Optional[Dict]:
        """Get signal for specific timeframe using existing analysis logic"""
        try:
            symbol = symbol_data['symbol']
            
            # Fetch data for this timeframe
            df = self.exchange_manager.fetch_ohlcv_data(symbol, timeframe)
            if df.empty or len(df) < 50:
                self.logger.debug(f"Insufficient {timeframe} data for {symbol}")
                return None
            
            # Apply full technical analysis
            df = self.enhanced_ta.calculate_all_indicators(df, self.config)
            
            # Volume profile analysis
            volume_profile = self.volume_analyzer.calculate_volume_profile(df)
            volume_entry = self.volume_analyzer.find_optimal_entry_from_volume(
                df, symbol_data['current_price'], 'buy'
            )
            
            # Fibonacci & confluence analysis
            fibonacci_data = self.fibonacci_analyzer.calculate_fibonacci_levels(df)
            confluence_zones = self.fibonacci_analyzer.find_confluence_zones(
                df, volume_profile, symbol_data['current_price']
            )
            
            # Generate comprehensive signal
            timeframe_signal = self.signal_generator.analyze_symbol_comprehensive(
                df, symbol_data, volume_entry, fibonacci_data, confluence_zones, timeframe
            )
            
            return timeframe_signal
            
        except Exception as e:
            self.logger.error(f"Error getting {timeframe} signal for {symbol_data.get('symbol', 'unknown')}: {e}")
            return None
    
    def _perform_level_optimization(self, aligned_signals: List[Dict], 
                                  current_price: float, side: str) -> Dict:
        """Perform level optimization using the configurable optimizer"""
        try:
            if len(aligned_signals) < 2:  # Need at least primary + 1 confirmation
                return {
                    'optimization_applied': False,
                    'reason': 'insufficient_signals',
                    'signal_count': len(aligned_signals)
                }
            
            # Extract level data
            entry_levels = []
            stop_levels = []
            tp1_levels = []
            tp2_levels = []
            
            for signal_data in aligned_signals:
                signal = signal_data['signal']
                if not signal or not all(key in signal for key in ['entry_price', 'stop_loss', 'take_profit_1']):
                    continue
                
                weight = signal_data['weight']
                tf = signal_data['timeframe']
                confidence = signal_data['confidence_contribution']
                role = signal_data['role']
                
                entry_levels.append({
                    'price': signal['entry_price'],
                    'weight': weight,
                    'timeframe': tf,
                    'confidence': confidence,
                    'role': role
                })
                
                stop_levels.append({
                    'price': signal['stop_loss'],
                    'weight': weight,
                    'timeframe': tf,
                    'role': role
                })
                
                tp1_levels.append({
                    'price': signal['take_profit_1'],
                    'weight': weight,
                    'timeframe': tf,
                    'role': role
                })
                
                if 'take_profit_2' in signal:
                    tp2_levels.append({
                        'price': signal['take_profit_2'],
                        'weight': weight,
                        'timeframe': tf,
                        'role': role
                    })
            
            if len(entry_levels) < 2:
                return {
                    'optimization_applied': False,
                    'reason': 'insufficient_valid_levels',
                    'valid_levels': len(entry_levels)
                }
            
            # Calculate optimized levels using adaptive method
            optimized_entry = self._calculate_weighted_level(entry_levels, current_price, 'entry')
            optimized_stop = self._calculate_weighted_level(stop_levels, optimized_entry, 'stop', side)
            optimized_tp1 = self._calculate_weighted_level(tp1_levels, optimized_entry, 'tp1', side)
            optimized_tp2 = self._calculate_weighted_level(tp2_levels, optimized_entry, 'tp2', side) if tp2_levels else None
            
            # Calculate risk/reward
            if side == 'buy':
                risk = optimized_entry - optimized_stop
                reward = optimized_tp1 - optimized_entry if self.config.default_tp_level == 'take_profit_1' else optimized_tp2 - optimized_entry
            else:
                risk = optimized_stop - optimized_entry
                reward = optimized_entry - optimized_tp1 if self.config.default_tp_level == 'take_profit_1' else optimized_entry - optimized_tp2
                
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Validate levels
            if not self._validate_levels(optimized_entry, optimized_stop, optimized_tp1, side, risk_reward_ratio):
                return {
                    'optimization_applied': False,
                    'reason': 'validation_failed',
                    'attempted_levels': {
                        'entry': optimized_entry,
                        'stop': optimized_stop,
                        'tp1': optimized_tp1,
                        'rr': risk_reward_ratio
                    }
                }
            
            return {
                'optimization_applied': True,
                'optimized_levels': {
                    'entry_price': optimized_entry,
                    'stop_loss': optimized_stop,
                    'take_profit_1': optimized_tp1,
                    'take_profit_2': optimized_tp2,
                    'risk_reward_ratio': risk_reward_ratio
                },
                'timeframes_used': [s['timeframe'] for s in aligned_signals],
                'total_weight': sum(s['weight'] for s in aligned_signals),
                'optimization_method': 'adaptive_weighted'
            }
            
        except Exception as e:
            self.logger.error(f"Level optimization error: {e}")
            return {
                'optimization_applied': False,
                'reason': 'optimization_error',
                'error': str(e)
            }
    
    def _calculate_weighted_level(self, levels: List[Dict], reference_price: float, 
                                level_type: str, side: str = None) -> float:
        """Calculate weighted level with role-based adjustments"""
        if not levels:
            if level_type == 'entry':
                return reference_price
            elif level_type == 'stop':
                return reference_price * (0.98 if side == 'buy' else 1.02)
            else:  # tp1, tp2
                return reference_price * (1.04 if side == 'buy' else 0.96)
        
        total_weighted_price = 0
        total_weight = 0
        
        for level in levels:
            weight = level['weight']
            
            # Role-based weight adjustments
            if level.get('role') == TimeframeRole.HIGHER_CONFIRMATION:
                if level_type == 'stop':
                    weight *= 2.0  # Higher TF stops are much more important
                elif level_type in ['tp1', 'tp2']:
                    weight *= 1.5  # Higher TF targets are more reliable
                else:  # entry
                    weight *= 1.2
            elif level.get('role') == TimeframeRole.LOWER_TIMING:
                if level_type == 'entry':
                    weight *= 1.1  # Lower TF good for entry timing
                else:
                    weight *= 0.7  # But less important for stops/targets
            
            # Confidence weighting
            confidence = level.get('confidence', 1.0)
            weight *= confidence
            
            # Distance-based adjustment for stops
            if level_type == 'stop' and side:
                distance = abs(level['price'] - reference_price) / reference_price
                optimal_distance = 0.025  # 2.5% ideal
                distance_score = 1 / (1 + abs(distance - optimal_distance) / optimal_distance)
                weight *= distance_score
            
            total_weighted_price += level['price'] * weight
            total_weight += weight
        
        return total_weighted_price / total_weight if total_weight > 0 else reference_price
    
    def _validate_levels(self, entry: float, stop: float, tp1: float, side: str, rr: float) -> bool:
        """Validate optimized levels"""
        try:
            # Direction validation
            if side == 'buy':
                if not (stop < entry < tp1):
                    return False
            else:
                if not (tp1 < entry < stop):
                    return False
            
            # Risk/reward validation
            if rr < 1.2:
                return False
            
            # Risk percentage validation
            risk_pct = abs(entry - stop) / entry
            if risk_pct < 0.005 or risk_pct > 0.15:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_optimization_to_primary_signal(self, primary_signal: Dict, optimization_result: Dict):
        """Apply optimization results to primary signal"""
        if not optimization_result.get('optimization_applied'):
            return
        
        optimized_levels = optimization_result['optimized_levels']
        
        # Store original levels
        original_levels = {
            'entry_price': primary_signal.get('entry_price'),
            'stop_loss': primary_signal.get('stop_loss'),
            'take_profit_1': primary_signal.get('take_profit_1'),
            'take_profit_2': primary_signal.get('take_profit_2'),
            'risk_reward_ratio': primary_signal.get('risk_reward_ratio', 0)
        }
        
        # Update signal
        primary_signal.update(optimized_levels)
        
        # Add metadata
        primary_signal['level_optimization'] = {
            'enabled': True,
            'method': optimization_result['optimization_method'],
            'timeframes_used': optimization_result['timeframes_used'],
            'total_weight': optimization_result['total_weight'],
            'original_levels': original_levels,
            'improvements': {
                'rr_improvement': optimized_levels['risk_reward_ratio'] - original_levels['risk_reward_ratio'],
                'entry_change_pct': abs(optimized_levels['entry_price'] - original_levels['entry_price']) / original_levels['entry_price'] * 100 if original_levels['entry_price'] else 0
            }
        }
    
    def _get_fallback_mtf_result(self) -> Dict:
        """Fallback MTF result"""
        return {
            'confirmed_timeframes': [],
            'conflicting_timeframes': [],
            'neutral_timeframes': self.config.confirmation_timeframes.copy(),
            'confirmation_strength': 0.0,
            'mtf_confidence_boost': 0.0,
            'timeframe_signals': {},
            'level_optimization_applied': False,
            'optimization_details': None,
            'timeframe_configuration': self.optimizer.get_configuration_summary()
        }


# Example usage and configuration
def example_usage():
    """Example of how to use the configurable MTF optimizer"""
    
    # Your current configuration works perfectly!
    config = {
        'timeframe': '1h',                    # Primary
        'confirmation_timeframes': ['2h', '4h', '6h'],  # Confirmations
        'mtf_confirmation_required': True,
        'mtf_weight_multiplier': 1.3
    }
    
    print("🎯 CONFIGURABLE MTF OPTIMIZER EXAMPLE")
    print(f"Primary: {config['timeframe']}")
    print(f"Confirmations: {config['confirmation_timeframes']}")
    
    # The system automatically detects:
    print("\nAutomatic Detection:")
    print("✅ 2h, 4h, 6h = HIGHER_CONFIRMATION (trend validation)")
    print("✅ Proper hierarchy: Higher TFs confirm lower TF")
    print("✅ Adaptive weights: 6h gets highest weight")
    print("✅ Level optimization: Uses all aligned timeframes")
    
    # If you change configuration later:
    alternative_configs = [
        {
            'name': 'Scalping Setup',
            'timeframe': '15m',
            'confirmation_timeframes': ['30m', '1h', '2h'],
            'note': 'Lower TF primary with higher TF confirmation'
        },
        {
            'name': 'Swing Trading Setup', 
            'timeframe': '4h',
            'confirmation_timeframes': ['6h', '12h', '1d'],
            'note': 'Higher TF primary with even higher TF confirmation'
        },
        {
            'name': 'Mixed Strategy',
            'timeframe': '1h',
            'confirmation_timeframes': ['30m', '2h', '4h'],
            'note': '30m for timing, 2h/4h for confirmation'
        }
    ]
    
    print("\n🔧 SYSTEM ADAPTS TO ANY CONFIGURATION:")
    for config in alternative_configs:
        print(f"\n{config['name']}:")
        print(f"  Primary: {config['timeframe']}")
        print(f"  Confirmations: {config['confirmation_timeframes']}")
        print(f"  Logic: {config['note']}")
        print("  ✅ System auto-adapts weights and roles")


if __name__ == "__main__":
    example_usage()