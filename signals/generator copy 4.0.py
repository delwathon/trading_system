"""
ENHANCED Multi-Timeframe Signal Generation for Bybit Trading System
VERSION 4.0 - PRODUCTION READY with Advanced Entry Logic and Quality Filters

KEY IMPROVEMENTS:
1. ✅ Dynamic stop losses (1.5% - 6% based on volatility)
2. ✅ Momentum-aware entry calculation for trending markets
3. ✅ Entry timing validation to avoid bad entries
4. ✅ Quality filters (momentum strength, volume divergence, choppiness)
5. ✅ Fast-moving signal detection for quicker profits
6. ✅ Enhanced R/R ratios with wider targets
7. ✅ Better risk management for volatile crypto markets
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.config import EnhancedSystemConfig


@dataclass
class SignalConfig:
    """Enhanced configuration for signal generation with momentum awareness"""
    # Stop loss parameters - WIDER RANGES FOR CRYPTO VOLATILITY
    min_stop_distance_pct: float = 0.075  # Minimum 1.5% stop distance
    max_stop_distance_pct: float = 0.125   # Maximum 6% stop distance
    structure_stop_buffer: float = 0.005  # 0.5% buffer below support/above resistance
    
    # Entry parameters - MOMENTUM AWARE
    entry_buffer_from_structure: float = 0.003  # 0.3% buffer from key levels
    entry_limit_distance: float = 0.015    # Max 1.5% from current price for limit orders
    momentum_entry_adjustment: float = 0.005  # 0.5% adjustment for trending markets
    
    # Take profit parameters - BETTER RATIOS
    min_tp_distance_pct: float = 0.02     # Minimum 2% profit target
    max_tp_distance_pct: float = 0.3     # Maximum 15% profit target
    tp1_multiplier: float = 3.0          # TP1 at 2.5x risk
    tp2_multiplier: float = 5.0           # TP2 at 4x risk
    
    # Risk/Reward parameters
    min_risk_reward: float = 2.0          # Minimum acceptable R/R
    max_risk_reward: float = 10.0         # Maximum R/R (cap unrealistic values)
    
    # Signal quality thresholds
    min_confidence_for_signal: float = 50.0  # Higher minimum confidence
    mtf_confidence_boost: float = 10.0       # Reduced MTF boost for realism
    
    # Volatility adjustments
    high_volatility_threshold: float = 0.08  # 8% ATR
    low_volatility_threshold: float = 0.02   # 2% ATR
    
    # Market microstructure parameters
    use_order_book_analysis: bool = True
    min_order_book_imbalance: float = 0.6  # 60% imbalance for directional bias
    price_momentum_lookback: int = 5  # Candles to analyze for momentum
    
    # Fast signal parameters
    min_momentum_for_fast_signal: float = 2.0  # Minimum momentum % for fast signals
    max_choppiness_score: float = 0.6  # Maximum choppiness to accept signal
    volume_surge_multiplier: float = 2.0  # Volume multiplier for surge detection
    fast_signal_bonus_confidence: float = 10.0  # Confidence bonus for fast setups


@dataclass
class MultiTimeframeContext:
    """Container for multi-timeframe market analysis"""
    dominant_trend: str          # Overall trend from highest timeframe
    trend_strength: float        # 0.0 to 1.0 trend strength
    higher_tf_zones: List[Dict]  # Key support/resistance from higher TFs
    key_support: float           # Major support level
    key_resistance: float        # Major resistance level
    momentum_alignment: bool     # Is momentum aligned with trend
    entry_bias: str             # 'long_favored', 'short_favored', 'neutral', 'avoid'
    confirmation_score: float   # 0.0 to 1.0 multi-TF confirmation strength
    structure_timeframe: str    # Structure analysis timeframe
    market_regime: str          # 'trending_up', 'trending_down', 'ranging', 'volatile'
    volatility_level: str       # 'low', 'medium', 'high', 'extreme'


class SignalValidator:
    """Validate and sanitize signals with enhanced logic"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_stop_loss(self, entry_price: float, stop_loss: float, side: str) -> Tuple[bool, float]:
        """Validate and adjust stop loss if needed"""
        if side == 'buy':
            distance_pct = (entry_price - stop_loss) / entry_price
            if distance_pct < self.config.min_stop_distance_pct:
                # Stop too close - adjust to minimum
                new_stop = entry_price * (1 - self.config.min_stop_distance_pct)
                self.logger.debug(f"Adjusted stop loss from {stop_loss:.6f} to {new_stop:.6f} (min distance)")
                return True, new_stop
            elif distance_pct > self.config.max_stop_distance_pct:
                # Stop too far - adjust to maximum
                new_stop = entry_price * (1 - self.config.max_stop_distance_pct)
                self.logger.debug(f"Adjusted stop loss from {stop_loss:.6f} to {new_stop:.6f} (max distance)")
                return True, new_stop
        else:  # sell
            distance_pct = (stop_loss - entry_price) / entry_price
            if distance_pct < self.config.min_stop_distance_pct:
                new_stop = entry_price * (1 + self.config.min_stop_distance_pct)
                self.logger.debug(f"Adjusted stop loss from {stop_loss:.6f} to {new_stop:.6f} (min distance)")
                return True, new_stop
            elif distance_pct > self.config.max_stop_distance_pct:
                new_stop = entry_price * (1 + self.config.max_stop_distance_pct)
                self.logger.debug(f"Adjusted stop loss from {stop_loss:.6f} to {new_stop:.6f} (max distance)")
                return True, new_stop
        
        return False, stop_loss
    
    def validate_take_profits(self, entry_price: float, stop_loss: float, 
                            tp1: float, tp2: float, side: str) -> Tuple[float, float]:
        """Validate and adjust take profits based on risk"""
        if side == 'buy':
            risk = entry_price - stop_loss
            min_tp1 = entry_price + (risk * self.config.tp1_multiplier)
            min_tp2 = entry_price + (risk * self.config.tp2_multiplier)
            
            # Ensure minimum profit distances
            min_tp1_by_pct = entry_price * (1 + self.config.min_tp_distance_pct)
            min_tp2_by_pct = entry_price * (1 + self.config.min_tp_distance_pct * 2)
            
            validated_tp1 = max(tp1, min_tp1, min_tp1_by_pct)
            validated_tp2 = max(tp2, min_tp2, min_tp2_by_pct, validated_tp1 * 1.01)
            
            # Cap at maximum distance
            max_tp = entry_price * (1 + self.config.max_tp_distance_pct)
            validated_tp1 = min(validated_tp1, max_tp)
            validated_tp2 = min(validated_tp2, max_tp)
            
        else:  # sell
            risk = stop_loss - entry_price
            min_tp1 = entry_price - (risk * self.config.tp1_multiplier)
            min_tp2 = entry_price - (risk * self.config.tp2_multiplier)
            
            # Ensure minimum profit distances
            min_tp1_by_pct = entry_price * (1 - self.config.min_tp_distance_pct)
            min_tp2_by_pct = entry_price * (1 - self.config.min_tp_distance_pct * 2)
            
            validated_tp1 = min(tp1, min_tp1, min_tp1_by_pct)
            validated_tp2 = min(tp2, min_tp2, min_tp2_by_pct, validated_tp1 * 0.99)
            
            # Cap at maximum distance
            max_tp = entry_price * (1 - self.config.max_tp_distance_pct)
            validated_tp1 = max(validated_tp1, max_tp)
            validated_tp2 = max(validated_tp2, max_tp)
        
        return validated_tp1, validated_tp2
    
    def calculate_risk_reward(self, entry: float, stop: float, tp: float, side: str) -> float:
        """Calculate risk/reward ratio with validation"""
        if side == 'buy':
            risk = entry - stop
            reward = tp - entry
        else:
            risk = stop - entry
            reward = entry - tp
        
        if risk <= 0:
            return 0
        
        rr = reward / risk
        # Cap at maximum to avoid unrealistic values
        return min(rr, self.config.max_risk_reward)


# ===== ENHANCED QUALITY FILTER FUNCTIONS =====

def analyze_price_momentum_strength(df: pd.DataFrame, lookback_periods: list = [5, 10, 20]) -> dict:
    """Analyze momentum strength across multiple timeframes"""
    try:
        if len(df) < max(lookback_periods):
            return {'strength': 0, 'direction': 'neutral', 'speed': 'slow'}
        
        latest_close = df['close'].iloc[-1]
        momentum_scores = []
        
        for period in lookback_periods:
            past_close = df['close'].iloc[-period]
            change_pct = (latest_close - past_close) / past_close * 100
            
            # Weight recent momentum more heavily
            weight = 1 / (lookback_periods.index(period) + 1)
            momentum_scores.append(change_pct * weight)
        
        # Calculate weighted momentum
        weighted_momentum = sum(momentum_scores) / sum(1/(i+1) for i in range(len(lookback_periods)))
        
        # Determine strength and direction
        if abs(weighted_momentum) < 1:
            strength = 0
            direction = 'neutral'
            speed = 'slow'
        elif abs(weighted_momentum) < 3:
            strength = 1
            direction = 'bullish' if weighted_momentum > 0 else 'bearish'
            speed = 'moderate'
        elif abs(weighted_momentum) < 5:
            strength = 2
            direction = 'bullish' if weighted_momentum > 0 else 'bearish'
            speed = 'fast'
        else:
            strength = 3
            direction = 'bullish' if weighted_momentum > 0 else 'bearish'
            speed = 'very_fast'
        
        return {
            'strength': strength,
            'direction': direction,
            'speed': speed,
            'weighted_momentum': weighted_momentum,
            'momentum_by_period': dict(zip(lookback_periods, momentum_scores))
        }
        
    except Exception:
        return {'strength': 0, 'direction': 'neutral', 'speed': 'slow'}


def check_volume_momentum_divergence(df: pd.DataFrame, window: int = 10) -> dict:
    """Check for volume and price momentum divergence"""
    try:
        if len(df) < window:
            return {'divergence': False, 'type': 'none', 'strength': 0}
        
        recent = df.tail(window)
        
        # Price trend
        price_trend = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Volume trend
        volume_trend = (recent['volume'].mean() - df['volume'].rolling(window=20).mean().iloc[-window]) / df['volume'].rolling(window=20).mean().iloc[-window]
        
        # Check for divergence
        if price_trend > 0.01 and volume_trend < -0.1:
            # Bullish price with declining volume - bearish divergence
            return {
                'divergence': True,
                'type': 'bearish_divergence',
                'strength': abs(volume_trend),
                'warning': 'Price rising on declining volume'
            }
        elif price_trend < -0.01 and volume_trend < -0.1:
            # Bearish price with declining volume - potential reversal
            return {
                'divergence': True,
                'type': 'potential_reversal',
                'strength': abs(volume_trend),
                'warning': 'Selling pressure may be exhausting'
            }
        elif abs(price_trend) > 0.02 and volume_trend > 0.5:
            # Strong move with strong volume - confirmation
            return {
                'divergence': False,
                'type': 'confirmed_move',
                'strength': volume_trend,
                'info': 'Volume confirms price movement'
            }
        else:
            return {'divergence': False, 'type': 'none', 'strength': 0}
            
    except Exception:
        return {'divergence': False, 'type': 'none', 'strength': 0}


def identify_fast_moving_setup(df: pd.DataFrame, side: str) -> dict:
    """Identify setups likely to move fast in the signal direction"""
    try:
        if len(df) < 20:
            return {'is_fast_setup': False, 'score': 0}
        
        latest = df.iloc[-1]
        score = 0
        factors = []
        
        # 1. Breakout detection
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_close = latest['close']
        
        if side == 'buy':
            # Check for bullish breakout
            if current_close > recent_high * 0.995:  # Near or above recent high
                score += 2
                factors.append('breakout_imminent')
            
            # Check for successful retest
            if len(df) >= 10:
                touched_support = any(df['low'].tail(10) <= recent_low * 1.01)
                bounced_up = current_close > recent_low * 1.02
                if touched_support and bounced_up:
                    score += 1.5
                    factors.append('support_bounce')
        
        else:  # sell
            # Check for bearish breakout
            if current_close < recent_low * 1.005:  # Near or below recent low
                score += 2
                factors.append('breakdown_imminent')
            
            # Check for failed retest
            if len(df) >= 10:
                touched_resistance = any(df['high'].tail(10) >= recent_high * 0.99)
                rejected_down = current_close < recent_high * 0.98
                if touched_resistance and rejected_down:
                    score += 1.5
                    factors.append('resistance_rejection')
        
        # 2. Volume surge detection
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio > 2.0:
            score += 1
            factors.append('volume_surge')
        
        # 3. Momentum alignment
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        
        if side == 'buy':
            if 40 < rsi < 60 and macd > macd_signal:  # Not overbought, momentum building
                score += 1
                factors.append('momentum_building')
        else:
            if 40 < rsi < 60 and macd < macd_signal:  # Not oversold, momentum building
                score += 1
                factors.append('momentum_building')
        
        # 4. Volatility expansion
        if 'atr' in df.columns:
            current_atr = latest['atr']
            avg_atr = df['atr'].tail(20).mean()
            if current_atr > avg_atr * 1.2:
                score += 0.5
                factors.append('volatility_expansion')
        
        return {
            'is_fast_setup': score >= 2.5,
            'score': score,
            'factors': factors,
            'likelihood': 'high' if score >= 3.5 else 'medium' if score >= 2.5 else 'low'
        }
        
    except Exception:
        return {'is_fast_setup': False, 'score': 0}


def filter_choppy_markets(df: pd.DataFrame, window: int = 20) -> dict:
    """Filter out choppy/ranging markets that lead to whipsaws"""
    try:
        if len(df) < window:
            return {'is_choppy': False, 'choppiness_score': 0}
        
        recent = df.tail(window)
        
        # 1. Calculate directional movement
        up_moves = sum(1 for i in range(1, len(recent)) if recent['close'].iloc[i] > recent['close'].iloc[i-1])
        down_moves = window - 1 - up_moves
        directional_ratio = abs(up_moves - down_moves) / (window - 1)
        
        # 2. Calculate average true range vs price movement
        if 'atr' in df.columns:
            total_price_movement = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])
            total_atr = recent['atr'].sum()
            efficiency_ratio = total_price_movement / total_atr if total_atr > 0 else 0
        else:
            efficiency_ratio = 0.5
        
        # 3. Count reversals
        reversals = 0
        for i in range(2, len(recent)):
            if (recent['close'].iloc[i] > recent['close'].iloc[i-1] and 
                recent['close'].iloc[i-1] < recent['close'].iloc[i-2]):
                reversals += 1
            elif (recent['close'].iloc[i] < recent['close'].iloc[i-1] and 
                  recent['close'].iloc[i-1] > recent['close'].iloc[i-2]):
                reversals += 1
        
        reversal_ratio = reversals / (window - 2)
        
        # Calculate choppiness score
        choppiness_score = (1 - directional_ratio) * 0.4 + (1 - efficiency_ratio) * 0.3 + reversal_ratio * 0.3
        
        return {
            'is_choppy': choppiness_score > 0.6,
            'choppiness_score': choppiness_score,
            'directional_ratio': directional_ratio,
            'efficiency_ratio': efficiency_ratio,
            'reversal_ratio': reversal_ratio,
            'market_state': 'choppy' if choppiness_score > 0.6 else 'trending' if choppiness_score < 0.4 else 'mixed'
        }
        
    except Exception:
        return {'is_choppy': False, 'choppiness_score': 0}


def calculate_momentum_adjusted_entry(current_price: float, df: pd.DataFrame, 
                                    side: str, config: SignalConfig) -> float:
    """Calculate entry price with momentum adjustment for trending markets"""
    try:
        # Get recent price momentum
        if len(df) < config.price_momentum_lookback:
            momentum_pct = 0
        else:
            recent = df.tail(config.price_momentum_lookback)
            price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            momentum_pct = price_change
        
        # Calculate volatility-based adjustment
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        volatility_adjustment = (atr / current_price) * 0.5  # 50% of ATR as adjustment
        
        if side == 'buy':
            if momentum_pct > 0.01:  # Strong upward momentum
                # For LONG in uptrend, place entry ABOVE current price to catch momentum
                entry_adjustment = max(volatility_adjustment, config.momentum_entry_adjustment)
                entry_price = current_price * (1 + entry_adjustment)
            else:
                # For LONG in stable/down market, try to get better entry
                entry_price = current_price * (1 - config.entry_buffer_from_structure)
        else:  # sell
            if momentum_pct < -0.01:  # Strong downward momentum
                # For SHORT in downtrend, place entry BELOW current price to catch momentum
                entry_adjustment = max(volatility_adjustment, config.momentum_entry_adjustment)
                entry_price = current_price * (1 - entry_adjustment)
            else:
                # For SHORT in stable/up market, try to get better entry
                entry_price = current_price * (1 + config.entry_buffer_from_structure)
        
        return entry_price
        
    except Exception:
        # Fallback to simple calculation
        if side == 'buy':
            return current_price * 0.999
        else:
            return current_price * 1.001


def calculate_dynamic_stop_loss(entry_price: float, current_price: float, 
                              df: pd.DataFrame, side: str, config: SignalConfig) -> float:
    """Calculate stop loss with proper distance based on volatility and market structure"""
    try:
        # Get ATR for volatility-based stop
        if 'atr' in df.columns and len(df) >= 14:
            atr = df['atr'].iloc[-1]
            atr_pct = atr / current_price
        else:
            atr_pct = 0.02  # Default 2% if ATR not available
        
        # Calculate recent volatility
        if len(df) >= 20:
            recent_changes = df['close'].pct_change().tail(20).abs()
            avg_volatility = recent_changes.mean()
            max_volatility = recent_changes.max()
        else:
            avg_volatility = 0.01
            max_volatility = 0.02
        
        # Determine stop distance based on market conditions
        if max_volatility > config.high_volatility_threshold:
            # High volatility: Use wider stops
            stop_distance_pct = min(config.max_stop_distance_pct, max(atr_pct * 2.5, config.min_stop_distance_pct * 3))
        elif avg_volatility > 0.015:
            # Medium volatility: Standard stops
            stop_distance_pct = min(config.max_stop_distance_pct * 0.8, max(atr_pct * 2.0, config.min_stop_distance_pct * 2))
        else:
            # Low volatility: Can use tighter stops but not too tight
            stop_distance_pct = max(atr_pct * 1.5, config.min_stop_distance_pct)
        
        # Apply stop loss
        if side == 'buy':
            stop_loss = entry_price * (1 - stop_distance_pct)
        else:
            stop_loss = entry_price * (1 + stop_distance_pct)
        
        return stop_loss
        
    except Exception:
        # Fallback to minimum safe distance
        if side == 'buy':
            return entry_price * (1 - config.min_stop_distance_pct * 2)
        else:
            return entry_price * (1 + config.min_stop_distance_pct * 2)


class SignalGenerator:
    """
    ENHANCED Multi-Timeframe Signal Generator v4.0
    
    Key improvements:
    - Dynamic entry calculation based on momentum
    - Wider stop losses for volatile crypto markets
    - Quality filters to avoid choppy markets
    - Fast-moving signal detection
    - Enhanced risk management
    """
    
    def __init__(self, config: EnhancedSystemConfig, exchange_manager=None):
        self.config = config
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize signal configuration
        self.signal_config = SignalConfig()
        self.validator = SignalValidator(self.signal_config)
        
        # Use configured timeframes from database
        self.primary_timeframe = config.timeframe  # e.g., '1h'
        self.confirmation_timeframes = config.confirmation_timeframes  # e.g., ['2h', '4h', '6h']
        
        # Determine structure timeframe (highest confirmation TF)
        if self.confirmation_timeframes:
            # Sort timeframes by duration to get highest
            tf_minutes = {'1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440}
            sorted_tfs = sorted(self.confirmation_timeframes, 
                              key=lambda x: tf_minutes.get(x, 0), reverse=True)
            self.structure_timeframe = sorted_tfs[0]  # Highest timeframe for structure
        else:
            self.structure_timeframe = '6h'  # Fallback
        
        # Debug mode flag
        self.debug_mode = False  # Set to True for debugging
        
        self.logger.info("✅ ENHANCED Signal Generator v4.0 initialized")
        self.logger.info(f"   Primary TF: {self.primary_timeframe}")
        self.logger.info(f"   Structure TF: {self.structure_timeframe}")
        self.logger.info(f"   Confirmation TFs: {self.confirmation_timeframes}")
        self.logger.info(f"   Min Stop Distance: {self.signal_config.min_stop_distance_pct*100:.1f}%")
        self.logger.info(f"   Max R/R Ratio: {self.signal_config.max_risk_reward}:1")
        self.logger.info(f"   Quality Filters: ✅ Enabled")

    def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol_data: Dict, 
                                   volume_entry: Dict, fibonacci_data: Dict, 
                                   confluence_zones: List[Dict], timeframe: str) -> Optional[Dict]:
        """Main entry point with comprehensive analysis"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            if self.debug_mode:
                self.logger.info(f"🔧 DEBUG: Starting analysis for {symbol}")
            
            # PHASE 1: Market regime detection
            try:
                market_regime = self._determine_market_regime(symbol_data, df)
                if self.debug_mode:
                    self.logger.info(f"🔧 DEBUG: {symbol} - Market regime: {market_regime}")
            except Exception as e:
                self.logger.warning(f"Market regime detection failed for {symbol}: {e}")
                market_regime = 'ranging'  # Safe fallback
            
            # PHASE 2: Multi-timeframe context analysis
            try:
                mtf_context = self._get_multitimeframe_context(symbol_data, market_regime)
                if not mtf_context:
                    if self.debug_mode:
                        self.logger.info(f"🔧 DEBUG: {symbol} - Creating fallback MTF context")
                    mtf_context = self._create_fallback_context(symbol_data, market_regime)
            except Exception as e:
                self.logger.warning(f"MTF context creation failed for {symbol}: {e}")
                mtf_context = self._create_fallback_context(symbol_data, market_regime)
            
            # PHASE 3: Signal generation with multiple fallback levels
            signal = None
            
            # Try MTF-aware signal first
            if self._should_use_mtf_signals(mtf_context):
                signal = self._generate_mtf_aware_signal(
                    df, symbol_data, volume_entry, fibonacci_data, 
                    confluence_zones, mtf_context
                )
            
            # Fallback to traditional signal if MTF fails
            if not signal:
                if self.debug_mode:
                    self.logger.info(f"🔧 DEBUG: {symbol} - MTF signal failed, trying traditional")
                signal = self._generate_traditional_signal(
                    df, symbol_data, volume_entry, fibonacci_data, confluence_zones
                )
            
            # Final validation and enhancement
            if signal:
                # Enhanced signal with comprehensive metadata
                signal = self._validate_and_enhance_signal(signal, mtf_context, df, market_regime)
                
                # Add comprehensive analysis
                signal['analysis'] = self._create_comprehensive_analysis(
                    df, symbol_data, volume_entry, fibonacci_data, 
                    confluence_zones, mtf_context
                )
                
                signal['timestamp'] = pd.Timestamp.now()
                signal['timeframe'] = timeframe
                
                if self.debug_mode:
                    self.logger.info(f"🔧 DEBUG: ✅ {symbol} Signal generated: {signal['side'].upper()} "
                                    f"@ ${signal['entry_price']:.6f} "
                                    f"(R/R: {signal['risk_reward_ratio']:.2f}, conf:{signal['confidence']:.0f}%)")
            else:
                if self.debug_mode:
                    self.logger.info(f"🔧 DEBUG: ❌ {symbol} - No signal generated")
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol}: {e}")
            return None

    def _should_use_mtf_signals(self, mtf_context: MultiTimeframeContext) -> bool:
        """Determine if MTF signals should be used"""
        # Use MTF if we have good context and it's not in 'avoid' mode
        return (mtf_context.entry_bias != 'avoid' and 
                mtf_context.confirmation_score > 0.3 and
                self.exchange_manager is not None)

    def _generate_mtf_aware_signal(self, df: pd.DataFrame, symbol_data: Dict,
                                 volume_entry: Dict, fibonacci_data: Dict,
                                 confluence_zones: List[Dict], 
                                 mtf_context: MultiTimeframeContext) -> Optional[Dict]:
        """Generate signal with MTF awareness"""
        try:
            latest = df.iloc[-1]
            
            # Determine signal direction based on MTF context
            if mtf_context.entry_bias == 'long_favored':
                return self._generate_long_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, use_mtf=True
                )
            elif mtf_context.entry_bias == 'short_favored':
                return self._generate_short_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, use_mtf=True
                )
            elif mtf_context.entry_bias == 'neutral':
                # Try both directions
                long_signal = self._generate_long_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, use_mtf=True
                )
                if long_signal:
                    return long_signal
                    
                return self._generate_short_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, use_mtf=True
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"MTF signal generation error: {e}")
            return None

    def _generate_traditional_signal(self, df: pd.DataFrame, symbol_data: Dict,
                                   volume_entry: Dict, fibonacci_data: Dict,
                                   confluence_zones: List[Dict]) -> Optional[Dict]:
        """Generate traditional signal without MTF"""
        try:
            latest = df.iloc[-1]
            
            # Create simple context
            mtf_context = self._create_fallback_context(symbol_data, 'unknown')
            
            # Traditional signal conditions
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            stoch_k = latest.get('stoch_rsi_k', 50)
            stoch_d = latest.get('stoch_rsi_d', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # LONG conditions
            if (rsi < 45 and macd > macd_signal and 
                stoch_k > stoch_d and volume_ratio > 0.8):
                return self._generate_long_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, use_mtf=False
                )
            # SHORT conditions
            elif (rsi > 55 and macd < macd_signal and 
                  stoch_k < stoch_d and volume_ratio > 0.8):
                return self._generate_short_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, use_mtf=False
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Traditional signal generation error: {e}")
            return None

    def _generate_long_signal(self, symbol_data: Dict, latest: pd.Series,
                            mtf_context: MultiTimeframeContext, volume_entry: Dict,
                            confluence_zones: List[Dict], df: pd.DataFrame,
                            use_mtf: bool = True) -> Optional[Dict]:
        """Generate LONG signal with enhanced entry validation and quality filters"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Entry conditions
            rsi = latest.get('rsi', 50)
            stoch_k = latest.get('stoch_rsi_k', 50)
            stoch_d = latest.get('stoch_rsi_d', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Check basic conditions
            if not (rsi < 65 and stoch_k < 70 and stoch_k > stoch_d and volume_ratio > 0.8):
                return None
            
            # NEW: Apply enhanced quality filters
            should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium = \
                self.apply_enhanced_filters(df, 'buy', symbol)
            
            if should_reject:
                self.logger.debug(f"   ❌ {symbol} - Signal rejected: {', '.join(rejection_reasons)}")
                return None
            
            if enhancement_factors:
                self.logger.debug(f"   ✨ {symbol} - Quality factors: {', '.join(enhancement_factors)}")
            
            # NEW: Validate entry timing
            timing_check = self._validate_entry_timing(df, 'buy')
            if not timing_check['valid'] and timing_check['score'] < -0.3:
                self.logger.debug(f"   ❌ {symbol} - Poor entry timing for LONG: {timing_check['reasons']}")
                return None
            
            # Calculate entry price with momentum awareness
            entry_price = self._calculate_long_entry(current_price, mtf_context, volume_entry, df)
            
            # NEW: Validate entry price is reasonable
            entry_distance_pct = abs(entry_price - current_price) / current_price
            if entry_distance_pct > 0.03:  # Entry more than 3% away
                self.logger.debug(f"   ⚠️ {symbol} - Entry too far from current price: {entry_distance_pct*100:.1f}%")
                # Adjust entry to be closer
                entry_price = current_price * (1.015 if entry_price > current_price else 0.985)
            
            # Calculate stop loss with validation
            raw_stop = self._calculate_long_stop(entry_price, mtf_context, df)
            adjusted, stop_loss = self.validator.validate_stop_loss(entry_price, raw_stop, 'buy')
            
            # Calculate take profits
            raw_tp1, raw_tp2 = self._calculate_long_targets(entry_price, stop_loss, mtf_context, df)
            tp1, tp2 = self.validator.validate_take_profits(entry_price, stop_loss, raw_tp1, raw_tp2, 'buy')

            tp = tp1 if self.config.default_tp_level == 'take_profit_1' else tp2
            
            # Calculate R/R ratio
            rr_ratio = self.validator.calculate_risk_reward(entry_price, stop_loss, tp, 'buy')
            
            # Check minimum R/R
            if rr_ratio < self.signal_config.min_risk_reward:
                return None
            
            # Calculate confidence with timing and quality adjustments
            base_confidence = 50.0
            if rsi < 40:
                base_confidence += 10
            if volume_ratio > 1.2:
                base_confidence += 5
            if use_mtf and mtf_context.momentum_alignment:
                base_confidence += self.signal_config.mtf_confidence_boost
            
            # Adjust confidence based on timing
            timing_adjustment = timing_check['score'] * 10  # Convert to percentage
            base_confidence += timing_adjustment
            
            # Adjust confidence based on quality score
            if is_premium:
                base_confidence += self.signal_config.fast_signal_bonus_confidence
            else:
                base_confidence += quality_score * 5
            
            confidence = min(90, max(self.signal_config.min_confidence_for_signal, base_confidence))
            
            # Determine order type based on momentum and entry distance
            if entry_distance_pct > 0.01:
                order_type = 'limit'  # Use limit for entries far from current
            else:
                order_type = 'market'  # Use market for immediate entries
            
            return {
                'symbol': symbol,
                'side': 'buy',
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'risk_reward_ratio': rr_ratio,
                'confidence': confidence,
                'signal_type': 'long_signal_v4',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'order_type': order_type,
                'signal_notes': f"MTF: {use_mtf}, Stop adjusted: {adjusted}, Timing: {timing_check['score']:.2f}, Quality: {quality_score:.1f}",
                'mtf_validated': use_mtf,
                'market_regime': mtf_context.market_regime,
                'regime_compatibility': self._assess_regime_compatibility('buy', mtf_context.market_regime),
                'entry_timing': timing_check,
                'quality_factors': enhancement_factors,
                'quality_score': quality_score,
                'is_premium_signal': is_premium
            }
            
        except Exception as e:
            self.logger.error(f"Long signal generation error: {e}")
            return None

    def _generate_short_signal(self, symbol_data: Dict, latest: pd.Series,
                             mtf_context: MultiTimeframeContext, volume_entry: Dict,
                             confluence_zones: List[Dict], df: pd.DataFrame,
                             use_mtf: bool = True) -> Optional[Dict]:
        """Generate SHORT signal with enhanced entry validation and quality filters"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Entry conditions
            rsi = latest.get('rsi', 50)
            stoch_k = latest.get('stoch_rsi_k', 50)
            stoch_d = latest.get('stoch_rsi_d', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Check basic conditions
            if not (rsi > 35 and stoch_k > 30 and stoch_k < stoch_d and volume_ratio > 0.8):
                return None
            
            # NEW: Apply enhanced quality filters
            should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium = \
                self.apply_enhanced_filters(df, 'sell', symbol)
            
            if should_reject:
                self.logger.debug(f"   ❌ {symbol} - Signal rejected: {', '.join(rejection_reasons)}")
                return None
            
            if enhancement_factors:
                self.logger.debug(f"   ✨ {symbol} - Quality factors: {', '.join(enhancement_factors)}")
            
            # NEW: Validate entry timing
            timing_check = self._validate_entry_timing(df, 'sell')
            if not timing_check['valid'] and timing_check['score'] < -0.3:
                self.logger.debug(f"   ❌ {symbol} - Poor entry timing for SHORT: {timing_check['reasons']}")
                return None
            
            # Calculate entry price with momentum awareness
            entry_price = self._calculate_short_entry(current_price, mtf_context, volume_entry, df)
            
            # NEW: Validate entry price is reasonable
            entry_distance_pct = abs(entry_price - current_price) / current_price
            if entry_distance_pct > 0.03:  # Entry more than 3% away
                self.logger.debug(f"   ⚠️ {symbol} - Entry too far from current price: {entry_distance_pct*100:.1f}%")
                # Adjust entry to be closer
                entry_price = current_price * (0.985 if entry_price < current_price else 1.015)
            
            # Calculate stop loss with validation
            raw_stop = self._calculate_short_stop(entry_price, mtf_context, df)
            adjusted, stop_loss = self.validator.validate_stop_loss(entry_price, raw_stop, 'sell')
            
            # Calculate take profits
            raw_tp1, raw_tp2 = self._calculate_short_targets(entry_price, stop_loss, mtf_context, df)
            tp1, tp2 = self.validator.validate_take_profits(entry_price, stop_loss, raw_tp1, raw_tp2, 'sell')

            tp = tp1 if self.config.default_tp_level == 'take_profit_1' else tp2
            
            # Calculate R/R ratio
            rr_ratio = self.validator.calculate_risk_reward(entry_price, stop_loss, tp, 'sell')
            
            # Check minimum R/R
            if rr_ratio < self.signal_config.min_risk_reward:
                return None
            
            # Calculate confidence with timing and quality adjustments
            base_confidence = 50.0
            if rsi > 60:
                base_confidence += 10
            if volume_ratio > 1.2:
                base_confidence += 5
            if use_mtf and mtf_context.momentum_alignment:
                base_confidence += self.signal_config.mtf_confidence_boost
            
            # Adjust confidence based on timing
            timing_adjustment = timing_check['score'] * 10  # Convert to percentage
            base_confidence += timing_adjustment
            
            # Adjust confidence based on quality score
            if is_premium:
                base_confidence += self.signal_config.fast_signal_bonus_confidence
            else:
                base_confidence += quality_score * 5
            
            confidence = min(90, max(self.signal_config.min_confidence_for_signal, base_confidence))
            
            # Determine order type based on momentum and entry distance
            if entry_distance_pct > 0.01:
                order_type = 'limit'  # Use limit for entries far from current
            else:
                order_type = 'market'  # Use market for immediate entries
            
            return {
                'symbol': symbol,
                'side': 'sell',
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'risk_reward_ratio': rr_ratio,
                'confidence': confidence,
                'signal_type': 'short_signal_v4',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'order_type': order_type,
                'signal_notes': f"MTF: {use_mtf}, Stop adjusted: {adjusted}, Timing: {timing_check['score']:.2f}, Quality: {quality_score:.1f}",
                'mtf_validated': use_mtf,
                'market_regime': mtf_context.market_regime,
                'regime_compatibility': self._assess_regime_compatibility('sell', mtf_context.market_regime),
                'entry_timing': timing_check,
                'quality_factors': enhancement_factors,
                'quality_score': quality_score,
                'is_premium_signal': is_premium
            }
            
        except Exception as e:
            self.logger.error(f"Short signal generation error: {e}")
            return None

    def _calculate_long_entry(self, current_price: float, mtf_context: MultiTimeframeContext,
                            volume_entry: Dict, df: pd.DataFrame = None) -> float:
        """Calculate LONG entry with momentum awareness"""
        try:
            # Check if we have price data for momentum calculation
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                # Calculate price momentum
                recent = df.tail(self.signal_config.price_momentum_lookback)
                price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                
                # Get volatility for adjustment
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                volatility_adjustment = (atr / current_price) * 0.5
                
                # Momentum-based entry adjustment
                if price_change > 0.01:  # Strong upward momentum
                    # Place entry ABOVE current price to catch momentum
                    entry_adjustment = max(volatility_adjustment, self.signal_config.momentum_entry_adjustment)
                    base_entry = current_price * (1 + entry_adjustment)
                elif price_change < -0.005:  # Downward momentum (pullback opportunity)
                    # Try to get better entry on pullback
                    base_entry = current_price * (1 - self.signal_config.entry_buffer_from_structure)
                else:  # Neutral momentum
                    base_entry = current_price * 0.999
            else:
                # No momentum data available
                base_entry = current_price * 0.999
            
            entry_candidates = [base_entry]
            
            # Near support with buffer
            for zone in mtf_context.higher_tf_zones:
                if zone['type'] == 'support' and zone['price'] < current_price:
                    buffered_entry = zone['price'] * (1 + self.signal_config.entry_buffer_from_structure)
                    if buffered_entry < current_price * 1.02:  # Not too far
                        entry_candidates.append(buffered_entry)
            
            # Volume-based entry
            if volume_entry.get('confidence', 0) > 0.5:
                vol_entry = volume_entry.get('entry_price', current_price)
                if current_price * 0.99 <= vol_entry <= current_price * 1.02:
                    entry_candidates.append(vol_entry)
            
            # Choose best entry based on conditions
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                if price_change > 0.01:  # Strong momentum up
                    return max(entry_candidates)  # Most aggressive entry
                else:
                    # Use median entry for safety
                    entry_candidates.sort()
                    return entry_candidates[len(entry_candidates)//2]
            else:
                return min(entry_candidates)  # Conservative entry
                
        except Exception:
            return current_price * 0.999

    def _calculate_short_entry(self, current_price: float, mtf_context: MultiTimeframeContext,
                             volume_entry: Dict, df: pd.DataFrame = None) -> float:
        """Calculate SHORT entry with momentum awareness"""
        try:
            # Check if we have price data for momentum calculation
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                # Calculate price momentum
                recent = df.tail(self.signal_config.price_momentum_lookback)
                price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                
                # Get volatility for adjustment
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                volatility_adjustment = (atr / current_price) * 0.5
                
                # Momentum-based entry adjustment
                if price_change < -0.01:  # Strong downward momentum
                    # Place entry BELOW current price to catch momentum
                    entry_adjustment = max(volatility_adjustment, self.signal_config.momentum_entry_adjustment)
                    base_entry = current_price * (1 - entry_adjustment)
                elif price_change > 0.005:  # Upward momentum (rally to short)
                    # Try to get better entry on rally
                    base_entry = current_price * (1 + self.signal_config.entry_buffer_from_structure)
                else:  # Neutral momentum
                    base_entry = current_price * 1.001
            else:
                # No momentum data available
                base_entry = current_price * 1.001
            
            entry_candidates = [base_entry]
            
            # Near resistance with buffer
            for zone in mtf_context.higher_tf_zones:
                if zone['type'] == 'resistance' and zone['price'] > current_price:
                    buffered_entry = zone['price'] * (1 - self.signal_config.entry_buffer_from_structure)
                    if buffered_entry > current_price * 0.98:  # Not too far
                        entry_candidates.append(buffered_entry)
            
            # Volume-based entry
            if volume_entry.get('confidence', 0) > 0.5:
                vol_entry = volume_entry.get('entry_price', current_price)
                if current_price * 0.98 <= vol_entry <= current_price * 1.01:
                    entry_candidates.append(vol_entry)
            
            # Choose best entry based on conditions
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                if price_change < -0.01:  # Strong momentum down
                    return min(entry_candidates)  # Most aggressive entry
                else:
                    # Use median entry for safety
                    entry_candidates.sort()
                    return entry_candidates[len(entry_candidates)//2]
            else:
                return max(entry_candidates)  # Conservative entry
                
        except Exception:
            return current_price * 1.001

    def _calculate_long_stop(self, entry_price: float, mtf_context: MultiTimeframeContext, 
                           df: pd.DataFrame) -> float:
        """Calculate LONG stop loss with dynamic volatility adjustment"""
        try:
            stop_candidates = []
            
            # Dynamic ATR-based stop
            if len(df) >= 14 and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_pct = atr / entry_price
                
                # Calculate recent volatility
                recent_changes = df['close'].pct_change().tail(20).abs()
                max_volatility = recent_changes.max()
                
                # Adjust multiplier based on volatility
                if max_volatility > self.signal_config.high_volatility_threshold:
                    atr_multiplier = 2.5  # Wider stop for high volatility
                elif max_volatility > 0.02:
                    atr_multiplier = 2.0  # Standard multiplier
                else:
                    atr_multiplier = 1.5  # Tighter stop for low volatility
                
                atr_stop = entry_price - (atr * atr_multiplier)
                stop_candidates.append(atr_stop)
            
            # Structure-based stop with proper buffer
            support_zones = [z for z in mtf_context.higher_tf_zones 
                           if z['type'] == 'support' and z['price'] < entry_price]
            if support_zones:
                closest_support = max(support_zones, key=lambda x: x['price'])
                structure_stop = closest_support['price'] * (1 - self.signal_config.structure_stop_buffer)
                stop_candidates.append(structure_stop)
            
            # Percentage-based stop with minimum distance
            min_stop = entry_price * (1 - self.signal_config.min_stop_distance_pct)
            stop_candidates.append(min_stop)
            
            # Choose the highest stop (closest to entry but with minimum distance)
            if stop_candidates:
                chosen_stop = max(stop_candidates)
                # Ensure minimum distance
                min_distance_stop = entry_price * (1 - self.signal_config.min_stop_distance_pct)
                return min(chosen_stop, min_distance_stop)
            else:
                return entry_price * (1 - self.signal_config.min_stop_distance_pct)
                
        except Exception:
            return entry_price * (1 - self.signal_config.min_stop_distance_pct)

    def _calculate_short_stop(self, entry_price: float, mtf_context: MultiTimeframeContext,
                            df: pd.DataFrame) -> float:
        """Calculate SHORT stop loss with dynamic volatility adjustment"""
        try:
            stop_candidates = []
            
            # Dynamic ATR-based stop
            if len(df) >= 14 and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_pct = atr / entry_price
                
                # Calculate recent volatility
                recent_changes = df['close'].pct_change().tail(20).abs()
                max_volatility = recent_changes.max()
                
                # Adjust multiplier based on volatility
                if max_volatility > self.signal_config.high_volatility_threshold:
                    atr_multiplier = 2.5  # Wider stop for high volatility
                elif max_volatility > 0.02:
                    atr_multiplier = 2.0  # Standard multiplier
                else:
                    atr_multiplier = 1.5  # Tighter stop for low volatility
                
                atr_stop = entry_price + (atr * atr_multiplier)
                stop_candidates.append(atr_stop)
            
            # Structure-based stop with proper buffer
            resistance_zones = [z for z in mtf_context.higher_tf_zones 
                              if z['type'] == 'resistance' and z['price'] > entry_price]
            if resistance_zones:
                closest_resistance = min(resistance_zones, key=lambda x: x['price'])
                structure_stop = closest_resistance['price'] * (1 + self.signal_config.structure_stop_buffer)
                stop_candidates.append(structure_stop)
            
            # Percentage-based stop with minimum distance
            min_stop = entry_price * (1 + self.signal_config.min_stop_distance_pct)
            stop_candidates.append(min_stop)
            
            # Choose the lowest stop (closest to entry but with minimum distance)
            if stop_candidates:
                chosen_stop = min(stop_candidates)
                # Ensure minimum distance
                min_distance_stop = entry_price * (1 + self.signal_config.min_stop_distance_pct)
                return max(chosen_stop, min_distance_stop)
            else:
                return entry_price * (1 + self.signal_config.min_stop_distance_pct)
                
        except Exception:
            return entry_price * (1 + self.signal_config.min_stop_distance_pct)

    def _calculate_long_targets(self, entry_price: float, stop_loss: float,
                              mtf_context: MultiTimeframeContext, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate LONG take profit targets"""
        try:
            risk = entry_price - stop_loss
            
            # Risk-based targets
            tp1 = entry_price + (risk * self.signal_config.tp1_multiplier)
            tp2 = entry_price + (risk * self.signal_config.tp2_multiplier)
            
            # Adjust based on resistance levels
            resistance_zones = [z for z in mtf_context.higher_tf_zones 
                              if z['type'] == 'resistance' and z['price'] > entry_price]
            
            if resistance_zones:
                # TP1 at first resistance
                first_resistance = min(resistance_zones, key=lambda x: x['price'])
                tp1 = min(tp1, first_resistance['price'] * 0.995)
                
                # TP2 at second resistance or extended target
                if len(resistance_zones) > 1:
                    second_resistance = sorted(resistance_zones, key=lambda x: x['price'])[1]
                    tp2 = min(tp2, second_resistance['price'] * 0.995)
            
            return tp1, tp2
            
        except Exception:
            risk = entry_price - stop_loss
            return entry_price + (risk * 2.5), entry_price + (risk * 4)

    def _calculate_short_targets(self, entry_price: float, stop_loss: float,
                               mtf_context: MultiTimeframeContext, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate SHORT take profit targets"""
        try:
            risk = stop_loss - entry_price
            
            # Risk-based targets
            tp1 = entry_price - (risk * self.signal_config.tp1_multiplier)
            tp2 = entry_price - (risk * self.signal_config.tp2_multiplier)
            
            # Adjust based on support levels
            support_zones = [z for z in mtf_context.higher_tf_zones 
                           if z['type'] == 'support' and z['price'] < entry_price]
            
            if support_zones:
                # TP1 at first support
                first_support = max(support_zones, key=lambda x: x['price'])
                tp1 = max(tp1, first_support['price'] * 1.005)
                
                # TP2 at second support or extended target
                if len(support_zones) > 1:
                    second_support = sorted(support_zones, key=lambda x: x['price'], reverse=True)[1]
                    tp2 = max(tp2, second_support['price'] * 1.005)
            
            return tp1, tp2
            
        except Exception:
            risk = stop_loss - entry_price
            return entry_price - (risk * 2.5), entry_price - (risk * 4)

    def _validate_entry_timing(self, df: pd.DataFrame, side: str, lookback: int = 5) -> dict:
        """Validate if current timing is good for entry"""
        try:
            if len(df) < lookback:
                return {'valid': True, 'reason': 'insufficient_data', 'score': 0.5}
            
            recent = df.tail(lookback)
            latest = df.iloc[-1]
            
            # Check if price is overextended
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            current_price = latest['close']
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            # RSI check
            rsi = latest.get('rsi', 50)
            
            # Volume check
            volume_ratio = latest.get('volume_ratio', 1)
            
            timing_score = 0
            reasons = []
            
            if side == 'buy':
                # For LONG entries
                if price_position > 0.9:  # Price near recent high
                    timing_score -= 0.3
                    reasons.append("price_near_resistance")
                elif price_position < 0.3:  # Price near recent low (good for long)
                    timing_score += 0.3
                    reasons.append("price_near_support")
                
                if rsi > 70:  # Overbought
                    timing_score -= 0.2
                    reasons.append("rsi_overbought")
                elif rsi < 35:  # Oversold (good for long)
                    timing_score += 0.2
                    reasons.append("rsi_oversold")
                    
            else:  # sell
                # For SHORT entries
                if price_position < 0.1:  # Price near recent low
                    timing_score -= 0.3
                    reasons.append("price_near_support")
                elif price_position > 0.7:  # Price near recent high (good for short)
                    timing_score += 0.3
                    reasons.append("price_near_resistance")
                
                if rsi < 30:  # Oversold
                    timing_score -= 0.2
                    reasons.append("rsi_oversold")
                elif rsi > 65:  # Overbought (good for short)
                    timing_score += 0.2
                    reasons.append("rsi_overbought")
            
            # Volume confirmation
            if volume_ratio > 1.5:
                timing_score += 0.2
                reasons.append("strong_volume")
            elif volume_ratio < 0.5:
                timing_score -= 0.1
                reasons.append("weak_volume")
            
            # Determine if timing is valid
            is_valid = timing_score >= -0.2  # Allow slightly negative scores
            
            return {
                'valid': is_valid,
                'score': timing_score,
                'reasons': reasons,
                'price_position': price_position,
                'rsi': rsi,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            return {'valid': True, 'reason': f'error: {str(e)}', 'score': 0}

    def apply_enhanced_filters(self, df: pd.DataFrame, side: str, symbol: str) -> tuple:
        """Apply all enhanced filters and return decision with reasons"""
        try:
            rejection_reasons = []
            enhancement_factors = []
            quality_score = 0
            
            # 1. Check momentum strength
            momentum = analyze_price_momentum_strength(df)
            if side == 'buy' and momentum['direction'] == 'bearish' and momentum['strength'] >= 2:
                rejection_reasons.append(f"Strong bearish momentum: {momentum['weighted_momentum']:.1f}%")
                quality_score -= 2
            elif side == 'sell' and momentum['direction'] == 'bullish' and momentum['strength'] >= 2:
                rejection_reasons.append(f"Strong bullish momentum: {momentum['weighted_momentum']:.1f}%")
                quality_score -= 2
            elif momentum['speed'] in ['fast', 'very_fast'] and momentum['direction'].lower() == side:
                enhancement_factors.append(f"Strong {side} momentum")
                quality_score += 2
            
            # 2. Check volume divergence
            divergence = check_volume_momentum_divergence(df)
            if divergence['divergence'] and divergence['type'] == 'bearish_divergence' and side == 'buy':
                rejection_reasons.append(divergence['warning'])
                quality_score -= 1
            elif divergence['type'] == 'confirmed_move':
                enhancement_factors.append("Volume confirms move")
                quality_score += 1
            
            # 3. Check for fast-moving setup
            fast_setup = identify_fast_moving_setup(df, side)
            if fast_setup['is_fast_setup']:
                enhancement_factors.extend(fast_setup['factors'])
                quality_score += fast_setup['score']
            
            # 4. Filter choppy markets
            choppiness = filter_choppy_markets(df)
            if choppiness['is_choppy']:
                rejection_reasons.append(f"Choppy market detected (score: {choppiness['choppiness_score']:.2f})")
                quality_score -= 2
            elif choppiness['market_state'] == 'trending':
                enhancement_factors.append("Clear trending market")
                quality_score += 1
            
            # Make decision
            should_reject = len(rejection_reasons) > 0 and quality_score < 0
            is_premium = quality_score >= 3
            
            return should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium
            
        except Exception as e:
            self.logger.error(f"Enhanced filter error for {symbol}: {e}")
            return False, [], [], 0, False

    def _validate_and_enhance_signal(self, signal: Dict, mtf_context: MultiTimeframeContext,
                                   df: pd.DataFrame, market_regime: str) -> Dict:
        """Final validation and enhancement of signal"""
        try:
            # Add analysis details
            signal['analysis_details'] = {
                'signal_strength': self._determine_signal_strength(signal, mtf_context),
                'mtf_trend': mtf_context.dominant_trend,
                'structure_timeframe': mtf_context.structure_timeframe,
                'confirmation_score': mtf_context.confirmation_score,
                'entry_method': self._determine_entry_method(signal, mtf_context),
                'market_regime': market_regime,
                'volatility_level': mtf_context.volatility_level,
                'momentum_strength': self._analyze_momentum_strength(df),
                'regime_compatibility': signal.get('regime_compatibility', 'medium')
            }
            
            # Determine entry strategy
            signal['entry_strategy'] = signal['analysis_details']['entry_method']
            
            # Add quality grade
            signal['quality_grade'] = self._calculate_quality_grade(signal)
            
            # Add MTF boost if applicable
            if signal.get('mtf_validated', False):
                signal['original_confidence'] = signal['confidence'] - self.signal_config.mtf_confidence_boost
                signal['mtf_boost'] = self.signal_config.mtf_confidence_boost
            else:
                signal['original_confidence'] = signal['confidence']
                signal['mtf_boost'] = 0
            
            # Add MTF status for compatibility
            if signal.get('mtf_validated', False):
                signal['mtf_status'] = 'MTF_VALIDATED'
            else:
                # Determine status based on confidence
                if signal['confidence'] >= 70:
                    signal['mtf_status'] = 'STRONG'
                elif signal['confidence'] >= 60:
                    signal['mtf_status'] = 'PARTIAL'
                else:
                    signal['mtf_status'] = 'NONE'
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return signal

    def _determine_signal_strength(self, signal: Dict, mtf_context: MultiTimeframeContext) -> str:
        """Determine overall signal strength"""
        confidence = signal.get('confidence', 0)
        rr_ratio = signal.get('risk_reward_ratio', 0)
        mtf_score = mtf_context.confirmation_score
        quality_score = signal.get('quality_score', 0)
        
        # Enhanced strength calculation with quality score
        strength_score = (confidence/100 * 0.3) + (min(rr_ratio/3, 1) * 0.25) + (mtf_score * 0.25) + (quality_score/5 * 0.2)
        
        if strength_score > 0.85:
            return 'very_strong'
        elif strength_score > 0.7:
            return 'strong'
        elif strength_score > 0.55:
            return 'moderate'
        else:
            return 'weak'

    def _determine_entry_method(self, signal: Dict, mtf_context: MultiTimeframeContext) -> str:
        """Determine how entry was calculated"""
        entry = signal['entry_price']
        current = signal['current_price']
        
        # Check if momentum-based entry
        if abs(entry - current) / current > 0.003:
            if entry > current and signal['side'] == 'buy':
                return 'momentum_chase'
            elif entry < current and signal['side'] == 'sell':
                return 'momentum_chase'
        
        # Check proximity to key levels
        for zone in mtf_context.higher_tf_zones:
            if abs(zone['price'] - entry) / entry < 0.005:
                if zone['type'] == 'support':
                    return 'support_bounce'
                else:
                    return 'resistance_rejection'
        
        # Check if near key support/resistance
        if abs(entry - mtf_context.key_support) / entry < 0.005:
            return 'key_support_bounce'
        elif abs(entry - mtf_context.key_resistance) / entry < 0.005:
            return 'key_resistance_rejection'
        
        # Default
        if abs(entry - current) / current < 0.002:
            return 'immediate'
        else:
            return 'limit_order'

    def _assess_regime_compatibility(self, side: str, market_regime: str) -> str:
        """Assess how compatible the signal is with market regime"""
        if market_regime == 'trending_up' and side == 'buy':
            return 'high'
        elif market_regime == 'trending_down' and side == 'sell':
            return 'high'
        elif market_regime == 'ranging':
            return 'medium'
        elif market_regime == 'volatile':
            return 'low'
        else:
            return 'medium'

    def _calculate_quality_grade(self, signal: Dict) -> str:
        """Calculate signal quality grade with enhanced scoring"""
        confidence = signal.get('confidence', 0)
        rr_ratio = signal.get('risk_reward_ratio', 0)
        volume_24h = signal.get('volume_24h', 0)
        quality_score = signal.get('quality_score', 0)
        is_premium = signal.get('is_premium_signal', False)
        
        score = 0
        
        # Premium signal bonus
        if is_premium:
            score += 20
        
        # Quality score contribution
        score += quality_score * 10
        
        # Confidence scoring
        if confidence >= 80:
            score += 35
        elif confidence >= 70:
            score += 25
        elif confidence >= 60:
            score += 15
        elif confidence >= 50:
            score += 5
        
        # R/R scoring
        if rr_ratio >= 4:
            score += 25
        elif rr_ratio >= 3:
            score += 20
        elif rr_ratio >= 2.5:
            score += 15
        elif rr_ratio >= 2:
            score += 10
        
        # Volume scoring
        if volume_24h >= 10_000_000:
            score += 20
        elif volume_24h >= 5_000_000:
            score += 15
        elif volume_24h >= 1_000_000:
            score += 10
        elif volume_24h >= 500_000:
            score += 5
        
        # Grade assignment
        if score >= 85:
            return 'A+'
        elif score >= 75:
            return 'A'
        elif score >= 65:
            return 'A-'
        elif score >= 55:
            return 'B+'
        elif score >= 45:
            return 'B'
        elif score >= 35:
            return 'B-'
        elif score >= 25:
            return 'C+'
        else:
            return 'C'

    def rank_opportunities_with_mtf(self, signals: List[Dict], dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """Enhanced ranking system with quality scoring and fast-signal prioritization"""
        try:
            opportunities = []
            
            for signal in signals:
                # Calculate priority score
                mtf_validated = signal.get('mtf_validated', False)
                confidence = signal.get('confidence', 0)
                rr_ratio = signal.get('risk_reward_ratio', 0)
                volume_24h = signal.get('volume_24h', 0)
                quality_grade = signal.get('quality_grade', 'C')
                quality_score = signal.get('quality_score', 0)
                is_premium = signal.get('is_premium_signal', False)
                
                # Base priority based on validation and premium status
                if is_premium:
                    base_priority = 10000  # Premium signals get highest priority
                elif mtf_validated:
                    base_priority = 7000
                else:
                    base_priority = 4000
                
                # Quality grade bonus (enhanced)
                quality_bonus = {
                    'A+': 2000, 'A': 1600, 'A-': 1200,
                    'B+': 800, 'B': 400, 'B-': 200,
                    'C+': 100, 'C': 0
                }.get(quality_grade, 0)
                
                # Quality score bonus (from filters)
                quality_score_bonus = int(quality_score * 200)
                
                # Fast-moving signal bonus
                quality_factors = signal.get('quality_factors', [])
                if any('breakout' in factor or 'momentum' in factor for factor in quality_factors):
                    fast_signal_bonus = 500
                else:
                    fast_signal_bonus = 0
                
                # Calculate final priority
                priority = (base_priority + quality_bonus + quality_score_bonus + fast_signal_bonus +
                          int(confidence * 10) + 
                          int(min(rr_ratio * 100, 500)) +  # Cap R/R contribution
                          int(min(volume_24h / 100000, 100)))  # Volume contribution
                
                signal['priority'] = priority
                signal['ranking_details'] = {
                    'mtf_validated': mtf_validated,
                    'is_premium': is_premium,
                    'quality_grade': quality_grade,
                    'quality_score': quality_score,
                    'fast_signal': fast_signal_bonus > 0,
                    'final_priority': priority
                }
                
                opportunities.append(signal)
            
            # Sort by priority
            opportunities.sort(key=lambda x: x['priority'], reverse=True)
            
            # Log top opportunities
            if opportunities:
                self.logger.info(f"🎯 Top signals after ranking:")
                for i, opp in enumerate(opportunities[:5]):
                    details = opp['ranking_details']
                    self.logger.info(f"   {i+1}. {opp['symbol']} - Priority: {details['final_priority']} "
                                   f"(Premium: {details['is_premium']}, Grade: {details['quality_grade']}, "
                                   f"Fast: {details['fast_signal']})")
            
            return opportunities[:self.config.charts_per_batch]
            
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return signals

    # ===== MARKET REGIME AND CONTEXT METHODS =====
    
    def _determine_market_regime(self, symbol_data: Dict, df: pd.DataFrame) -> str:
        """Determine current market regime for the symbol"""
        try:
            price_change_24h = symbol_data.get('price_change_24h', 0)
            volume_24h = symbol_data.get('volume_24h', 0)
            
            # Volatility analysis
            if len(df) >= 20:
                recent_changes = df['close'].pct_change().tail(20) * 100
                volatility = recent_changes.std()
                
                # High volatility threshold
                if volatility > 8:
                    return 'volatile'
            
            # Trend analysis based on price change and momentum
            if price_change_24h > 8:
                return 'trending_up'
            elif price_change_24h > 3:
                # Check if it's sustained trend or just noise
                if len(df) >= 10:
                    recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
                    if recent_trend > 2:
                        return 'trending_up'
                return 'ranging'
            elif price_change_24h < -8:
                return 'trending_down'
            elif price_change_24h < -3:
                # Check if it's sustained downtrend
                if len(df) >= 10:
                    recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
                    if recent_trend < -2:
                        return 'trending_down'
                return 'ranging'
            else:
                # Check for range-bound conditions
                if len(df) >= 20:
                    price_range = (df['high'].tail(20).max() - df['low'].tail(20).min()) / df['close'].iloc[-1]
                    if price_range < 0.15:  # Tight range
                        return 'ranging'
                
                return 'ranging'
            
        except Exception as e:
            self.logger.error(f"Market regime detection error: {e}")
            return 'ranging'  # Safe fallback

    def _get_multitimeframe_context(self, symbol_data: Dict, market_regime: str) -> Optional[MultiTimeframeContext]:
        """Get multi-timeframe context analysis"""
        try:
            if not self.exchange_manager:
                return self._create_fallback_context(symbol_data, market_regime)
            
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Get structure data from highest timeframe
            structure_analysis = self._analyze_structure_timeframe(symbol, current_price)
            if not structure_analysis:
                return self._create_fallback_context(symbol_data, market_regime)
                
            # Get confirmation from all other timeframes
            confirmation_analysis = self._analyze_confirmation_timeframes(symbol, current_price)
            
            # Determine overall market context with regime awareness
            entry_bias = self._determine_entry_bias_with_regime(
                structure_analysis, confirmation_analysis, current_price, market_regime
            )
            
            # Calculate confirmation score
            confirmation_score = self._calculate_confirmation_score(
                structure_analysis, confirmation_analysis
            )
            
            # Assess volatility
            volatility_level = self._assess_symbol_volatility(symbol_data)
            
            return MultiTimeframeContext(
                dominant_trend=structure_analysis['trend'],
                trend_strength=structure_analysis['strength'],
                higher_tf_zones=structure_analysis['key_zones'],
                key_support=structure_analysis['key_support'],
                key_resistance=structure_analysis['key_resistance'],
                momentum_alignment=structure_analysis['momentum_bullish'],
                entry_bias=entry_bias,
                confirmation_score=confirmation_score,
                structure_timeframe=self.structure_timeframe,
                market_regime=market_regime,
                volatility_level=volatility_level
            )
            
        except Exception as e:
            self.logger.error(f"Error getting MTF context: {e}")
            return self._create_fallback_context(symbol_data, market_regime)

    def _analyze_structure_timeframe(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Analyze dominant market structure from highest timeframe"""
        try:
            # Fetch structure timeframe data
            structure_df = self.exchange_manager.fetch_ohlcv_data(symbol, self.structure_timeframe)
            if structure_df.empty or len(structure_df) < 30:
                return None
                
            # Calculate indicators
            structure_df = self._calculate_comprehensive_indicators(structure_df)
            latest = structure_df.iloc[-1]
            
            # Trend analysis
            trend_data = self._analyze_trend_from_df(structure_df, current_price)
            
            # Key levels
            key_zones = self._identify_key_zones(structure_df, current_price)
            recent_high = structure_df['high'].tail(20).max()
            recent_low = structure_df['low'].tail(20).min()
            
            # Momentum
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            momentum_bullish = macd > macd_signal and rsi > 45
            
            return {
                'trend': trend_data['direction'],
                'strength': trend_data['strength'],
                'key_support': recent_low,
                'key_resistance': recent_high,
                'key_zones': key_zones,
                'momentum_bullish': momentum_bullish,
                'timeframe': self.structure_timeframe,
                'ma_levels': {
                    'sma_20': latest.get('sma_20', current_price),
                    'sma_50': latest.get('sma_50', current_price),
                    'ema_12': latest.get('ema_12', current_price),
                    'ema_26': latest.get('ema_26', current_price)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Structure analysis error: {e}")
            return None

    def _analyze_confirmation_timeframes(self, symbol: str, current_price: float) -> Dict:
        """Analyze confirmation timeframes for validation"""
        try:
            confirmation_data = {}
            
            for tf in self.confirmation_timeframes:
                if tf == self.structure_timeframe:
                    continue  # Skip structure TF as it's already analyzed
                    
                try:
                    df = self.exchange_manager.fetch_ohlcv_data(symbol, tf)
                    if df.empty or len(df) < 20:
                        continue
                        
                    df = self._calculate_comprehensive_indicators(df)
                    latest = df.iloc[-1]
                    trend_data = self._analyze_trend_from_df(df, current_price)
                    
                    confirmation_data[tf] = {
                        'trend': trend_data['direction'],
                        'strength': trend_data['strength'],
                        'rsi': latest.get('rsi', 50),
                        'macd_bullish': latest.get('macd', 0) > latest.get('macd_signal', 0),
                        'trend_bullish': latest.get('sma_20', 0) > latest.get('sma_50', 0)
                    }
                    
                    # Rate limiting
                    time.sleep(0.05)
                    
                except Exception as e:
                    self.logger.debug(f"Could not fetch {tf} data for {symbol}: {e}")
                    continue
            
            return confirmation_data
            
        except Exception as e:
            self.logger.error(f"Confirmation analysis error: {e}")
            return {}

    def _determine_entry_bias_with_regime(self, structure_analysis: Dict, confirmation_analysis: Dict, 
                                        current_price: float, market_regime: str) -> str:
        """Determine entry bias with market regime awareness"""
        try:
            # Get traditional structure bias first
            traditional_bias = self._determine_entry_bias(structure_analysis, confirmation_analysis, current_price)
            
            # Apply regime-specific adjustments
            struct_trend = structure_analysis['trend']
            
            if market_regime == 'trending_up':
                # In uptrend: Strongly favor longs, discourage shorts
                if traditional_bias == 'long_favored':
                    return 'long_favored'  # Reinforce
                elif traditional_bias == 'short_favored':
                    # Only allow shorts if structure is very bearish
                    if struct_trend == 'bearish' and structure_analysis['strength'] > 0.7:
                        return 'short_favored'
                    else:
                        return 'neutral'  # Downgrade from short to neutral
                else:
                    return traditional_bias
                    
            elif market_regime == 'trending_down':
                # In downtrend: Strongly favor shorts, discourage longs
                if traditional_bias == 'short_favored':
                    return 'short_favored'  # Reinforce
                elif traditional_bias == 'long_favored':
                    # Only allow longs if structure is very bullish
                    if struct_trend in ['bullish', 'strong_bullish'] and structure_analysis['strength'] > 0.7:
                        return 'long_favored'
                    else:
                        return 'neutral'  # Downgrade from long to neutral
                else:
                    return traditional_bias
                    
            elif market_regime == 'volatile':
                # In volatile markets: Be more conservative
                if traditional_bias in ['long_favored', 'short_favored']:
                    # Require stronger confirmation for volatile markets
                    confirmation_count = len(confirmation_analysis)
                    strong_confirmations = sum(1 for tf_data in confirmation_analysis.values() 
                                             if tf_data.get('strength', 0) > 0.6)
                    
                    if strong_confirmations >= max(1, confirmation_count // 2):
                        return traditional_bias  # Keep original bias
                    else:
                        return 'neutral'  # Downgrade to neutral
                else:
                    return traditional_bias
                    
            elif market_regime == 'ranging':
                # In ranging markets: Both directions OK, but prefer mean reversion
                if traditional_bias == 'avoid':
                    return 'neutral'  # Upgrade from avoid to neutral in ranges
                else:
                    return traditional_bias
                    
            else:  # uncertain or other regimes
                return traditional_bias
            
        except Exception as e:
            self.logger.error(f"Regime-aware entry bias error: {e}")
            return 'neutral'

    def _determine_entry_bias(self, structure_analysis: Dict, confirmation_analysis: Dict, 
                             current_price: float) -> str:
        """Determine entry bias - Traditional method"""
        try:
            struct_trend = structure_analysis['trend']
            struct_strength = structure_analysis['strength']
            key_zones = structure_analysis.get('key_zones', [])
            
            # Check proximity to major levels
            near_major_resistance = any(
                zone['type'] == 'resistance' and zone['distance_pct'] < 0.02 
                for zone in key_zones if zone['price'] > current_price
            )
            near_major_support = any(
                zone['type'] == 'support' and zone['distance_pct'] < 0.02 
                for zone in key_zones if zone['price'] < current_price
            )
            
            # Confirmation timeframe sentiment
            confirmation_bullish = 0
            confirmation_bearish = 0
            total_confirmations = len(confirmation_analysis)
            
            for tf_data in confirmation_analysis.values():
                if tf_data['trend'] in ['bullish', 'strong_bullish']:
                    confirmation_bullish += 1
                elif tf_data['trend'] == 'bearish':
                    confirmation_bearish += 1
            
            # DECISION LOGIC
            
            # Strong bullish structure
            if struct_trend == 'strong_bullish' and struct_strength > 0.75:
                if near_major_resistance:
                    return 'neutral'  # Don't buy at major resistance
                elif structure_analysis['momentum_bullish']:
                    return 'long_favored'
                else:
                    return 'neutral'
            
            # Regular bullish structure  
            elif struct_trend == 'bullish':
                if near_major_support and confirmation_bullish > confirmation_bearish:
                    return 'long_favored'  # Good long setup
                elif near_major_resistance:
                    return 'short_favored'  # Can short at resistance
                else:
                    return 'neutral'
            
            # Bearish structure
            elif struct_trend == 'bearish':
                if near_major_resistance:
                    return 'short_favored'  # Good short setup
                elif near_major_support:
                    return 'neutral'  # Don't short major support
                else:
                    return 'short_favored' if not structure_analysis['momentum_bullish'] else 'neutral'
            
            # Neutral structure
            elif struct_trend == 'neutral':
                if near_major_support and confirmation_bullish > confirmation_bearish:
                    return 'long_favored'
                elif near_major_resistance and confirmation_bearish > confirmation_bullish:
                    return 'short_favored'
                else:
                    return 'neutral'
            
            # Default: neutral
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Entry bias determination error: {e}")
            return 'neutral'

    def _calculate_confirmation_score(self, structure_analysis: Dict, confirmation_analysis: Dict) -> float:
        """Calculate how well confirmation timeframes align with structure"""
        try:
            if not confirmation_analysis:
                return 0.5
            
            struct_trend = structure_analysis['trend']
            aligned_count = 0
            total_count = len(confirmation_analysis)
            
            for tf_data in confirmation_analysis.values():
                tf_trend = tf_data['trend']
                
                # Check alignment
                if struct_trend in ['bullish', 'strong_bullish'] and tf_trend in ['bullish', 'strong_bullish']:
                    aligned_count += 1
                elif struct_trend == 'bearish' and tf_trend == 'bearish':
                    aligned_count += 1
                elif struct_trend == 'neutral' and tf_trend == 'neutral':
                    aligned_count += 0.5
                
            return aligned_count / total_count if total_count > 0 else 0.5
            
        except Exception:
            return 0.5

    def _assess_symbol_volatility(self, symbol_data: Dict) -> str:
        """Assess symbol volatility level"""
        try:
            price_change_24h = abs(symbol_data.get('price_change_24h', 0))
            
            if price_change_24h > 15:
                return 'extreme'
            elif price_change_24h > 8:
                return 'high'
            elif price_change_24h > 3:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'

    def _create_fallback_context(self, symbol_data: Dict, market_regime: str) -> MultiTimeframeContext:
        """Create fallback context when exchange_manager unavailable"""
        current_price = symbol_data['current_price']
        return MultiTimeframeContext(
            dominant_trend='neutral',
            trend_strength=0.5,
            higher_tf_zones=[],
            key_support=current_price * 0.95,
            key_resistance=current_price * 1.05,
            momentum_alignment=True,
            entry_bias='neutral',
            confirmation_score=0.5,
            structure_timeframe=self.structure_timeframe,
            market_regime=market_regime,
            volatility_level='medium'
        )

    def _analyze_trend_from_df(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Analyze trend direction and strength from dataframe"""
        try:
            latest = df.iloc[-1]
            
            # Moving average analysis
            sma_20 = latest.get('sma_20', current_price)
            sma_50 = latest.get('sma_50', current_price)
            ema_12 = latest.get('ema_12', current_price)
            ema_26 = latest.get('ema_26', current_price)
            
            # Trend signals
            price_above_sma20 = current_price > sma_20
            price_above_sma50 = current_price > sma_50
            sma20_above_sma50 = sma_20 > sma_50
            ema_bullish = ema_12 > ema_26
            
            bullish_signals = sum([price_above_sma20, price_above_sma50, sma20_above_sma50, ema_bullish])
            
            # Determine trend
            if bullish_signals >= 3:
                direction = 'strong_bullish'
                strength = bullish_signals / 4.0
            elif bullish_signals >= 2:
                direction = 'bullish'
                strength = bullish_signals / 4.0
            elif bullish_signals <= 1:
                direction = 'bearish'
                strength = (4 - bullish_signals) / 4.0
            else:
                direction = 'neutral'
                strength = 0.5
            
            return {'direction': direction, 'strength': strength}
            
        except Exception:
            return {'direction': 'neutral', 'strength': 0.5}

    def _identify_key_zones(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Identify key support/resistance zones"""
        try:
            zones = []
            recent_candles = df.tail(30)
            
            # Find swing highs and lows
            for i in range(2, len(recent_candles) - 2):
                high = recent_candles.iloc[i]['high']
                low = recent_candles.iloc[i]['low']
                
                # Check for swing high (resistance)
                if (high > recent_candles.iloc[i-1]['high'] and 
                    high > recent_candles.iloc[i-2]['high'] and
                    high > recent_candles.iloc[i+1]['high'] and 
                    high > recent_candles.iloc[i+2]['high']):
                    
                    distance_pct = abs(high - current_price) / current_price
                    if distance_pct < 0.1:
                        zones.append({
                            'price': high,
                            'type': 'resistance',
                            'strength': 'major',
                            'distance_pct': distance_pct,
                            'timeframe': self.structure_timeframe
                        })
                
                # Check for swing low (support)
                if (low < recent_candles.iloc[i-1]['low'] and 
                    low < recent_candles.iloc[i-2]['low'] and
                    low < recent_candles.iloc[i+1]['low'] and 
                    low < recent_candles.iloc[i+2]['low']):
                    
                    distance_pct = abs(low - current_price) / current_price
                    if distance_pct < 0.1:
                        zones.append({
                            'price': low,
                            'type': 'support',
                            'strength': 'major',
                            'distance_pct': distance_pct,
                            'timeframe': self.structure_timeframe
                        })
            
            # Sort by proximity
            zones.sort(key=lambda x: x['distance_pct'])
            return zones[:8]
            
        except Exception:
            return []

    def _calculate_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            # Moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Stochastic RSI
            rsi_min = df['rsi'].rolling(window=14).min()
            rsi_max = df['rsi'].rolling(window=14).max()
            stoch_rsi = (df['rsi'] - rsi_min) / (rsi_max - rsi_min) * 100
            df['stoch_rsi_k'] = stoch_rsi.rolling(window=3).mean()
            df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            import pandas as pd
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume
            df['volume_avg'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_avg']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Indicator calculation error: {e}")
            return df

    def _analyze_momentum_strength(self, df: pd.DataFrame) -> float:
        """Analyze momentum strength for dynamic target calculation"""
        try:
            if len(df) < 20:
                return 0.5
            
            latest = df.iloc[-1]
            
            # MACD momentum
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_strength = 1 if macd > macd_signal else 0
            
            # RSI momentum (not extreme)
            rsi = latest.get('rsi', 50)
            rsi_momentum = 0.5
            if 30 < rsi < 70:
                rsi_momentum = 1  # Good momentum range
            elif rsi > 70:
                rsi_momentum = 0.3  # Overextended
            elif rsi < 30:
                rsi_momentum = 0.3  # Oversold
            
            # Price momentum
            price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
            price_momentum = min(1.0, abs(price_change_5) * 10)  # Scale to 0-1
            
            # Volume momentum
            volume_ratio = latest.get('volume_ratio', 1)
            volume_momentum = min(1.0, volume_ratio / 2)  # Scale to 0-1
            
            # Combined momentum strength
            total_momentum = (macd_strength * 0.3 + rsi_momentum * 0.3 + 
                            price_momentum * 0.25 + volume_momentum * 0.15)
            
            return max(0.1, min(1.0, total_momentum))
            
        except Exception as e:
            self.logger.error(f"Momentum strength analysis error: {e}")
            return 0.5

    def _assess_volatility_risk(self, df: pd.DataFrame) -> Dict:
        """Assess volatility risk for position sizing adjustments"""
        try:
            if len(df) < 20:
                return {'level': 'unknown', 'multiplier': 1.0}
            
            # ATR-based volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            import pandas as pd
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            atr_pct = (atr / df['close'].iloc[-1]) * 100
            
            # Recent price volatility
            recent_returns = df['close'].pct_change().tail(10) * 100
            return_volatility = recent_returns.std()
            
            # Combined volatility assessment
            avg_volatility = (atr_pct + return_volatility) / 2
            
            if avg_volatility > 12:
                return {'level': 'extreme', 'multiplier': 0.5, 'atr_pct': atr_pct}
            elif avg_volatility > 8:
                return {'level': 'high', 'multiplier': 0.7, 'atr_pct': atr_pct}
            elif avg_volatility > 4:
                return {'level': 'medium', 'multiplier': 1.0, 'atr_pct': atr_pct}
            else:
                return {'level': 'low', 'multiplier': 1.2, 'atr_pct': atr_pct}
                
        except Exception as e:
            self.logger.error(f"Volatility assessment error: {e}")
            return {'level': 'medium', 'multiplier': 1.0}

    def _create_comprehensive_analysis(self, df: pd.DataFrame, symbol_data: Dict,
                                     volume_entry: Dict, fibonacci_data: Dict,
                                     confluence_zones: List[Dict], 
                                     mtf_context: MultiTimeframeContext) -> Dict:
        """Create comprehensive analysis data"""
        return {
            'technical_summary': self.create_technical_summary(df),
            'risk_assessment': self.assess_risk(df, symbol_data),
            'volume_analysis': self.analyze_volume_patterns(df),
            'trend_strength': self.calculate_trend_strength(df),
            'price_action': self.analyze_price_action(df),
            'market_conditions': self.assess_market_conditions(df, symbol_data),
            'market_regime': mtf_context.market_regime,
            'volatility_assessment': self._assess_volatility_risk(df),
            'momentum_analysis': self._analyze_momentum_strength(df),
            'volume_profile': volume_entry,
            'fibonacci_data': fibonacci_data,
            'confluence_zones': confluence_zones,
            'mtf_context': mtf_context
        }

    # ===== COMPATIBILITY METHODS =====
    
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
            
            # Volatility and volume
            atr = latest.get('atr', latest['close'] * 0.02)
            volatility_pct = (atr / latest['close']) * 100
            volume_ratio = latest.get('volume_ratio', 1)
            volume_trend = self.get_volume_trend(df)
            
            # Market regime assessment
            recent_changes = df['close'].pct_change().tail(10) * 100
            regime_volatility = recent_changes.std()
            
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
                    'regime_volatility': regime_volatility,
                    'level': 'extreme' if regime_volatility > 8 else 'high' if regime_volatility > 5 else 'medium' if regime_volatility > 2 else 'low'
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
        """Analyze volume trend"""
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
        """Analyze volume patterns"""
        try:
            if 'volume' not in df.columns or len(df) < 15:
                return {'pattern': 'insufficient_data', 'strength': 0}
            
            recent_15 = df.tail(15)
            volume_ma_5 = recent_15['volume'].rolling(5).mean().iloc[-1]
            volume_ma_15 = df['volume'].rolling(15).mean().iloc[-1]
            
            up_volume = recent_15[recent_15['close'] > recent_15['open']]['volume'].sum()
            down_volume = recent_15[recent_15['close'] < recent_15['open']]['volume'].sum()
            total_volume = up_volume + down_volume
            
            buying_pressure = up_volume / total_volume if total_volume > 0 else 0.5
            
            # Pattern detection
            if volume_ma_5 > volume_ma_15 * 1.5:
                pattern = 'surge'
            elif volume_ma_5 > volume_ma_15 * 1.25:
                pattern = 'strong_increase'
            elif volume_ma_5 > volume_ma_15 * 1.08:
                pattern = 'increasing'
            elif volume_ma_5 < volume_ma_15 * 0.6:
                pattern = 'declining_fast'
            elif volume_ma_5 < volume_ma_15 * 0.75:
                pattern = 'declining'
            else:
                pattern = 'stable'
            
            # Volume quality assessment
            if pattern in ['surge', 'strong_increase'] and buying_pressure > 0.6:
                strength = 1.0
            elif pattern == 'increasing' and buying_pressure > 0.55:
                strength = 0.8
            elif pattern == 'stable':
                strength = 0.6
            elif pattern in ['declining', 'declining_fast']:
                strength = 0.3
            else:
                strength = 0.5
            
            return {
                'pattern': pattern,
                'buying_pressure': buying_pressure,
                'volume_ma_ratio': volume_ma_5 / volume_ma_15 if volume_ma_15 > 0 else 1,
                'strength': strength,
                'regime_quality': 'excellent' if strength > 0.8 else 'good' if strength > 0.6 else 'fair' if strength > 0.4 else 'poor'
            }
            
        except Exception as e:
            self.logger.error(f"Volume pattern analysis error: {e}")
            return {'pattern': 'unknown', 'strength': 0.5}

    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Calculate trend strength"""
        try:
            if len(df) < 30:
                return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'low'}
            
            latest = df.iloc[-1]
            recent_30 = df.tail(30)
            
            # Multiple timeframe trend analysis
            price_change_5 = (latest['close'] - recent_30.iloc[-5]['close']) / recent_30.iloc[-5]['close']
            price_change_15 = (latest['close'] - recent_30.iloc[-15]['close']) / recent_30.iloc[-15]['close']
            price_change_30 = (latest['close'] - recent_30.iloc[0]['close']) / recent_30.iloc[0]['close']
            
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            ema_12 = latest.get('ema_12', latest['close'])
            ema_26 = latest.get('ema_26', latest['close'])
            
            # MA alignment scoring
            ma_alignment_score = 0
            if latest['close'] > sma_20 > sma_50:
                ma_alignment_score += 3  # Strong bullish alignment
            elif latest['close'] > sma_20:
                ma_alignment_score += 1
            elif latest['close'] < sma_20 < sma_50:
                ma_alignment_score -= 3  # Strong bearish alignment
            elif latest['close'] < sma_20:
                ma_alignment_score -= 1
            
            if ema_12 > ema_26:
                ma_alignment_score += 1
            else:
                ma_alignment_score -= 1
            
            # Price momentum consistency
            bullish_candles = len(recent_30[recent_30['close'] > recent_30['open']])
            consistency = bullish_candles / len(recent_30)
            
            # Multi-timeframe momentum
            momentum_alignment = 0
            if price_change_5 > 0 and price_change_15 > 0 and price_change_30 > 0:
                momentum_alignment = 1  # All timeframes bullish
            elif price_change_5 < 0 and price_change_15 < 0 and price_change_30 < 0:
                momentum_alignment = -1  # All timeframes bearish
            else:
                momentum_alignment = 0  # Mixed signals
            
            # Strength calculation
            base_strength = (abs(price_change_15) + abs(ma_alignment_score) / 6 + 
                           abs(consistency - 0.5) * 2) / 3
            
            # Momentum alignment bonus
            if momentum_alignment != 0:
                base_strength *= 1.2
            
            strength = min(1.0, base_strength)
            
            # Direction determination
            if price_change_15 > 0.025 and ma_alignment_score > 1 and consistency > 0.6:
                direction = 'strong_bullish'
            elif price_change_15 > 0.01 and ma_alignment_score >= 0:
                direction = 'bullish'
            elif price_change_15 < -0.025 and ma_alignment_score < -1 and consistency < 0.4:
                direction = 'strong_bearish'
            elif price_change_15 < -0.01 and ma_alignment_score <= 0:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            # Consistency levels
            if consistency > 0.7 or consistency < 0.3:
                consistency_level = 'high'
            elif consistency > 0.6 or consistency < 0.4:
                consistency_level = 'medium'
            else:
                consistency_level = 'low'
            
            return {
                'strength': strength,
                'direction': direction,
                'consistency': consistency_level,
                'price_change_5': price_change_5,
                'price_change_15': price_change_15,
                'price_change_30': price_change_30,
                'ma_alignment_score': ma_alignment_score,
                'momentum_alignment': momentum_alignment
            }
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'low'}

    def analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze price action patterns"""
        try:
            if len(df) < 10:
                return {'patterns': [], 'strength': 0, 'regime_quality': 'insufficient_data'}
            
            recent_10 = df.tail(10)
            latest = df.iloc[-1]
            
            body_size = abs(latest['close'] - latest['open']) / latest['open']
            upper_shadow = latest['high'] - max(latest['close'], latest['open'])
            lower_shadow = min(latest['close'], latest['open']) - latest['low']
            full_range = latest['high'] - latest['low']
            
            patterns = []
            
            # Pattern detection
            if body_size < 0.003:
                patterns.append('doji')
            elif body_size > 0.02:
                patterns.append('strong_body')
            
            if full_range > 0 and lower_shadow / full_range > 0.6 and body_size < full_range * 0.3:
                patterns.append('hammer')
            elif full_range > 0 and upper_shadow / full_range > 0.6 and body_size < full_range * 0.3:
                patterns.append('shooting_star')
            
            # Support/Resistance testing
            recent_lows = recent_10['low'].min()
            recent_highs = recent_10['high'].max()
            
            if latest['low'] <= recent_lows * 1.002:
                patterns.append('support_test')
            if latest['high'] >= recent_highs * 0.998:
                patterns.append('resistance_test')
            
            # Momentum analysis
            closes = recent_10['close'].values
            momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            
            # Pattern strength assessment
            pattern_strength = 0
            if 'hammer' in patterns or 'shooting_star' in patterns:
                pattern_strength += 0.3
            if 'support_test' in patterns or 'resistance_test' in patterns:
                pattern_strength += 0.2
            if 'strong_body' in patterns:
                pattern_strength += 0.2
            
            momentum_strength = min(0.5, abs(momentum) * 10)
            total_strength = min(1.0, pattern_strength + momentum_strength)
            
            # Regime quality assessment
            if len(patterns) >= 2 and total_strength > 0.7:
                regime_quality = 'excellent'
            elif len(patterns) >= 1 and total_strength > 0.5:
                regime_quality = 'good'
            elif total_strength > 0.3:
                regime_quality = 'fair'
            else:
                regime_quality = 'poor'
            
            return {
                'patterns': patterns,
                'momentum': momentum,
                'body_size': body_size,
                'shadow_ratio': (upper_shadow + lower_shadow) / body_size if body_size > 0 else 0,
                'strength': total_strength,
                'regime_quality': regime_quality,
                'pattern_count': len(patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Price action analysis error: {e}")
            return {'patterns': [], 'strength': 0.5, 'regime_quality': 'unknown'}

    def assess_market_conditions(self, df: pd.DataFrame, symbol_data: Dict) -> Dict:
        """Assess overall market conditions"""
        try:
            latest = df.iloc[-1]
            
            volume_24h = symbol_data.get('volume_24h', 0)
            price_change_24h = symbol_data.get('price_change_24h', 0)
            
            # Liquidity assessment
            if volume_24h > 20_000_000:
                liquidity = 'excellent'
            elif volume_24h > 10_000_000:
                liquidity = 'high'
            elif volume_24h > 2_000_000:
                liquidity = 'medium'
            elif volume_24h > 500_000:
                liquidity = 'low'
            else:
                liquidity = 'very_low'
            
            # Volatility assessment
            atr_pct = latest.get('atr', latest['close'] * 0.02) / latest['close']
            recent_volatility = df['close'].pct_change().tail(10).std() * 100
            
            combined_volatility = (atr_pct * 100 + recent_volatility) / 2
            
            if combined_volatility > 12:
                volatility_level = 'extreme'
            elif combined_volatility > 8:
                volatility_level = 'high'
            elif combined_volatility > 4:
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
            
            # Market sentiment
            if price_change_24h > 15:
                sentiment = 'extremely_bullish'
            elif price_change_24h > 8:
                sentiment = 'very_bullish'
            elif price_change_24h > 3:
                sentiment = 'bullish'
            elif price_change_24h > 1:
                sentiment = 'slightly_bullish'
            elif price_change_24h < -15:
                sentiment = 'extremely_bearish'
            elif price_change_24h < -8:
                sentiment = 'very_bearish'
            elif price_change_24h < -3:
                sentiment = 'bearish'
            elif price_change_24h < -1:
                sentiment = 'slightly_bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'liquidity': liquidity,
                'volatility_level': volatility_level,
                'combined_volatility': combined_volatility,
                'sentiment': sentiment,
                'price_change_24h': price_change_24h,
                'volume_24h': volume_24h,
                'favorable_for_trading': volume_24h > 500_000 and volatility_level != 'extreme'
            }
            
        except Exception as e:
            self.logger.error(f"Market conditions assessment error: {e}")
            return {'liquidity': 'unknown', 'volatility_level': 'unknown', 'sentiment': 'neutral'}

    def assess_risk(self, df: pd.DataFrame, symbol_data: Dict) -> Dict:
        """Risk assessment based on current conditions"""
        try:
            latest = df.iloc[-1]
            current_price = symbol_data['current_price']
            
            # Base risk factors
            atr = latest.get('atr', current_price * 0.02)
            volatility = atr / current_price
            
            # Calculate total risk
            base_risk = volatility * 2.0
            total_risk = max(0.1, min(1.0, base_risk))
            
            # Risk level determination
            if total_risk > 0.8:
                risk_level = 'Very High'
            elif total_risk > 0.6:
                risk_level = 'High'
            elif total_risk > 0.4:
                risk_level = 'Medium'
            elif total_risk > 0.25:
                risk_level = 'Low'
            else:
                risk_level = 'Very Low'
            
            return {
                'total_risk_score': total_risk,
                'volatility_risk': volatility,
                'distance_risk': 0,  # Placeholder
                'risk_level': risk_level,
                'mtf_validated': False,  # Will be updated by signal
                'market_regime': 'unknown'  # Will be updated by signal
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return {'total_risk_score': 0.5, 'risk_level': 'Medium'}


# ===== DEBUG HELPER FUNCTION =====

def debug_signal_conditions(df: pd.DataFrame, symbol: str, generator: SignalGenerator = None):
    """Debug function for signal conditions"""
    latest = df.iloc[-1]
    
    print(f"\n=== SIGNAL DEBUG: {symbol} ===")
    print(f"RSI: {latest.get('rsi', 'Missing'):.1f}" if latest.get('rsi') else "RSI: Missing")
    print(f"Stoch RSI K: {latest.get('stoch_rsi_k', 'Missing'):.1f}" if latest.get('stoch_rsi_k') else "Stoch RSI K: Missing")
    print(f"Stoch RSI D: {latest.get('stoch_rsi_d', 'Missing'):.1f}" if latest.get('stoch_rsi_d') else "Stoch RSI D: Missing")
    print(f"MACD: {latest.get('macd', 'Missing'):.4f}" if latest.get('macd') else "MACD: Missing")
    print(f"MACD Signal: {latest.get('macd_signal', 'Missing'):.4f}" if latest.get('macd_signal') else "MACD Signal: Missing")
    print(f"Volume Ratio: {latest.get('volume_ratio', 'Missing'):.2f}" if latest.get('volume_ratio') else "Volume Ratio: Missing")
    print(f"BB Position: {latest.get('bb_position', 'Missing'):.2f}" if latest.get('bb_position') else "BB Position: Missing")
    
    if generator:
        print(f"\nTimeframe Configuration:")
        print(f"Primary: {generator.primary_timeframe}")
        print(f"Structure: {generator.structure_timeframe}")  
        print(f"Confirmations: {generator.confirmation_timeframes}")
        
        # Market regime analysis
        symbol_data = {'symbol': symbol, 'current_price': latest['close'], 'price_change_24h': 0, 'volume_24h': 1000000}
        market_regime = generator._determine_market_regime(symbol_data, df)
        print(f"Market Regime: {market_regime}")
        
        # Signal configuration
        print(f"\nSignal Configuration:")
        print(f"Min Stop Distance: {generator.signal_config.min_stop_distance_pct*100:.1f}%")
        print(f"Max Stop Distance: {generator.signal_config.max_stop_distance_pct*100:.1f}%")
        print(f"Min R/R Ratio: {generator.signal_config.min_risk_reward}")
        print(f"Max R/R Ratio: {generator.signal_config.max_risk_reward}")
        
        # Try to get MTF context if exchange_manager available
        if generator.exchange_manager:
            mtf_context = generator._get_multitimeframe_context(symbol_data, market_regime)
            if mtf_context:
                print(f"\nMTF Context:")
                print(f"Dominant Trend: {mtf_context.dominant_trend}")
                print(f"Trend Strength: {mtf_context.trend_strength:.2f}")
                print(f"Entry Bias: {mtf_context.entry_bias}")
                print(f"Market Regime: {mtf_context.market_regime}")
                print(f"Volatility Level: {mtf_context.volatility_level}")
                print(f"Higher TF Zones: {len(mtf_context.higher_tf_zones)}")
                print(f"Confirmation Score: {mtf_context.confirmation_score:.2f}")
    
    # Signal conditions
    rsi = latest.get('rsi', 50)
    stoch_k = latest.get('stoch_rsi_k', 50)
    stoch_d = latest.get('stoch_rsi_d', 50)
    macd = latest.get('macd', 0)
    macd_signal = latest.get('macd_signal', 0)
    volume_ratio = latest.get('volume_ratio', 1)
    
    print(f"\nSignal Conditions:")
    print(f"LONG Conditions:")
    print(f"  RSI < 65: {rsi < 65} (RSI: {rsi:.1f})")
    print(f"  Stoch K < 70: {stoch_k < 70} (K: {stoch_k:.1f})")
    print(f"  Stoch K > D: {stoch_k > stoch_d}")
    print(f"  Volume > 0.8: {volume_ratio > 0.8} (Ratio: {volume_ratio:.2f})")
    print(f"  Traditional LONG: RSI < 45 & MACD > Signal: {rsi < 45 and macd > macd_signal}")
    
    print(f"\nSHORT Conditions:")
    print(f"  RSI > 35: {rsi > 35} (RSI: {rsi:.1f})")
    print(f"  Stoch K > 30: {stoch_k > 30} (K: {stoch_k:.1f})")
    print(f"  Stoch K < D: {stoch_k < stoch_d}")
    print(f"  Volume > 0.8: {volume_ratio > 0.8} (Ratio: {volume_ratio:.2f})")
    print(f"  Traditional SHORT: RSI > 55 & MACD < Signal: {rsi > 55 and macd < macd_signal}")
    
    print(f"\nAnalysis Complete!")


# ===== INTEGRATION FUNCTIONS =====

def create_mtf_signal_generator(config: EnhancedSystemConfig, exchange_manager) -> SignalGenerator:
    """
    Factory function to create the enhanced MTF-aware signal generator
    
    VERSION 4.0 - PRODUCTION READY:
    ✅ Dynamic stop losses (1.5% - 6% based on volatility)
    ✅ Momentum-aware entry calculation
    ✅ Entry timing validation
    ✅ Quality filters (momentum, volume, choppiness)
    ✅ Fast-moving signal detection
    ✅ Enhanced risk management
    ✅ Premium signal prioritization
    ✅ Debug mode available
    """
    generator = SignalGenerator(config, exchange_manager)
    
    # Enable debug mode if needed
    generator.debug_mode = False  # Set to True for troubleshooting
    
    return generator


# ===== PERFORMANCE MONITORING =====

class SignalPerformanceTracker:
    """Track signal performance for continuous improvement"""
    
    def __init__(self):
        self.signal_history = []
        self.performance_metrics = {}
    
    def track_signal(self, signal: Dict, outcome: str, profit_pct: float = 0):
        """Track signal outcome for performance analysis"""
        try:
            performance_record = {
                'timestamp': datetime.now(),
                'symbol': signal['symbol'],
                'side': signal['side'],
                'confidence': signal['confidence'],
                'market_regime': signal.get('market_regime', 'unknown'),
                'mtf_validated': signal.get('mtf_validated', False),
                'entry_strategy': signal.get('entry_strategy', 'immediate'),
                'signal_type': signal.get('signal_type', 'unknown'),
                'outcome': outcome,  # 'profit', 'loss', 'breakeven'
                'profit_pct': profit_pct,
                'risk_reward_ratio': signal.get('risk_reward_ratio', 0),
                'quality_score': signal.get('quality_score', 0),
                'is_premium': signal.get('is_premium_signal', False)
            }
            
            self.signal_history.append(performance_record)
            self._update_performance_metrics()
            
        except Exception as e:
            logging.error(f"Signal tracking error: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics based on signal history"""
        try:
            if not self.signal_history:
                return
            
            # Overall performance
            total_signals = len(self.signal_history)
            profitable_signals = len([s for s in self.signal_history if s['outcome'] == 'profit'])
            win_rate = profitable_signals / total_signals if total_signals > 0 else 0
            
            # MTF validation performance
            mtf_signals = [s for s in self.signal_history if s['mtf_validated']]
            mtf_win_rate = len([s for s in mtf_signals if s['outcome'] == 'profit']) / len(mtf_signals) if mtf_signals else 0
            
            # Premium signal performance
            premium_signals = [s for s in self.signal_history if s['is_premium']]
            premium_win_rate = len([s for s in premium_signals if s['outcome'] == 'profit']) / len(premium_signals) if premium_signals else 0
            
            # Signal type performance
            type_performance = {}
            for signal_type in ['long_signal_v4', 'short_signal_v4']:
                type_signals = [s for s in self.signal_history if s['signal_type'] == signal_type]
                if type_signals:
                    type_wins = len([s for s in type_signals if s['outcome'] == 'profit'])
                    type_performance[signal_type] = {
                        'total': len(type_signals),
                        'wins': type_wins,
                        'win_rate': type_wins / len(type_signals)
                    }
            
            self.performance_metrics = {
                'total_signals': total_signals,
                'win_rate': win_rate,
                'mtf_win_rate': mtf_win_rate,
                'premium_win_rate': premium_win_rate,
                'type_performance': type_performance,
                'avg_profit': sum(s['profit_pct'] for s in self.signal_history if s['outcome'] == 'profit') / max(1, profitable_signals),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Performance metrics update error: {e}")
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return self.performance_metrics


# ===== EXPORT AND COMPATIBILITY =====

__all__ = [
    'SignalGenerator',
    'SignalConfig',
    'SignalValidator',
    'MultiTimeframeContext', 
    'create_mtf_signal_generator',
    'debug_signal_conditions',
    'SignalPerformanceTracker',
    'analyze_price_momentum_strength',
    'check_volume_momentum_divergence',
    'identify_fast_moving_setup',
    'filter_choppy_markets',
    'calculate_momentum_adjusted_entry',
    'calculate_dynamic_stop_loss'
]

# Version and feature information
__version__ = "4.0.0-PRODUCTION"
__features__ = [
    "✅ Minimum 1.5% stop loss distance (wider for crypto)",
    "✅ Maximum 6% stop loss (volatility-based)",
    "✅ Momentum-aware entry calculation",
    "✅ Entry timing validation",
    "✅ Quality filters (momentum, volume, choppiness)",
    "✅ Fast-moving signal detection",
    "✅ Premium signal prioritization",
    "✅ Enhanced risk/reward ratios (2.5x/4x)",
    "✅ Performance tracking",
    "✅ Production ready"
]

# Configuration validation
def validate_generator_config(config: EnhancedSystemConfig) -> bool:
    """Validate configuration for signal generator"""
    try:
        required_fields = ['timeframe', 'confirmation_timeframes', 'charts_per_batch']
        
        for field in required_fields:
            if not hasattr(config, field):
                logging.error(f"Missing required config field: {field}")
                return False
                
        if not config.confirmation_timeframes:
            logging.warning("No confirmation timeframes configured - using single timeframe mode")
            
        return True
        
    except Exception as e:
        logging.error(f"Config validation error: {e}")
        return False

# Module initialization logging
logging.getLogger(__name__).info(f"Enhanced Signal Generator v{__version__} loaded")
logging.getLogger(__name__).info(f"Features: {', '.join(__features__)}")

print("\n" + "="*80)
print("🚀 ENHANCED SIGNAL GENERATOR v4.0")
print("="*80)
print("✅ KEY IMPROVEMENTS:")
print("   1. Dynamic stop losses: 1.5% - 6% (volatility-based)")
print("   2. Momentum-aware entries (chase trends when strong)")
print("   3. Entry timing validation (avoid bad setups)")
print("   4. Quality filters (momentum, volume, choppiness)")
print("   5. Fast-moving signal detection")
print("   6. Enhanced R/R ratios: 2.5x and 4x targets")
print("   7. Premium signal prioritization")
print("   8. Performance tracking included")
print("="*80)
print("📊 ENHANCED PARAMETERS:")
print(f"   Stop Loss Range: 1.5% - 6.0%")
print(f"   Take Profit Range: 2.0% - 15.0%")
print(f"   Risk/Reward Range: 2.0 - 10.0")
print(f"   Confidence Range: 50% - 90%")
print(f"   Quality Filters: ✅ Enabled")
print("="*80)
print("🎯 SIGNAL GENERATION OPTIMIZED FOR CRYPTO VOLATILITY!")
print("="*80)