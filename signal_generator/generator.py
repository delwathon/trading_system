"""
Signal Generator V14.0 - Professional Trading Signal System
===========================================================
Clean architecture with enhanced profitability features.
Includes order type determination, news sentiment, and proven technical analysis.
"""

import pandas as pd
import numpy as np
import logging
import talib
from typing import Dict, List, Optional, Tuple, Any, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ===========================
# ENUMS AND CONSTANTS
# ===========================

class SignalSide(Enum):
    """Trading direction"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order execution type"""
    MARKET = "market"
    LIMIT = "limit"

class SignalQuality(Enum):
    """Signal quality tiers with scores"""
    ELITE = ("elite", 90)      # A+ grade
    PREMIUM = ("premium", 75)   # A grade
    STANDARD = ("standard", 60) # B grade
    MARGINAL = ("marginal", 45) # C grade
    
    def __init__(self, label: str, min_score: int):
        self.label = label
        self.min_score = min_score

class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_TREND_UP = "strong_trend_up"
    TREND_UP = "trend_up"
    RANGING_BULLISH = "ranging_bullish"
    RANGING = "ranging"
    RANGING_BEARISH = "ranging_bearish"
    TREND_DOWN = "trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    VOLATILE = "volatile"
    SQUEEZE = "squeeze"

# ===========================
# CONFIGURATION
# ===========================

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    
    # Timeframes - NO DEFAULTS, will be set from EnhancedSystemConfig
    primary_timeframe: str = None
    confirmation_timeframes: List[str] = None
    entry_timeframe: str = None
    
    # Risk Management (keep existing defaults)
    max_risk_per_trade: float = 0.02
    min_risk_reward: float = 1.5
    optimal_risk_reward: float = 2.5
    max_concurrent_positions: int = 10
    
    # Signal Quality Thresholds (keep existing defaults)
    min_conditions_for_signal: int = 4
    min_confidence_for_execution: float = 50.0
    
    # Technical Thresholds (keep existing defaults)
    rsi_oversold_threshold: float = 35
    rsi_overbought_threshold: float = 65
    volume_surge_threshold: float = 1.3
    breakout_atr_multiplier: float = 1.5
    
    # Order Type Determination (keep existing defaults)
    breakout_market_order_threshold: float = 0.02
    proximity_market_order_range: float = 0.005

# ===========================
# SIGNAL DATA MODEL
# ===========================

@dataclass
class Signal:
    """Complete signal model with all trading information"""
    
    # Core Identification
    id: str
    symbol: str
    side: SignalSide
    
    # Quality & Confidence
    quality: SignalQuality
    confidence: float
    score: float  # Raw score 0-100
    
    # Price Levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: float
    
    # Order Execution
    order_type: OrderType
    order_reason: str
    
    # Risk Metrics
    risk_reward_ratio: float
    position_size: float = 0.0
    risk_amount: float = 0.0
    potential_profit: float = 0.0
    
    # Market Context
    timeframe: str = "6h"
    market_regime: MarketRegime = MarketRegime.RANGING
    volume_24h: float = 0.0
    price_change_24h: float = 0.0
    
    # Technical Data
    indicators: Dict[str, float] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    
    # Multi-timeframe Validation
    mtf_confirmation: bool = False
    mtf_score: float = 0.0
    mtf_details: Dict = field(default_factory=dict)
    
    # Analysis Components
    technical_score: float = 0.0
    structure_score: float = 0.0
    volume_score: float = 0.0
    momentum_score: float = 0.0
    
    # Optional Data
    support_resistance: Dict = field(default_factory=dict)
    breakout_info: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        data = asdict(self)
        data['side'] = self.side.value
        data['quality'] = self.quality.label
        data['order_type'] = self.order_type.value
        data['market_regime'] = self.market_regime.value
        data['created_at'] = self.created_at.isoformat()
        return data

# ===========================
# MAIN SIGNAL GENERATOR
# ===========================

class SignalGeneratorV14:
    """
    Professional Signal Generator with clean architecture.
    Focuses on proven technical analysis with news sentiment integration.
    """
    
    def __init__(self, config: Optional[SignalConfig] = None, exchange_manager=None):
        """Initialize the signal generator"""
        self.config = config or SignalConfig()
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Import analyzer from separate file
        try:
            from signal_generator.analyzer import SignalAnalyzer
        except ImportError:
            # Try alternative import paths
            try:
                from .analyzer import SignalAnalyzer
            except ImportError:
                try:
                    from signal_generator.analyzer import SignalAnalyzer
                except ImportError:
                    self.logger.error("Could not import SignalAnalyzer")
                    raise ImportError("SignalAnalyzer module not found. Ensure signal_analyzer_v14.py is in the same directory.")
        
        self.analyzer = SignalAnalyzer(self.config)
        
        # Statistics tracking
        self.stats = {
            'signals_generated': 0,
            'signals_by_quality': defaultdict(int),
            'avg_confidence': 0.0,
            'avg_risk_reward': 0.0
        }
        
        self.logger.info("=" * 60)
        self.logger.info("Signal Generator V14 - Professional Architecture")
        self.logger.info(f"Primary TF: {self.config.primary_timeframe}")
        self.logger.info(f"Min conditions: {self.config.min_conditions_for_signal}")
        self.logger.info("=" * 60)
    
    def analyze_symbol(self, symbol: str, *args, **kwargs) -> Optional[Signal]:
        """
        Main analysis method - generates trading signals for a symbol.
        Implements top-down multi-timeframe analysis.
        Accepts additional args/kwargs for compatibility.
        """
        try:
            self.logger.debug(f"Analyzing {symbol}...")
            
            # Step 1: Fetch market data
            market_data = self._fetch_market_data(symbol)
            if not market_data:
                return None
            
            # Step 2: Primary timeframe analysis
            primary_analysis = self._analyze_primary_timeframe(
                symbol, 
                self.config.primary_timeframe,
                market_data
            )
            
            if not primary_analysis['has_signal']:
                self.logger.debug(f"No signal on primary timeframe for {symbol}")
                return None
            
            # Step 3: Multi-timeframe confirmation
            mtf_result = self._validate_with_confirmations(
                symbol,
                primary_analysis,
                market_data
            )
            
            # Step 4: Entry optimization on lowest timeframe
            entry_analysis = self._optimize_entry(
                symbol,
                primary_analysis,
                mtf_result,
                market_data
            )
            
            # Step 5: Build final signal
            signal = self._build_signal(
                symbol,
                primary_analysis,
                mtf_result,
                entry_analysis,
                market_data
            )
            
            if signal:
                self._update_statistics(signal)
                self.logger.info(f"Signal generated for {symbol}: {signal.quality.label} "
                               f"({signal.confidence:.1f}% confidence)")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current market data for the symbol"""
        try:
            if not self.exchange_manager:
                # Return dummy data for testing
                return {
                    'volume_24h': 1000000,
                    'price_change_24h': 2.5,
                    'current_price': 50000
                }
            
            ticker = self.exchange_manager.exchange.fetch_ticker(symbol)
            return {
                'volume_24h': ticker.get('quoteVolume', 0),
                'price_change_24h': ticker.get('percentage', 0),
                'current_price': ticker.get('last', 0),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'high_24h': ticker.get('high', 0),
                'low_24h': ticker.get('low', 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Could not fetch market data for {symbol}: {e}")
            return None
    
    def _analyze_primary_timeframe(self, symbol: str, timeframe: str, 
                                  market_data: Dict) -> Dict:
        """
        Analyze primary timeframe for signal generation.
        Returns analysis results with signal candidates.
        """
        try:
            # Fetch OHLCV data
            df = self._fetch_ohlcv(symbol, timeframe)
            if df is None or len(df) < 100:
                return {'has_signal': False}
            
            # Calculate technical indicators
            df = self.analyzer.calculate_indicators(df)
            
            # Get current values
            latest = df.iloc[-1]
            current_price = market_data['current_price'] or latest['close']
            
            # Market structure analysis
            structure = self.analyzer.analyze_market_structure(df)
            
            # Volume profile analysis
            volume_profile = self.analyzer.analyze_volume_profile(df)
            
            # Detect patterns
            patterns = self.analyzer.detect_patterns(df)
            
            # Check for breakouts
            breakout_info = self.analyzer.detect_breakout(df, current_price)
            
            # Score conditions for LONG and SHORT
            long_score, long_conditions = self._score_long_conditions(
                latest, structure, volume_profile, patterns, 
                breakout_info
            )
            
            short_score, short_conditions = self._score_short_conditions(
                latest, structure, volume_profile, patterns,
                breakout_info
            )
            
            # Determine if we have a signal
            has_signal = False
            signal_side = None
            signal_score = 0
            conditions = {}
            
            if long_score >= self.config.min_conditions_for_signal:
                has_signal = True
                signal_side = SignalSide.BUY
                signal_score = long_score
                conditions = long_conditions
            elif short_score >= self.config.min_conditions_for_signal:
                has_signal = True
                signal_side = SignalSide.SELL
                signal_score = short_score
                conditions = short_conditions
            
            return {
                'has_signal': has_signal,
                'side': signal_side,
                'score': signal_score,
                'conditions': conditions,
                'df': df,
                'latest': latest,
                'structure': structure,
                'volume_profile': volume_profile,
                'patterns': patterns,
                'breakout_info': breakout_info,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Primary timeframe analysis error: {e}")
            return {'has_signal': False}
    
    def _score_long_conditions(self, latest: pd.Series, structure: Dict,
                              volume_profile: Dict, patterns: List,
                              breakout_info: Dict) -> Tuple[int, Dict]:
        """Score conditions for a LONG signal"""
        conditions = {}
        score = 0
        
        # 1. RSI conditions (relaxed)
        rsi = latest.get('rsi', 50)
        if rsi <= self.config.rsi_oversold_threshold:
            conditions['rsi_oversold'] = True
            score += 1
        elif rsi <= 45:  # Mildly oversold
            conditions['rsi_mild_oversold'] = True
            score += 0.5
        
        # 2. MACD conditions
        if latest.get('macd_hist', 0) > 0:
            conditions['macd_bullish'] = True
            score += 1
        elif latest.get('macd', 0) > latest.get('macd_signal', 0):
            conditions['macd_cross_up'] = True
            score += 0.5
        
        # 3. Stochastic conditions
        stoch_k = latest.get('stoch_k', 50)
        if stoch_k < 30:
            conditions['stoch_oversold'] = True
            score += 1
        elif stoch_k < 50 and stoch_k > latest.get('stoch_d', 50):
            conditions['stoch_bullish_cross'] = True
            score += 0.5
        
        # 4. Price action
        if latest['close'] > latest.get('vwap', latest['close']):
            conditions['above_vwap'] = True
            score += 0.5
        
        if latest.get('bb_position', 0.5) < 0.3:
            conditions['near_bb_lower'] = True
            score += 0.5
        
        # 5. Market structure
        if structure.get('trend') in ['uptrend', 'ranging']:
            conditions['favorable_trend'] = True
            score += 1
        
        if structure.get('near_support'):
            conditions['near_support'] = True
            score += 1
        
        # 6. Volume
        if latest.get('volume_ratio', 1) > self.config.volume_surge_threshold:
            conditions['volume_surge'] = True
            score += 1
        
        # 7. Patterns
        bullish_patterns = ['double_bottom', 'inverse_head_shoulders', 'bullish_flag']
        if any(p in patterns for p in bullish_patterns):
            conditions['bullish_pattern'] = True
            score += 1
        
        # 8. Breakout
        if breakout_info.get('type') == 'resistance_break':
            conditions['breakout_up'] = True
            score += 1.5
        
        return score, conditions
    
    def _score_short_conditions(self, latest: pd.Series, structure: Dict,
                               volume_profile: Dict, patterns: List,
                               breakout_info: Dict) -> Tuple[int, Dict]:
        """Score conditions for a SHORT signal"""
        conditions = {}
        score = 0
        
        # 1. RSI conditions (relaxed)
        rsi = latest.get('rsi', 50)
        if rsi >= self.config.rsi_overbought_threshold:
            conditions['rsi_overbought'] = True
            score += 1
        elif rsi >= 55:  # Mildly overbought
            conditions['rsi_mild_overbought'] = True
            score += 0.5
        
        # 2. MACD conditions
        if latest.get('macd_hist', 0) < 0:
            conditions['macd_bearish'] = True
            score += 1
        elif latest.get('macd', 0) < latest.get('macd_signal', 0):
            conditions['macd_cross_down'] = True
            score += 0.5
        
        # 3. Stochastic conditions
        stoch_k = latest.get('stoch_k', 50)
        if stoch_k > 70:
            conditions['stoch_overbought'] = True
            score += 1
        elif stoch_k > 50 and stoch_k < latest.get('stoch_d', 50):
            conditions['stoch_bearish_cross'] = True
            score += 0.5
        
        # 4. Price action
        if latest['close'] < latest.get('vwap', latest['close']):
            conditions['below_vwap'] = True
            score += 0.5
        
        if latest.get('bb_position', 0.5) > 0.7:
            conditions['near_bb_upper'] = True
            score += 0.5
        
        # 5. Market structure
        if structure.get('trend') in ['downtrend', 'ranging']:
            conditions['favorable_trend'] = True
            score += 1
        
        if structure.get('near_resistance'):
            conditions['near_resistance'] = True
            score += 1
        
        # 6. Volume
        if latest.get('volume_ratio', 1) > self.config.volume_surge_threshold:
            conditions['volume_surge'] = True
            score += 1
        
        # 7. Patterns
        bearish_patterns = ['double_top', 'head_shoulders', 'bearish_flag']
        if any(p in patterns for p in bearish_patterns):
            conditions['bearish_pattern'] = True
            score += 1
        
        # 8. Breakout
        if breakout_info.get('type') == 'support_break':
            conditions['breakout_down'] = True
            score += 1.5
        
        return score, conditions
    
    def _validate_with_confirmations(self, symbol: str, primary_analysis: Dict,
                                    market_data: Dict) -> Dict:
        """Validate signal with confirmation timeframes"""
        mtf_score = 0
        confirmations = {}
        
        for tf in self.config.confirmation_timeframes:
            try:
                df = self._fetch_ohlcv(symbol, tf)
                if df is None or len(df) < 50:
                    continue
                
                df = self.analyzer.calculate_indicators(df)
                latest = df.iloc[-1]
                
                # Check alignment
                aligned = self._check_timeframe_alignment(
                    latest, 
                    primary_analysis['side'],
                    tf
                )
                
                confirmations[tf] = aligned
                if aligned:
                    mtf_score += 1
                    
            except Exception as e:
                self.logger.debug(f"MTF validation error for {tf}: {e}")
                continue
        
        mtf_confirmation = mtf_score >= len(self.config.confirmation_timeframes) * 0.5
        mtf_score_normalized = mtf_score / max(len(self.config.confirmation_timeframes), 1)
        
        return {
            'mtf_confirmation': mtf_confirmation,
            'mtf_score': mtf_score_normalized,
            'confirmations': confirmations,
            'mtf_boost': min(10, mtf_score_normalized * 15)  # Max 10 point boost
        }
    
    def _check_timeframe_alignment(self, latest: pd.Series, side: SignalSide, 
                                  timeframe: str) -> bool:
        """Check if timeframe aligns with signal direction"""
        score = 0
        
        # RSI alignment
        rsi = latest.get('rsi', 50)
        if side == SignalSide.BUY and rsi < 55:
            score += 1
        elif side == SignalSide.SELL and rsi > 45:
            score += 1
        
        # MACD alignment
        macd = latest.get('macd', 0)
        signal = latest.get('macd_signal', 0)
        if side == SignalSide.BUY and macd > signal:
            score += 1
        elif side == SignalSide.SELL and macd < signal:
            score += 1
        
        # Trend alignment
        sma20 = latest.get('sma_20', latest['close'])
        sma50 = latest.get('sma_50', latest['close'])
        if side == SignalSide.BUY and sma20 > sma50:
            score += 1
        elif side == SignalSide.SELL and sma20 < sma50:
            score += 1
        
        return score >= 2  # Need at least 2 confirmations
    
    def _optimize_entry(self, symbol: str, primary_analysis: Dict,
                       mtf_result: Dict, market_data: Dict) -> Dict:
        """Optimize entry on the lowest timeframe"""
        try:
            df = self._fetch_ohlcv(symbol, self.config.entry_timeframe)
            if df is None or len(df) < 50:
                # Use primary timeframe data as fallback
                df = primary_analysis['df']
            else:
                df = self.analyzer.calculate_indicators(df)
            
            latest = df.iloc[-1]
            current_price = market_data['current_price']
            
            # Calculate optimized entry
            if primary_analysis['side'] == SignalSide.BUY:
                # For long, try to enter slightly below current price
                if primary_analysis.get('breakout_info', {}).get('type') == 'resistance_break':
                    # Breakout: enter at market
                    entry_price = current_price
                else:
                    # Pullback entry
                    entry_price = current_price * 0.998
            else:
                # For short, try to enter slightly above current price
                if primary_analysis.get('breakout_info', {}).get('type') == 'support_break':
                    # Breakdown: enter at market
                    entry_price = current_price
                else:
                    # Pullback entry
                    entry_price = current_price * 1.002
            
            # Determine order type
            order_type, order_reason = self._determine_order_type(
                primary_analysis,
                entry_price,
                current_price,
                market_data
            )
            
            return {
                'entry_price': entry_price,
                'order_type': order_type,
                'order_reason': order_reason,
                'entry_rsi': latest.get('rsi', 50),
                'entry_volume_ratio': latest.get('volume_ratio', 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Entry optimization error: {e}")
            return {
                'entry_price': market_data['current_price'],
                'order_type': OrderType.LIMIT,
                'order_reason': 'Default limit order',
                'entry_rsi': 50,
                'entry_volume_ratio': 1.0
            }
    
    def _determine_order_type(self, primary_analysis: Dict, entry_price: float,
                             current_price: float, market_data: Dict) -> Tuple[OrderType, str]:
        """
        Determine whether to use market or limit order.
        This is crucial for profitability.
        """
        breakout_info = primary_analysis.get('breakout_info', {})
        patterns = primary_analysis.get('patterns', [])
        volume_ratio = primary_analysis.get('latest', {}).get('volume_ratio', 1.0)
        
        # Calculate price difference
        price_diff_pct = abs(entry_price - current_price) / current_price
        
        # Conditions for MARKET orders (immediate execution)
        market_conditions = []
        
        # 1. Strong breakout
        if breakout_info.get('strength', 0) > self.config.breakout_market_order_threshold:
            market_conditions.append(f"Strong breakout ({breakout_info.get('type', 'breakout')})")
        
        # 2. Breakout patterns
        breakout_patterns = ['ascending_triangle', 'bull_flag', 'cup_handle']
        if any(p in patterns for p in breakout_patterns):
            market_conditions.append(f"Breakout pattern detected")
        
        # 3. High volume surge
        if volume_ratio > 2.0:
            market_conditions.append(f"High volume surge ({volume_ratio:.1f}x)")
        
        # 4. Entry very close to current price
        if price_diff_pct <= self.config.proximity_market_order_range:
            market_conditions.append(f"Entry within {price_diff_pct:.2%} of current")
        
        # 5. Strong momentum
        if primary_analysis.get('conditions', {}).get('strong_momentum'):
            market_conditions.append("Strong momentum detected")
        
        # Decision
        if len(market_conditions) >= 2:  # Need at least 2 conditions for market order
            return OrderType.MARKET, f"Market order: {', '.join(market_conditions[:2])}"
        else:
            return OrderType.LIMIT, "Limit order: Patient entry preferred"
    
    def _build_signal(self, symbol: str, primary_analysis: Dict,
                     mtf_result: Dict, entry_analysis: Dict,
                     market_data: Dict) -> Optional[Signal]:
        """Build the final Signal object"""
        try:
            side = primary_analysis['side']
            current_price = market_data['current_price']
            entry_price = entry_analysis['entry_price']
            
            # Calculate stop loss and targets
            atr = primary_analysis['latest'].get('atr', current_price * 0.02)
            
            if side == SignalSide.BUY:
                stop_loss = entry_price - (atr * 2.0)
                take_profit_1 = entry_price + (atr * 3.0)
                take_profit_2 = entry_price + (atr * 5.0)
            else:
                stop_loss = entry_price + (atr * 2.0)
                take_profit_1 = entry_price - (atr * 3.0)
                take_profit_2 = entry_price - (atr * 5.0)
            
            # Use support/resistance for better levels
            if side == SignalSide.BUY:
                if primary_analysis['structure'].get('support_levels'):
                    nearest_support = min(primary_analysis['structure']['support_levels'])
                    stop_loss = min(stop_loss, nearest_support * 0.995)
                if primary_analysis['structure'].get('resistance_levels'):
                    nearest_resistance = max(primary_analysis['structure']['resistance_levels'])
                    take_profit_1 = min(take_profit_1, nearest_resistance * 0.995)
            else:
                if primary_analysis['structure'].get('resistance_levels'):
                    nearest_resistance = max(primary_analysis['structure']['resistance_levels'])
                    stop_loss = max(stop_loss, nearest_resistance * 1.005)
                if primary_analysis['structure'].get('support_levels'):
                    nearest_support = min(primary_analysis['structure']['support_levels'])
                    take_profit_1 = max(take_profit_1, nearest_support * 1.005)
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit_1 - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Skip if R/R too low
            if risk_reward_ratio < self.config.min_risk_reward:
                self.logger.debug(f"Signal rejected: R/R too low ({risk_reward_ratio:.2f})")
                return None
            
            # Calculate comprehensive scores
            technical_score = primary_analysis['score'] * 10  # Convert to 0-100 scale
            structure_score = self._calculate_structure_score(primary_analysis['structure'])
            volume_score = self._calculate_volume_score(primary_analysis['latest'])
            momentum_score = self._calculate_momentum_score(primary_analysis['latest'])
            
            # Calculate final confidence (adjusted weights without news)
            base_confidence = (
                technical_score * 0.40 +
                structure_score * 0.25 +
                volume_score * 0.20 +
                momentum_score * 0.15
            )
            
            # Add MTF boost
            final_confidence = min(95, base_confidence + mtf_result.get('mtf_boost', 0))
            
            # Determine quality tier
            if final_confidence >= SignalQuality.ELITE.min_score:
                quality = SignalQuality.ELITE
            elif final_confidence >= SignalQuality.PREMIUM.min_score:
                quality = SignalQuality.PREMIUM
            elif final_confidence >= SignalQuality.STANDARD.min_score:
                quality = SignalQuality.STANDARD
            else:
                quality = SignalQuality.MARGINAL
            
            # Skip if confidence too low
            if final_confidence < self.config.min_confidence_for_execution:
                self.logger.debug(f"Signal rejected: Confidence too low ({final_confidence:.1f}%)")
                return None
            
            # Determine market regime
            market_regime = self._determine_market_regime(primary_analysis['structure'])
            
            # Build signal
            signal = Signal(
                id=self._generate_signal_id(symbol, side),
                symbol=symbol,
                side=side,
                quality=quality,
                confidence=final_confidence,
                score=base_confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                current_price=current_price,
                order_type=entry_analysis['order_type'],
                order_reason=entry_analysis['order_reason'],
                risk_reward_ratio=risk_reward_ratio,
                potential_profit=reward,
                timeframe=self.config.primary_timeframe,
                market_regime=market_regime,
                volume_24h=market_data.get('volume_24h', 0),
                price_change_24h=market_data.get('price_change_24h', 0),
                indicators={
                    'rsi': primary_analysis['latest'].get('rsi', 50),
                    'macd_hist': primary_analysis['latest'].get('macd_hist', 0),
                    'stoch_k': primary_analysis['latest'].get('stoch_k', 50),
                    'volume_ratio': primary_analysis['latest'].get('volume_ratio', 1.0),
                    'atr_percent': primary_analysis['latest'].get('atr_percent', 2.0)
                },
                patterns=primary_analysis.get('patterns', []),
                mtf_confirmation=mtf_result['mtf_confirmation'],
                mtf_score=mtf_result['mtf_score'],
                mtf_details=mtf_result['confirmations'],
                technical_score=technical_score,
                structure_score=structure_score,
                volume_score=volume_score,
                momentum_score=momentum_score,
                support_resistance=primary_analysis['structure'],
                breakout_info=primary_analysis.get('breakout_info', {}),
                warnings=self._identify_warnings(primary_analysis, risk_reward_ratio)
            )
            
            if final_confidence >= 60:
                self.logger.info(f"Signal {signal.id} for {symbol} - {quality.label} ({final_confidence:.1f}% confidence)")
                return signal
            else:
                return None
            
        except Exception as e:
            self.logger.error(f"Signal building error: {e}")
            return None
    
    def _calculate_structure_score(self, structure: Dict) -> float:
        """Calculate market structure score (0-100)"""
        score = 50  # Base score
        
        trend = structure.get('trend', 'ranging')
        if trend in ['uptrend', 'downtrend']:
            score += 20
        elif trend == 'strong_trend':
            score += 30
        
        if structure.get('near_support') or structure.get('near_resistance'):
            score += 15
        
        if structure.get('breakout_detected'):
            score += 15
        
        return min(100, score)
    
    def _calculate_volume_score(self, latest: pd.Series) -> float:
        """Calculate volume score (0-100)"""
        volume_ratio = latest.get('volume_ratio', 1.0)
        
        if volume_ratio >= 2.0:
            return 90
        elif volume_ratio >= 1.5:
            return 75
        elif volume_ratio >= 1.2:
            return 60
        elif volume_ratio >= 1.0:
            return 50
        else:
            return 30
    
    def _calculate_momentum_score(self, latest: pd.Series) -> float:
        """Calculate momentum score (0-100)"""
        score = 50
        
        # RSI momentum
        rsi = latest.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 10  # Healthy range
        if 40 <= rsi <= 60:
            score += 10  # Optimal range
        
        # MACD momentum
        if latest.get('macd_hist', 0) > 0:
            score += 15
        
        # Price momentum (ROC)
        if latest.get('roc_10', 0) > 0:
            score += 15
        
        return min(100, score)
    
    def _determine_market_regime(self, structure: Dict) -> MarketRegime:
        """Determine current market regime"""
        trend = structure.get('trend', 'ranging')
        volatility = structure.get('volatility', 'normal')
        
        if volatility == 'high':
            return MarketRegime.VOLATILE
        elif trend == 'strong_uptrend':
            return MarketRegime.STRONG_TREND_UP
        elif trend == 'uptrend':
            return MarketRegime.TREND_UP
        elif trend == 'strong_downtrend':
            return MarketRegime.STRONG_TREND_DOWN
        elif trend == 'downtrend':
            return MarketRegime.TREND_DOWN
        elif trend == 'ranging_bullish':
            return MarketRegime.RANGING_BULLISH
        elif trend == 'ranging_bearish':
            return MarketRegime.RANGING_BEARISH
        else:
            return MarketRegime.RANGING
    
    def _identify_warnings(self, analysis: Dict, risk_reward: float) -> List[str]:
        """Identify any warnings for the signal"""
        warnings = []
        
        # Check volume
        if analysis['latest'].get('volume_ratio', 1) < 0.8:
            warnings.append("Low volume - possible false signal")
        
        # Check volatility
        if analysis['latest'].get('atr_percent', 2) > 5:
            warnings.append("High volatility environment")
        
        # Check R/R
        if risk_reward < 2.0:
            warnings.append(f"Below optimal R/R ratio ({risk_reward:.2f})")
        
        return warnings
    
    def _fetch_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for analysis"""
        try:
            if not self.exchange_manager:
                # Return dummy data for testing
                return self._generate_dummy_data()
            
            return self.exchange_manager.fetch_ohlcv_data(symbol, timeframe)
            
        except Exception as e:
            self.logger.error(f"OHLCV fetch error for {symbol}: {e}")
            return None
    
    def _generate_dummy_data(self) -> pd.DataFrame:
        """Generate dummy data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
        data = {
            'timestamp': dates,
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50100,
            'low': np.random.randn(200).cumsum() + 49900,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 200)
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _generate_signal_id(self, symbol: str, side: SignalSide) -> str:
        """Generate unique signal ID"""
        content = f"{symbol}_{side.value}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _update_statistics(self, signal: Signal):
        """Update internal statistics"""
        self.stats['signals_generated'] += 1
        self.stats['signals_by_quality'][signal.quality.label] += 1
        
        # Update averages
        n = self.stats['signals_generated']
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (n - 1) + signal.confidence) / n
        )
        self.stats['avg_risk_reward'] = (
            (self.stats['avg_risk_reward'] * (n - 1) + signal.risk_reward_ratio) / n
        )
    
    def rank_opportunities_with_mtf(self, signals: List[Dict], 
                                  dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """
        Rank signals by quality - compatible with existing interface.
        
        Args:
            signals: List of signal dictionaries to rank
            dfs: Optional dataframes (not used in V14)
        
        Returns:
            Ranked list of signals with priority scores
        """
        try:
            if not signals:
                return []
            
            # Sort signals by multiple criteria
            ranked_signals = sorted(
                signals,
                key=lambda x: (
                    self._get_quality_score(x.get('quality_tier', 'marginal')),
                    x.get('confidence', 0),
                    x.get('risk_reward_ratio', 0),
                    -x.get('current_price', 0)  # Tie breaker
                ),
                reverse=True
            )
            
            # Add ranking and priority
            for i, signal in enumerate(ranked_signals, 1):
                signal['rank'] = i
                signal['priority'] = 1000 - (i * 10)  # Priority score
                
                # Add quality score for display
                signal['quality_score'] = self._get_quality_score(
                    signal.get('quality_tier', 'marginal')
                )
            
            return ranked_signals[:self.config.charts_per_batch]
            
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return signals
    
    def _get_quality_score(self, quality_tier: str) -> int:
        """Convert quality tier to numeric score for ranking"""
        quality_scores = {
            'elite': 100,
            'premium': 75,
            'standard': 50,
            'marginal': 25
        }
        return quality_scores.get(quality_tier.lower() if isinstance(quality_tier, str) else 'marginal', 0)
    
    def analyze_symbol_comprehensive(self, symbol: str) -> Optional[Dict]:
        """
        Wrapper method for backward compatibility.
        Calls the main analyze_symbol method and returns dict format.
        """
        signal = self.analyze_symbol(symbol)
        if signal:
            return signal.to_dict()
        return None
    
    def assess_risk(self, df: pd.DataFrame = None, symbol_data: Dict = None) -> Dict:
        """Risk assessment - compatibility method"""
        return {
            'total_risk_score': 0.5,
            'volatility_risk': 0.02,
            'risk_level': 'Medium'
        }
    
    def assess_market_conditions(self, df: pd.DataFrame = None, symbol_data: Dict = None) -> Dict:
        """Market conditions assessment - compatibility method"""
        return {
            'liquidity': 'medium',
            'sentiment': 'neutral',
            'price_change_24h': symbol_data.get('price_change_24h', 0) if symbol_data else 0,
            'volume_24h': symbol_data.get('volume_24h', 0) if symbol_data else 0
        }
    
    def create_technical_summary(self, df: pd.DataFrame = None, latest: pd.Series = None) -> Dict:
        """Technical summary - compatibility method"""
        return {
            'trend': {'direction': 'neutral', 'strength': 0.5},
            'momentum': {'rsi': 50, 'macd_bullish': False},
            'volatility': {'atr_percentage': 2.0},
            'volume': {'ratio': 1.0}
        }

# ===========================
# PUBLIC API FUNCTIONS
# ===========================

def create_signal_generator(config: Optional[Any] = None, exchange_manager=None) -> SignalGeneratorV14:
    """
    Factory function to create signal generator.
    Uses timeframes from EnhancedSystemConfig.
    """
    if not config:
        raise ValueError("EnhancedSystemConfig is required")
    
    # Create SignalConfig with timeframes from EnhancedSystemConfig
    signal_config = SignalConfig()
    
    # Get timeframes from the EnhancedSystemConfig
    signal_config.primary_timeframe = config.timeframe  # e.g., '1h' from database
    signal_config.confirmation_timeframes = config.confirmation_timeframes  # e.g., ['2h', '4h', '6h'] from database
    
    # Derive entry timeframe as the lowest confirmation timeframe
    if config.confirmation_timeframes:
        # Simple logic to get the lowest timeframe
        timeframes_in_minutes = []
        for tf in config.confirmation_timeframes:
            if tf.endswith('m'):
                minutes = int(tf[:-1])
            elif tf.endswith('h'):
                minutes = int(tf[:-1]) * 60
            elif tf.endswith('d'):
                minutes = int(tf[:-1]) * 1440
            else:
                minutes = 60  # Default to 1 hour
            timeframes_in_minutes.append((tf, minutes))
        
        # Sort and get the lowest
        timeframes_in_minutes.sort(key=lambda x: x[1])
        signal_config.entry_timeframe = timeframes_in_minutes[0][0]
    else:
        signal_config.entry_timeframe = signal_config.primary_timeframe
    
    # Copy other parameters from EnhancedSystemConfig if they exist
    if hasattr(config, 'max_risk_per_trade'):
        signal_config.max_risk_per_trade = config.max_risk_per_trade
    if hasattr(config, 'min_risk_reward'):
        signal_config.min_risk_reward = config.min_risk_reward
    if hasattr(config, 'min_conditions_for_signal'):
        signal_config.min_conditions_for_signal = config.min_conditions_for_signal
    if hasattr(config, 'min_confidence_for_execution'):
        signal_config.min_confidence_for_execution = config.min_confidence_for_execution
    
    return SignalGeneratorV14(signal_config, exchange_manager)

# Maintain backward compatibility
SignalGenerator = SignalGeneratorV14
create_mtf_signal_generator = create_signal_generator

# ===========================
# EXPORTS
# ===========================

__all__ = [
    'SignalGeneratorV14',
    'SignalGenerator',
    'SignalConfig',
    'Signal',
    'SignalSide',
    'OrderType',
    'SignalQuality',
    'MarketRegime',
    'create_signal_generator',
    'create_mtf_signal_generator'
]