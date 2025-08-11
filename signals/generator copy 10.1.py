"""
ENHANCED Multi-Timeframe Signal Generation with Market Intelligence
VERSION 10.1 - FIXED STOP LOSS ISSUES

CRITICAL FIXES FROM V10.0:
✅ Fixed stop distances from 0.05-0.10% to 1.0-5.0% (crypto appropriate)
✅ Fixed entry logic to avoid chasing momentum
✅ Tightened signal requirements (4.5 score vs 3.0)
✅ Added leverage awareness for stop calculation
✅ Fixed volatility-based stop multipliers
✅ Better ATR utilization without artificial caps

COMPLETE IMPLEMENTATION - PRODUCTION READY
"""

import pandas as pd
import numpy as np
import logging
import time
import aiohttp
import asyncio
import requests
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

from config.config import EnhancedSystemConfig
from signals.mtf_validator import StrictMTFValidator

# ===== API CONFIGURATION =====
CRYPTOPANIC_API_KEY = '2c2a4ce275d7c36a8bb5ac71bf6a3b5a61e60cb8'

# ===== V10.1 CONFIGURATION CLASSES WITH FIXED STOP LOSSES =====

class SignalQualityTier(Enum):
    """Signal quality tiers for risk management"""
    PREMIUM = "premium"        # 80%+ confidence, all conditions met
    STANDARD = "standard"      # 65-80% confidence, most conditions met
    SPECULATIVE = "speculative"  # 50-65% confidence, relaxed conditions
    EXPERIMENTAL = "experimental"  # 45-50% confidence, very relaxed

class MarketCondition(Enum):
    """Market condition states"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    EXTREME = "extreme"

@dataclass
class AdaptiveSignalConfig:
    """ADAPTIVE configuration that adjusts to market conditions - FIXED FOR CRYPTO"""
    
    # FIXED: Crypto-appropriate stop distances (was 0.05-0.10%, now 1.0-5.0%)
    min_stop_distance_pct: float = 1.0  # 1% minimum stop (was 0.05%)
    max_stop_distance_pct: float = 5.0  # 5% maximum stop (was 0.10%)
    structure_stop_buffer: float = 0.01  # 1% buffer from structure (was 0.003)
    
    # FIXED: Entry logic to avoid chasing
    entry_buffer_from_structure: float = 0.002
    entry_limit_distance: float = 0.02
    momentum_entry_adjustment: float = 0.002  # Reduced from 0.003
    use_pullback_entries: bool = True  # NEW: Wait for pullbacks in momentum
    
    # Take profit distances - kept reasonable
    min_tp_distance_pct: float = 0.015
    max_tp_distance_pct: float = 0.20
    tp1_multiplier: float = 2.0
    tp2_multiplier: float = 3.5
    use_market_based_tp1: bool = True
    use_market_based_tp2: bool = True
    
    # Risk/Reward - adjusted for wider stops
    min_risk_reward: float = 1.5  # Reduced from 1.8 since stops are wider
    max_risk_reward: float = 10.0
    
    # Signal quality - tightened
    min_confidence_for_signal: float = 60.0  # Raised from 55.0
    mtf_confidence_boost: float = 10.0
    
    # RSI thresholds - kept same
    min_rsi_for_short: float = 50.0
    max_rsi_for_short: float = 75.0
    min_rsi_for_long: float = 25.0
    max_rsi_for_long: float = 50.0
    
    # Stochastic thresholds - kept same
    min_stoch_for_short: float = 50.0
    max_stoch_for_long: float = 50.0
    
    # Volatility thresholds - adjusted for crypto
    high_volatility_threshold: float = 0.10  # 10% daily volatility (was 0.08)
    low_volatility_threshold: float = 0.02
    extreme_volatility_threshold: float = 0.15  # NEW: For crypto flash crashes
    
    # Market microstructure
    use_order_book_analysis: bool = True
    min_order_book_imbalance: float = 0.6
    price_momentum_lookback: int = 5
    
    # Fast signal parameters - tightened
    min_momentum_for_fast_signal: float = 2.5  # Raised from 2.0
    max_choppiness_score: float = 0.5  # Reduced from 0.6
    volume_surge_multiplier: float = 2.5  # Raised from 2.0
    fast_signal_bonus_confidence: float = 10.0
    
    # Support/Resistance
    support_resistance_lookback: int = 20
    min_distance_from_support: float = 0.02
    min_distance_from_resistance: float = 0.02
    
    # Volume requirements - tightened
    min_volume_ratio_for_signal: float = 1.0  # Raised from 0.8
    strong_volume_threshold: float = 2.0  # Raised from 1.5
    
    # Quality filter thresholds
    min_quality_score_to_reject: float = -2.0
    quality_warning_threshold: float = 0
    
    # NEW: Leverage awareness
    leverage_stop_multiplier: Dict[int, float] = field(default_factory=lambda: {
        1: 1.0,   # No leverage: normal stops
        5: 1.2,   # 5x leverage: 20% wider stops
        10: 1.5,  # 10x leverage: 50% wider stops
        25: 2.0,  # 25x leverage: 100% wider stops
        50: 2.5,  # 50x leverage: 150% wider stops
        100: 3.0  # 100x leverage: 200% wider stops
    })
    
    # Adaptive parameters
    current_market_condition: MarketCondition = MarketCondition.RANGING
    adaptation_factor: float = 1.0
    last_adaptation_time: datetime = field(default_factory=datetime.now)

    # ADD these new fields:
    require_mtf_validation: bool = True  # Force MTF validation
    min_mtf_alignment: float = 0.6  # Minimum 60% MTF alignment
    block_none_mtf_signals: bool = True  # Never return NONE signals
    min_tp1_confidence: float = 0.6  # Minimum 60% TP1 reachability
    
    def adapt_to_market(self, volatility: float, volume: float, trend_strength: float, 
                        recent_win_rate: float = 0.5) -> None:
        """Dynamically adjust parameters based on market conditions - ENHANCED"""
        
        # Determine market condition with crypto-specific thresholds
        if volatility > self.extreme_volatility_threshold:
            self.current_market_condition = MarketCondition.EXTREME
        elif volatility > self.high_volatility_threshold:
            self.current_market_condition = MarketCondition.VOLATILE
        elif trend_strength > 0.7:
            if trend_strength > 0:
                self.current_market_condition = MarketCondition.TRENDING_UP
            else:
                self.current_market_condition = MarketCondition.TRENDING_DOWN
        else:
            self.current_market_condition = MarketCondition.RANGING
        
        # FIXED: Adjust parameters with proper crypto ranges
        if self.current_market_condition == MarketCondition.EXTREME:
            # Very conservative in extreme conditions - WIDER STOPS
            self.min_risk_reward = 1.8  # Reduced from 2.5
            self.min_confidence_for_signal = 70.0  # Raised from 65.0
            self.min_stop_distance_pct = 2.0  # Much wider (was 0.08)
            self.max_stop_distance_pct = 8.0  # Much wider
            self.min_volume_ratio_for_signal = 1.5  # Raised from 1.2
            self.adaptation_factor = 0.5
            
        elif self.current_market_condition == MarketCondition.VOLATILE:
            # Conservative in high volatility - WIDER STOPS
            self.min_risk_reward = 1.5  # Reduced from 2.0
            self.min_confidence_for_signal = 65.0  # Raised from 60.0
            self.min_stop_distance_pct = 1.5  # Wider (was 0.06)
            self.max_stop_distance_pct = 6.0  # Wider
            self.min_volume_ratio_for_signal = 1.2  # Raised from 1.0
            self.adaptation_factor = 0.7
            
        elif self.current_market_condition in [MarketCondition.TRENDING_UP, MarketCondition.TRENDING_DOWN]:
            # Standard parameters in trending markets
            self.min_risk_reward = 1.5  # Reduced from 1.8
            self.min_confidence_for_signal = 60.0  # Raised from 55.0
            self.min_stop_distance_pct = 1.0  # Standard (was 0.05)
            self.max_stop_distance_pct = 5.0  # Standard
            self.min_volume_ratio_for_signal = 1.0  # Raised from 0.8
            self.adaptation_factor = 1.0
            
        else:  # RANGING
            # Tighter in ranging markets but still crypto-appropriate
            self.min_risk_reward = 1.3  # Reduced from 1.5
            self.min_confidence_for_signal = 55.0  # Raised from 50.0
            self.min_stop_distance_pct = 0.8  # Tighter but reasonable (was 0.04)
            self.max_stop_distance_pct = 4.0  # Still reasonable
            self.min_volume_ratio_for_signal = 0.9  # Raised from 0.7
            self.adaptation_factor = 1.2
        
        # Adjust based on recent performance
        if recent_win_rate < 0.3:
            # Tighten parameters if losing - but keep stops wide
            self.min_risk_reward *= 1.2
            self.min_confidence_for_signal = min(80.0, self.min_confidence_for_signal * 1.1)
            self.min_stop_distance_pct *= 1.2  # Make stops even wider when losing
        elif recent_win_rate > 0.7:
            # Can be slightly more aggressive if winning
            self.min_risk_reward *= 0.95
            self.min_confidence_for_signal *= 0.95
        
        # Adjust RSI ranges based on trend
        if self.current_market_condition == MarketCondition.TRENDING_UP:
            self.min_rsi_for_long = 30.0
            self.max_rsi_for_long = 60.0
            self.min_rsi_for_short = 60.0
            self.max_rsi_for_short = 80.0
        elif self.current_market_condition == MarketCondition.TRENDING_DOWN:
            self.min_rsi_for_long = 20.0
            self.max_rsi_for_long = 40.0
            self.min_rsi_for_short = 40.0
            self.max_rsi_for_short = 70.0
        
        self.last_adaptation_time = datetime.now()
    
    def get_leverage_adjusted_stop_distance(self, base_stop_pct: float, leverage: int) -> float:
        """NEW: Adjust stop distance based on leverage"""
        # Find the appropriate multiplier for the leverage
        multiplier = 1.0
        for lev_threshold in sorted(self.leverage_stop_multiplier.keys()):
            if leverage >= lev_threshold:
                multiplier = self.leverage_stop_multiplier[lev_threshold]
        
        return base_stop_pct * multiplier

@dataclass
class PerformanceMetrics:
    """Track performance for adaptive adjustments"""
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    recent_signals: deque = field(default_factory=lambda: deque(maxlen=50))
    
    @property
    def win_rate(self) -> float:
        if self.total_signals == 0:
            return 0.5
        return self.winning_signals / self.total_signals
    
    @property
    def recent_win_rate(self) -> float:
        if len(self.recent_signals) < 10:
            return 0.5
        recent_wins = sum(1 for s in self.recent_signals if s['profitable'])
        return recent_wins / len(self.recent_signals)

@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent excessive losses"""
    max_consecutive_losses: int = 5
    max_hourly_signals: int = 10
    min_win_rate_threshold: float = 0.25
    cooldown_period_minutes: int = 30
    
    consecutive_losses: int = 0
    hourly_signals: deque = field(default_factory=lambda: deque())
    is_active: bool = False
    cooldown_until: Optional[datetime] = None
    
    def check_signal_allowed(self, signal: Dict, performance: PerformanceMetrics) -> Tuple[bool, str]:
        """Check if signal should be allowed"""
        
        # Check if in cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return False, f"Circuit breaker cooldown until {self.cooldown_until}"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.activate_cooldown()
            return False, f"Circuit breaker: {self.consecutive_losses} consecutive losses"
        
        # Check hourly signal rate
        current_time = datetime.now()
        self.hourly_signals = deque(
            [s for s in self.hourly_signals if s > current_time - timedelta(hours=1)]
        )
        
        if len(self.hourly_signals) >= self.max_hourly_signals:
            # Only allow high quality signals
            if signal.get('quality_tier') != SignalQualityTier.PREMIUM:
                return False, f"Too many signals ({len(self.hourly_signals)}/hour), only PREMIUM allowed"
        
        # Check win rate if enough history
        if performance.total_signals >= 20:
            if performance.recent_win_rate < self.min_win_rate_threshold:
                # Only allow premium signals during drawdown
                if signal.get('quality_tier') not in [SignalQualityTier.PREMIUM, SignalQualityTier.STANDARD]:
                    return False, f"Low win rate ({performance.recent_win_rate:.1%}), only PREMIUM/STANDARD allowed"
        
        return True, "Signal allowed"
    
    def activate_cooldown(self):
        """Activate circuit breaker cooldown"""
        self.is_active = True
        self.cooldown_until = datetime.now() + timedelta(minutes=self.cooldown_period_minutes)
        self.consecutive_losses = 0
    
    def record_signal_result(self, profitable: bool):
        """Record signal result for circuit breaker tracking"""
        if profitable:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        self.hourly_signals.append(datetime.now())

@dataclass
class MultiTimeframeContext:
    """Container for multi-timeframe market analysis"""
    dominant_trend: str
    trend_strength: float
    higher_tf_zones: List[Dict]
    key_support: float
    key_resistance: float
    momentum_alignment: bool
    entry_bias: str
    confirmation_score: float
    structure_timeframe: str
    market_regime: str
    volatility_level: str

@dataclass
class MarketIntelligence:
    """Market intelligence data from APIs with caching"""
    # Fear & Greed
    fear_greed_index: float = 50.0
    fear_greed_classification: str = 'Neutral'
    fear_greed_change: float = 0.0
    
    # Funding rates
    funding_rate: float = 0.0
    funding_sentiment: str = 'neutral'
    predicted_funding: float = 0.0
    avg_funding_24h: float = 0.0
    
    # News sentiment
    news_sentiment_score: float = 0.0
    news_count_24h: int = 0
    bullish_news_count: int = 0
    bearish_news_count: int = 0
    important_news: List[Dict] = field(default_factory=list)
    news_classification: str = 'neutral'
    
    # Overall scores
    overall_api_sentiment: float = 50.0
    api_signal_modifier: float = 1.0
    
    # Cache metadata
    last_update: datetime = field(default_factory=datetime.now)
    is_cached: bool = False

# ===== ENHANCED API COMPONENTS WITH INTELLIGENT FALLBACKS =====

class IntelligentAPICache:
    """Intelligent caching system with historical fallbacks"""
    
    def __init__(self, cache_duration: int = 3600):
        self.cache = {}
        self.historical_data = deque(maxlen=100)
        self.cache_duration = cache_duration
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if fresh"""
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                return value
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache value with timestamp"""
        self.cache[key] = (time.time(), value)
        self.historical_data.append({
            'key': key,
            'value': value,
            'timestamp': time.time()
        })
    
    def get_intelligent_fallback(self, key_pattern: str) -> Any:
        """Get intelligent fallback based on historical data"""
        # Try to find similar recent data
        similar_data = [
            d['value'] for d in self.historical_data 
            if key_pattern in d['key'] and time.time() - d['timestamp'] < 86400
        ]
        
        if similar_data:
            # Return average of recent similar data
            if isinstance(similar_data[0], (int, float)):
                return sum(similar_data) / len(similar_data)
            else:
                # Return most recent
                return similar_data[-1]
        
        # Return sensible defaults
        if 'fear_greed' in key_pattern:
            return {'value': 50, 'classification': 'Neutral', 'change': 0}
        elif 'funding' in key_pattern:
            return {'current_rate': 0.0001, 'sentiment': 'neutral'}
        elif 'news' in key_pattern:
            return {'sentiment_score': 0, 'classification': 'neutral'}
        
        return None

class EnhancedFearGreedAnalyzer:
    """Enhanced Fear & Greed analyzer with async support and intelligent fallbacks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = IntelligentAPICache(cache_duration=3600)
        self.session = None
    
    async def get_fear_greed_index_async(self) -> Dict:
        """Async version with intelligent fallback"""
        try:
            # Check cache first
            cached = self.cache.get('fear_greed_latest')
            if cached:
                return cached
            
            async with aiohttp.ClientSession() as session:
                url = "https://api.alternative.me/fng/?limit=2"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = self._process_fear_greed_data(data)
                        self.cache.set('fear_greed_latest', result)
                        return result
            
            # Fallback to historical average
            return self.cache.get_intelligent_fallback('fear_greed')
            
        except Exception as e:
            self.logger.debug(f"Fear & Greed async error: {e}")
            return self.cache.get_intelligent_fallback('fear_greed')
    
    def get_fear_greed_index(self) -> Dict:
        """Synchronous version with intelligent fallback"""
        try:
            # Check cache first
            cached = self.cache.get('fear_greed_latest')
            if cached:
                return cached
            
            url = "https://api.alternative.me/fng/?limit=2"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result = self._process_fear_greed_data(data)
                self.cache.set('fear_greed_latest', result)
                return result
            
            return self.cache.get_intelligent_fallback('fear_greed')
            
        except Exception as e:
            self.logger.debug(f"Fear & Greed sync error: {e}")
            return self.cache.get_intelligent_fallback('fear_greed')
    
    def _process_fear_greed_data(self, data: Dict) -> Dict:
        """Process fear & greed API response"""
        if data and 'data' in data and len(data['data']) > 0:
            current = data['data'][0]
            result = {
                'value': float(current['value']),
                'classification': current['value_classification'],
                'timestamp': current['timestamp'],
                'time_until_update': current.get('time_until_update', 0)
            }
            
            if len(data['data']) > 1:
                yesterday = data['data'][1]
                result['yesterday_value'] = float(yesterday['value'])
                result['change'] = result['value'] - result['yesterday_value']
            else:
                result['change'] = 0
            
            return result
        
        return {'value': 50, 'classification': 'Neutral', 'change': 0}

class EnhancedFundingRateAnalyzer:
    """Enhanced funding rate analyzer with caching"""
    
    def __init__(self, exchange_manager=None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange_manager
        self.cache = IntelligentAPICache(cache_duration=300)
    
    async def get_funding_rate_async(self, symbol: str) -> Dict:
        """Async funding rate fetch"""
        # For now, use sync version in async wrapper
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get_funding_rate, symbol
        )
    
    def get_funding_rate(self, symbol: str) -> Dict:
        """Get funding rate with intelligent fallback"""
        try:
            cache_key = f'funding_{symbol}'
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            
            if not self.exchange:
                return self.cache.get_intelligent_fallback('funding')
            
            # Fetch from exchange
            funding = self.exchange.exchange.fetch_funding_rate(symbol)
            
            # Try to get history for average
            try:
                funding_history = self.exchange.exchange.fetch_funding_rate_history(
                    symbol, limit=8
                )
                avg_funding = sum(f['fundingRate'] for f in funding_history) / len(funding_history)
            except:
                avg_funding = funding['fundingRate']
            
            result = {
                'current_rate': funding['fundingRate'],
                'predicted_rate': funding.get('predictedFundingRate', funding['fundingRate']),
                'funding_timestamp': funding['timestamp'],
                'avg_24h': avg_funding,
                'sentiment': self._interpret_funding_rate(funding['fundingRate']),
                'next_funding_time': funding.get('fundingDatetime', '')
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.debug(f"Funding rate error: {e}")
            return self.cache.get_intelligent_fallback('funding')
    
    def _interpret_funding_rate(self, rate: float) -> str:
        """Interpret funding rate sentiment"""
        if rate > 0.0005:
            return 'extremely_bullish'
        elif rate > 0.0002:
            return 'bullish'
        elif rate > 0.00005:
            return 'slightly_bullish'
        elif rate < -0.0005:
            return 'extremely_bearish'
        elif rate < -0.0002:
            return 'bearish'
        elif rate < -0.00005:
            return 'slightly_bearish'
        else:
            return 'neutral'

class EnhancedNewsAnalyzer:
    """Enhanced news analyzer with better caching"""
    
    def __init__(self, api_key: str = CRYPTOPANIC_API_KEY):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.cache = IntelligentAPICache(cache_duration=600)
    
    async def get_news_sentiment_async(self, symbol: str) -> Dict:
        """Async news sentiment fetch"""
        try:
            base_currency = symbol.split('/')[0]
            cache_key = f'news_{base_currency}'
            
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            
            if not self.api_key:
                return self.cache.get_intelligent_fallback('news')
            
            async with aiohttp.ClientSession() as session:
                url = f"https://cryptopanic.com/api/v1/posts/"
                params = {
                    'auth_token': self.api_key,
                    'currencies': base_currency,
                    'filter': 'hot',
                    'public': 'true'
                }
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = self._analyze_news_items(data.get('results', []), base_currency)
                        self.cache.set(cache_key, result)
                        return result
            
            return self.cache.get_intelligent_fallback('news')
            
        except Exception as e:
            self.logger.debug(f"News API async error: {e}")
            return self.cache.get_intelligent_fallback('news')
    
    def get_news_sentiment(self, symbol: str) -> Dict:
        """Synchronous news sentiment fetch"""
        try:
            base_currency = symbol.split('/')[0]
            cache_key = f'news_{base_currency}'
            
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            
            if not self.api_key:
                return self.cache.get_intelligent_fallback('news')
            
            url = f"https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': self.api_key,
                'currencies': base_currency,
                'filter': 'hot',
                'public': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = self._analyze_news_items(data.get('results', []), base_currency)
                self.cache.set(cache_key, result)
                return result
            
            return self.cache.get_intelligent_fallback('news')
            
        except Exception as e:
            self.logger.debug(f"News API sync error: {e}")
            return self.cache.get_intelligent_fallback('news')
    
    def _analyze_news_items(self, news_items: List[Dict], currency: str) -> Dict:
        """Analyze news items for sentiment"""
        try:
            if not news_items:
                return self._get_default_news_sentiment()
            
            total_score = 0
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            important_news = []
            
            for item in news_items[:20]:
                votes = item.get('votes', {})
                positive = votes.get('positive', 0)
                negative = votes.get('negative', 0)
                importance = votes.get('important', 0)
                
                if positive + negative > 0:
                    item_sentiment = (positive - negative) / (positive + negative)
                else:
                    item_sentiment = 0
                
                weight = 1 + (importance * 0.1)
                total_score += item_sentiment * weight
                
                if item_sentiment > 0.2:
                    bullish_count += 1
                elif item_sentiment < -0.2:
                    bearish_count += 1
                else:
                    neutral_count += 1
                
                if importance > 5 or (positive + negative) > 10:
                    important_news.append({
                        'title': item.get('title', ''),
                        'sentiment': item_sentiment,
                        'importance': importance,
                        'published_at': item.get('published_at', '')
                    })
            
            news_count = len(news_items)
            sentiment_score = total_score / news_count if news_count > 0 else 0
            
            return {
                'sentiment_score': sentiment_score,
                'news_count_24h': news_count,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'important_news': important_news[:5],
                'sentiment_classification': self._classify_sentiment(sentiment_score)
            }
            
        except Exception as e:
            self.logger.error(f"News analysis error: {e}")
            return self._get_default_news_sentiment()
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score"""
        if score > 0.5:
            return 'very_bullish'
        elif score > 0.2:
            return 'bullish'
        elif score > -0.2:
            return 'neutral'
        elif score > -0.5:
            return 'bearish'
        else:
            return 'very_bearish'
    
    def _get_default_news_sentiment(self) -> Dict:
        """Default news sentiment data"""
        return {
            'sentiment_score': 0,
            'news_count_24h': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'important_news': [],
            'sentiment_classification': 'neutral'
        }

class EnhancedMarketIntelligenceAggregator:
    """Enhanced aggregator with async support and intelligent fallbacks"""
    
    def __init__(self, exchange_manager=None):
        self.logger = logging.getLogger(__name__)
        self.fear_greed = EnhancedFearGreedAnalyzer()
        self.funding = EnhancedFundingRateAnalyzer(exchange_manager)
        self.news = EnhancedNewsAnalyzer()
        self.cache = IntelligentAPICache(cache_duration=300)
    
    async def gather_intelligence_async(self, symbol: str) -> MarketIntelligence:
        """Gather all intelligence asynchronously"""
        try:
            # Run all API calls in parallel
            tasks = [
                self.fear_greed.get_fear_greed_index_async(),
                self.funding.get_funding_rate_async(symbol),
                self.news.get_news_sentiment_async(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            intel = MarketIntelligence()
            
            # Process Fear & Greed
            if not isinstance(results[0], Exception):
                fear_greed_data = results[0]
                intel.fear_greed_index = fear_greed_data.get('value', 50)
                intel.fear_greed_classification = fear_greed_data.get('classification', 'Neutral')
                intel.fear_greed_change = fear_greed_data.get('change', 0)
            
            # Process Funding
            if not isinstance(results[1], Exception):
                funding_data = results[1]
                intel.funding_rate = funding_data.get('current_rate', 0.0001)
                intel.funding_sentiment = funding_data.get('sentiment', 'neutral')
                intel.predicted_funding = funding_data.get('predicted_rate', intel.funding_rate)
                intel.avg_funding_24h = funding_data.get('avg_24h', intel.funding_rate)
            
            # Process News
            if not isinstance(results[2], Exception):
                news_data = results[2]
                intel.news_sentiment_score = news_data.get('sentiment_score', 0)
                intel.news_count_24h = news_data.get('news_count_24h', 0)
                intel.bullish_news_count = news_data.get('bullish_count', 0)
                intel.bearish_news_count = news_data.get('bearish_count', 0)
                intel.important_news = news_data.get('important_news', [])
                intel.news_classification = news_data.get('sentiment_classification', 'neutral')
            
            intel = self._calculate_overall_sentiment(intel)
            intel.last_update = datetime.now()
            
            return intel
            
        except Exception as e:
            self.logger.error(f"Async intelligence gathering error: {e}")
            return self._get_cached_or_default_intelligence(symbol)
    
    def gather_intelligence(self, symbol: str) -> MarketIntelligence:
        """Synchronous intelligence gathering with fallbacks"""
        try:
            # Check cache first
            cache_key = f'intel_{symbol}'
            cached = self.cache.get(cache_key)
            if cached:
                cached.is_cached = True
                return cached
            
            intel = MarketIntelligence()
            
            # Gather each component with error handling
            try:
                fear_greed_data = self.fear_greed.get_fear_greed_index()
                intel.fear_greed_index = fear_greed_data['value']
                intel.fear_greed_classification = fear_greed_data['classification']
                intel.fear_greed_change = fear_greed_data.get('change', 0)
            except Exception as e:
                self.logger.debug(f"Fear & Greed fetch failed: {e}")
            
            try:
                funding_data = self.funding.get_funding_rate(symbol)
                intel.funding_rate = funding_data['current_rate']
                intel.funding_sentiment = funding_data['sentiment']
                intel.predicted_funding = funding_data.get('predicted_rate', funding_data['current_rate'])
                intel.avg_funding_24h = funding_data.get('avg_24h', funding_data['current_rate'])
            except Exception as e:
                self.logger.debug(f"Funding rate fetch failed: {e}")
            
            try:
                news_data = self.news.get_news_sentiment(symbol)
                intel.news_sentiment_score = news_data['sentiment_score']
                intel.news_count_24h = news_data['news_count_24h']
                intel.bullish_news_count = news_data['bullish_count']
                intel.bearish_news_count = news_data['bearish_count']
                intel.important_news = news_data.get('important_news', [])
                intel.news_classification = news_data.get('sentiment_classification', 'neutral')
            except Exception as e:
                self.logger.debug(f"News sentiment fetch failed: {e}")
            
            intel = self._calculate_overall_sentiment(intel)
            intel.last_update = datetime.now()
            
            # Cache the result
            self.cache.set(cache_key, intel)
            
            return intel
            
        except Exception as e:
            self.logger.error(f"Intelligence gathering error: {e}")
            return self._get_cached_or_default_intelligence(symbol)
    
    def _calculate_overall_sentiment(self, intel: MarketIntelligence) -> MarketIntelligence:
        """Calculate overall API sentiment with weighted average"""
        try:
            # Normalize scores to 0-100 scale
            fear_greed_score = intel.fear_greed_index
            
            # Convert funding rate to sentiment score
            if intel.funding_rate > 0:
                funding_score = max(0, min(100, 50 - intel.funding_rate * 50000))
            else:
                funding_score = max(0, min(100, 50 - intel.funding_rate * 50000))
            
            # Convert news sentiment to 0-100 scale
            news_score = (intel.news_sentiment_score + 1) * 50
            
            # Weighted average
            weights = {'fear_greed': 0.35, 'funding': 0.35, 'news': 0.30}
            
            intel.overall_api_sentiment = (
                fear_greed_score * weights['fear_greed'] +
                funding_score * weights['funding'] +
                news_score * weights['news']
            )
            
            # Calculate signal modifier
            if intel.overall_api_sentiment > 75 or intel.overall_api_sentiment < 25:
                intel.api_signal_modifier = 1.15
            elif intel.overall_api_sentiment > 65 or intel.overall_api_sentiment < 35:
                intel.api_signal_modifier = 1.08
            else:
                intel.api_signal_modifier = 1.0
            
            return intel
            
        except Exception as e:
            self.logger.error(f"Sentiment calculation error: {e}")
            intel.api_signal_modifier = 1.0
            return intel
    
    def _get_cached_or_default_intelligence(self, symbol: str) -> MarketIntelligence:
        """Get cached intelligence or return defaults"""
        cache_key = f'intel_{symbol}'
        cached = self.cache.get(cache_key)
        
        if cached:
            cached.is_cached = True
            return cached
        
        # Return default intelligence
        intel = MarketIntelligence()
        intel.is_cached = False
        return intel

# ===== QUALITY FILTER FUNCTIONS FROM V9 =====

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
            
            weight = 1 / (lookback_periods.index(period) + 1)
            momentum_scores.append(change_pct * weight)
        
        weighted_momentum = sum(momentum_scores) / sum(1/(i+1) for i in range(len(lookback_periods)))
        
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
        
        price_trend = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        volume_trend = (recent['volume'].mean() - df['volume'].rolling(window=20).mean().iloc[-window]) / df['volume'].rolling(window=20).mean().iloc[-window]
        
        if price_trend > 0.01 and volume_trend < -0.1:
            return {
                'divergence': True,
                'type': 'bearish_divergence',
                'strength': abs(volume_trend),
                'warning': 'Price rising on declining volume'
            }
        elif price_trend < -0.01 and volume_trend < -0.1:
            return {
                'divergence': True,
                'type': 'potential_reversal',
                'strength': abs(volume_trend),
                'warning': 'Selling pressure may be exhausting'
            }
        elif abs(price_trend) > 0.02 and volume_trend > 0.5:
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

def detect_divergence(df: pd.DataFrame, side: str, window: int = 10) -> dict:
    """Detect bullish or bearish divergence based on price and RSI"""
    try:
        if len(df) < window * 2:
            return {'has_divergence': False, 'type': 'none', 'strength': 0}
        
        recent = df.tail(window)
        older = df.tail(window * 2).head(window)
        
        recent_low = recent['low'].min()
        older_low = older['low'].min()
        recent_high = recent['high'].max()
        older_high = older['high'].max()
        
        recent_rsi_low = recent['rsi'].min()
        older_rsi_low = older['rsi'].min()
        recent_rsi_high = recent['rsi'].max()
        older_rsi_high = older['rsi'].max()
        
        if recent_low < older_low and recent_rsi_low > older_rsi_low:
            strength = abs((recent_rsi_low - older_rsi_low) / older_rsi_low)
            return {
                'has_divergence': True,
                'type': 'bullish_divergence',
                'strength': min(1.0, strength),
                'favorable_for': 'buy',
                'description': 'Price making lower lows, RSI making higher lows'
            }
        
        if recent_high > older_high and recent_rsi_high < older_rsi_high:
            strength = abs((older_rsi_high - recent_rsi_high) / older_rsi_high)
            return {
                'has_divergence': True,
                'type': 'bearish_divergence',
                'strength': min(1.0, strength),
                'favorable_for': 'sell',
                'description': 'Price making higher highs, RSI making lower highs'
            }
        
        return {'has_divergence': False, 'type': 'none', 'strength': 0}
        
    except Exception:
        return {'has_divergence': False, 'type': 'none', 'strength': 0}

def check_near_support_resistance(df: pd.DataFrame, current_price: float, side: str, window: int = 20) -> dict:
    """Check if price is near recent support or resistance levels"""
    try:
        if len(df) < window:
            return {'near_level': False, 'level_type': 'none', 'distance_pct': 1.0}
        
        recent = df.tail(window)
        
        recent_high = recent['high'].max()
        recent_low = recent['low'].min()
        
        distance_from_resistance = (recent_high - current_price) / current_price
        distance_from_support = (current_price - recent_low) / recent_low
        
        threshold = 0.015  # 1.5% threshold
        
        if side == 'buy':
            if distance_from_support < threshold:
                return {
                    'near_level': True,
                    'level_type': 'support',
                    'distance_pct': distance_from_support,
                    'favorable': True,
                    'level_price': recent_low
                }
            elif distance_from_resistance < threshold:
                return {
                    'near_level': True,
                    'level_type': 'resistance',
                    'distance_pct': distance_from_resistance,
                    'favorable': False,
                    'level_price': recent_high
                }
        else:
            if distance_from_resistance < threshold:
                return {
                    'near_level': True,
                    'level_type': 'resistance',
                    'distance_pct': distance_from_resistance,
                    'favorable': True,
                    'level_price': recent_high
                }
            elif distance_from_support < threshold:
                return {
                    'near_level': True,
                    'level_type': 'support',
                    'distance_pct': distance_from_support,
                    'favorable': False,
                    'level_price': recent_low
                }
        
        return {'near_level': False, 'level_type': 'none', 'distance_pct': 1.0}
        
    except Exception:
        return {'near_level': False, 'level_type': 'none', 'distance_pct': 1.0}

def identify_fast_moving_setup(df: pd.DataFrame, side: str) -> dict:
    """Identify setups likely to move fast"""
    try:
        if len(df) < 20:
            return {'is_fast_setup': False, 'score': 0}
        
        latest = df.iloc[-1]
        score = 0
        factors = []
        
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_close = latest['close']
        
        if side == 'buy':
            if current_close > recent_high * 0.99:
                score += 2
                factors.append('breakout_imminent')
            
            if len(df) >= 10:
                touched_support = any(df['low'].tail(10) <= recent_low * 1.015)
                bounced_up = current_close > recent_low * 1.015
                if touched_support and bounced_up:
                    score += 1.5
                    factors.append('support_bounce')
        
        else:
            if current_close < recent_low * 1.01:
                score += 2
                factors.append('breakdown_imminent')
            
            if len(df) >= 10:
                touched_resistance = any(df['high'].tail(10) >= recent_high * 0.985)
                rejected_down = current_close < recent_high * 0.985
                if touched_resistance and rejected_down:
                    score += 1.5
                    factors.append('resistance_rejection')
        
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio > 1.8:
            score += 1
            factors.append('volume_surge')
        elif volume_ratio > 1.3:
            score += 0.5
            factors.append('increased_volume')
        
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        
        if side == 'buy':
            if 30 < rsi < 55 and macd > macd_signal:
                score += 1
                factors.append('momentum_building')
        else:
            if 45 < rsi < 70 and macd < macd_signal:
                score += 1
                factors.append('momentum_building')
        
        if 'atr' in df.columns:
            current_atr = latest['atr']
            avg_atr = df['atr'].tail(20).mean()
            if current_atr > avg_atr * 1.15:
                score += 0.5
                factors.append('volatility_expansion')
        
        return {
            'is_fast_setup': score >= 2.0,
            'score': score,
            'factors': factors,
            'likelihood': 'high' if score >= 3.0 else 'medium' if score >= 2.0 else 'low'
        }
        
    except Exception:
        return {'is_fast_setup': False, 'score': 0}

def filter_choppy_markets(df: pd.DataFrame, window: int = 20) -> dict:
    """Filter out choppy/ranging markets"""
    try:
        if len(df) < window:
            return {'is_choppy': False, 'choppiness_score': 0}
        
        recent = df.tail(window)
        
        up_moves = sum(1 for i in range(1, len(recent)) if recent['close'].iloc[i] > recent['close'].iloc[i-1])
        down_moves = window - 1 - up_moves
        directional_ratio = abs(up_moves - down_moves) / (window - 1)
        
        if 'atr' in df.columns:
            total_price_movement = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])
            total_atr = recent['atr'].sum()
            efficiency_ratio = total_price_movement / total_atr if total_atr > 0 else 0
        else:
            efficiency_ratio = 0.5
        
        reversals = 0
        for i in range(2, len(recent)):
            if (recent['close'].iloc[i] > recent['close'].iloc[i-1] and 
                recent['close'].iloc[i-1] < recent['close'].iloc[i-2]):
                reversals += 1
            elif (recent['close'].iloc[i] < recent['close'].iloc[i-1] and 
                  recent['close'].iloc[i-1] > recent['close'].iloc[i-2]):
                reversals += 1
        
        reversal_ratio = reversals / (window - 2)
        
        choppiness_score = (1 - directional_ratio) * 0.4 + (1 - efficiency_ratio) * 0.3 + reversal_ratio * 0.3
        
        return {
            'is_choppy': choppiness_score > 0.7,
            'choppiness_score': choppiness_score,
            'directional_ratio': directional_ratio,
            'efficiency_ratio': efficiency_ratio,
            'reversal_ratio': reversal_ratio,
            'market_state': 'choppy' if choppiness_score > 0.7 else 'trending' if choppiness_score < 0.4 else 'mixed'
        }
        
    except Exception:
        return {'is_choppy': False, 'choppiness_score': 0}

def calculate_momentum_adjusted_entry(current_price: float, df: pd.DataFrame, 
                                    side: str, config: AdaptiveSignalConfig) -> float:
    """FIXED: Calculate entry price WITHOUT chasing momentum"""
    try:
        if len(df) < config.price_momentum_lookback:
            momentum_pct = 0
        else:
            recent = df.tail(config.price_momentum_lookback)
            price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            momentum_pct = price_change
        
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        volatility_adjustment = (atr / current_price) * 0.5
        
        if config.use_pullback_entries:
            # FIXED: Don't chase momentum - wait for pullback
            if side == 'buy':
                if momentum_pct > 0.01:  # Strong upward momentum
                    # Set entry BELOW current price for pullback
                    entry_price = current_price * (1 - config.entry_buffer_from_structure)
                else:
                    # Normal entry slightly below
                    entry_price = current_price * 0.999
            else:  # sell
                if momentum_pct < -0.01:  # Strong downward momentum
                    # Set entry ABOVE current price for pullback
                    entry_price = current_price * (1 + config.entry_buffer_from_structure)
                else:
                    # Normal entry slightly above
                    entry_price = current_price * 1.001
        else:
            # Original logic (not recommended)
            if side == 'buy':
                if momentum_pct > 0.01:
                    entry_adjustment = max(volatility_adjustment, config.momentum_entry_adjustment)
                    entry_price = current_price * (1 + entry_adjustment)
                else:
                    entry_price = current_price * (1 - config.entry_buffer_from_structure)
            else:
                if momentum_pct < -0.01:
                    entry_adjustment = max(volatility_adjustment, config.momentum_entry_adjustment)
                    entry_price = current_price * (1 - entry_adjustment)
                else:
                    entry_price = current_price * (1 + config.entry_buffer_from_structure)
        
        return entry_price
        
    except Exception:
        if side == 'buy':
            return current_price * 0.999
        else:
            return current_price * 1.001

def calculate_dynamic_stop_loss(entry_price: float, current_price: float, 
                              df: pd.DataFrame, side: str, config: AdaptiveSignalConfig,
                              leverage: int = 1) -> float:
    """FIXED: Calculate stop loss with PROPER crypto-appropriate distance"""
    try:
        if 'atr' in df.columns and len(df) >= 14:
            atr = df['atr'].iloc[-1]
            atr_pct = atr / current_price
        else:
            atr_pct = 0.03  # 3% default for crypto
        
        if len(df) >= 20:
            recent_changes = df['close'].pct_change().tail(20).abs()
            avg_volatility = recent_changes.mean()
            max_volatility = recent_changes.max()
        else:
            avg_volatility = 0.02
            max_volatility = 0.04
        
        # FIXED: Proper stop distances for crypto volatility
        if max_volatility > config.extreme_volatility_threshold:
            # Extreme volatility - use very wide stops
            stop_distance_pct = max(atr_pct * 4.0, config.min_stop_distance_pct * 2.0)
        elif max_volatility > config.high_volatility_threshold:
            # High volatility - use wide stops
            stop_distance_pct = max(atr_pct * 3.5, config.min_stop_distance_pct * 1.5)
        elif avg_volatility > 0.03:
            # Medium-high volatility
            stop_distance_pct = max(atr_pct * 3.0, config.min_stop_distance_pct * 1.3)
        elif avg_volatility > 0.015:
            # Medium volatility
            stop_distance_pct = max(atr_pct * 2.5, config.min_stop_distance_pct * 1.1)
        else:
            # Low volatility - still use reasonable stops
            stop_distance_pct = max(atr_pct * 2.0, config.min_stop_distance_pct)
        
        # Apply leverage adjustment
        stop_distance_pct = config.get_leverage_adjusted_stop_distance(stop_distance_pct, leverage)
        
        # Ensure within configured bounds (but with proper crypto limits)
        stop_distance_pct = min(config.max_stop_distance_pct, stop_distance_pct)
        stop_distance_pct = max(config.min_stop_distance_pct, stop_distance_pct)
        
        if side == 'buy':
            stop_loss = entry_price * (1 - stop_distance_pct / 100)
        else:
            stop_loss = entry_price * (1 + stop_distance_pct / 100)
        
        return stop_loss
        
    except Exception:
        # Fallback with proper crypto distances
        if side == 'buy':
            return entry_price * (1 - config.min_stop_distance_pct / 100)
        else:
            return entry_price * (1 + config.min_stop_distance_pct / 100)

# ===== ENHANCED SIGNAL VALIDATOR =====

class EnhancedSignalValidator:
    """Enhanced validator with quality tier assignment"""
    
    def __init__(self, config: AdaptiveSignalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_and_categorize_signal(self, signal: Dict) -> Dict:
        """Validate signal and assign quality tier"""
        
        # Basic validation
        signal = self._validate_basic_requirements(signal)
        
        # Calculate comprehensive score
        quality_score = self._calculate_comprehensive_quality_score(signal)
        signal['quality_score'] = quality_score
        
        # Assign quality tier
        signal['quality_tier'] = self._assign_quality_tier(signal)
        
        # Add risk warnings if needed
        signal['risk_warnings'] = self._identify_risk_warnings(signal)
        
        return signal
    
    def _validate_basic_requirements(self, signal: Dict) -> Dict:
        """Ensure signal has all required fields"""
        required_fields = [
            'symbol', 'side', 'entry_price', 'stop_loss',
            'take_profit_1', 'take_profit_2', 'confidence',
            'risk_reward_ratio'
        ]
        
        for field in required_fields:
            if field not in signal:
                raise ValueError(f"Signal missing required field: {field}")
        
        # Validate stop loss
        entry = signal['entry_price']
        stop = signal['stop_loss']
        side = signal['side']
        
        adjusted, new_stop = self.validate_stop_loss(entry, stop, side)
        if adjusted:
            signal['stop_loss'] = new_stop
            signal['stop_loss_adjusted'] = True
        
        # Validate take profits
        tp1, tp2 = self.validate_take_profits(
            entry, stop, signal['take_profit_1'], 
            signal['take_profit_2'], side
        )
        signal['take_profit_1'] = tp1
        signal['take_profit_2'] = tp2
        
        # Recalculate R/R if adjusted
        if adjusted:
            signal['risk_reward_ratio'] = self.calculate_risk_reward(
                entry, new_stop, tp1, side
            )
        
        return signal
    
    def _calculate_comprehensive_quality_score(self, signal: Dict) -> float:
        """Calculate comprehensive quality score"""
        score = 0.0
        
        # Confidence component (0-30 points)
        confidence = signal.get('confidence', 0)
        score += (confidence / 100) * 30
        
        # Risk/Reward component (0-25 points)
        rr = signal.get('risk_reward_ratio', 0)
        if rr >= 3.0:
            score += 25
        elif rr >= 2.5:
            score += 20
        elif rr >= 2.0:
            score += 15
        elif rr >= 1.5:
            score += 10
        else:
            score += 5
        
        # Volume component (0-15 points)
        volume_24h = signal.get('volume_24h', 0)
        if volume_24h >= 10_000_000:
            score += 15
        elif volume_24h >= 5_000_000:
            score += 10
        elif volume_24h >= 1_000_000:
            score += 5
        
        # MTF validation (0-15 points)
        if signal.get('mtf_validated', False):
            score += 15
        elif signal.get('mtf_status') == 'STRONG':
            score += 10
        elif signal.get('mtf_status') == 'PARTIAL':
            score += 5
        
        # API sentiment (0-10 points)
        intel = signal.get('market_intelligence', {})
        api_sentiment = intel.get('overall_api_sentiment', 50)
        
        if signal['side'] == 'buy' and api_sentiment < 30:
            score += 10  # Contrarian long
        elif signal['side'] == 'sell' and api_sentiment > 70:
            score += 10  # Contrarian short
        elif 40 < api_sentiment < 60:
            score += 5  # Neutral is okay
        
        # Quality factors (0-5 points)
        quality_factors = signal.get('quality_factors', [])
        if quality_factors:
            score += min(5, len(quality_factors))
        
        return score / 100  # Normalize to 0-1
    
    def _assign_quality_tier(self, signal: Dict) -> SignalQualityTier:
        """Assign quality tier based on comprehensive analysis"""
        score = signal.get('quality_score', 0)
        confidence = signal.get('confidence', 0)
        
        if score >= 0.8 and confidence >= 80:
            return SignalQualityTier.PREMIUM
        elif score >= 0.65 and confidence >= 65:
            return SignalQualityTier.STANDARD
        elif score >= 0.5 and confidence >= 50:
            return SignalQualityTier.SPECULATIVE
        else:
            return SignalQualityTier.EXPERIMENTAL
    
    def _identify_risk_warnings(self, signal: Dict) -> List[str]:
        """Identify risk warnings for the signal"""
        warnings = []
        
        # Check for low volume
        if signal.get('volume_24h', 0) < 500_000:
            warnings.append("Low liquidity - wide spreads possible")
        
        # Check for extreme RSI
        rsi = signal.get('analysis', {}).get('rsi', 50)
        if rsi > 80:
            warnings.append("RSI extremely overbought")
        elif rsi < 20:
            warnings.append("RSI extremely oversold")
        
        # Check for API disagreement
        intel = signal.get('market_intelligence', {})
        if signal['side'] == 'buy' and intel.get('fear_greed_index', 50) > 80:
            warnings.append("Extreme greed - contrarian signal")
        elif signal['side'] == 'sell' and intel.get('fear_greed_index', 50) < 20:
            warnings.append("Extreme fear - contrarian signal")
        
        # Check for high volatility
        if signal.get('market_regime') == 'volatile':
            warnings.append("High volatility environment")
        
        return warnings
    
    def validate_stop_loss(self, entry_price: float, stop_loss: float, side: str) -> Tuple[bool, float]:
        """FIXED: Validate stop loss with proper crypto distances"""
        if side == 'buy':
            distance_pct = ((entry_price - stop_loss) / entry_price) * 100
            if distance_pct < self.config.min_stop_distance_pct:
                new_stop = entry_price * (1 - self.config.min_stop_distance_pct / 100)
                return True, new_stop
            elif distance_pct > self.config.max_stop_distance_pct:
                new_stop = entry_price * (1 - self.config.max_stop_distance_pct / 100)
                return True, new_stop
        else:
            distance_pct = ((stop_loss - entry_price) / entry_price) * 100
            if distance_pct < self.config.min_stop_distance_pct:
                new_stop = entry_price * (1 + self.config.min_stop_distance_pct / 100)
                return True, new_stop
            elif distance_pct > self.config.max_stop_distance_pct:
                new_stop = entry_price * (1 + self.config.max_stop_distance_pct / 100)
                return True, new_stop
        
        return False, stop_loss
    
    def validate_take_profits(self, entry_price: float, stop_loss: float,
                            tp1: float, tp2: float, side: str) -> Tuple[float, float]:
        """Validate and adjust take profits"""
        if side == 'buy':
            min_tp1 = entry_price * (1 + self.config.min_tp_distance_pct)
            validated_tp1 = max(tp1, min_tp1)
            validated_tp2 = max(tp2, validated_tp1 * 1.01)
            
            max_tp = entry_price * (1 + self.config.max_tp_distance_pct)
            validated_tp1 = min(validated_tp1, max_tp)
            validated_tp2 = min(validated_tp2, max_tp)
        else:
            min_tp1 = entry_price * (1 - self.config.min_tp_distance_pct)
            validated_tp1 = min(tp1, min_tp1)
            validated_tp2 = min(tp2, validated_tp1 * 0.99)
            
            max_tp = entry_price * (1 - self.config.max_tp_distance_pct)
            validated_tp1 = max(validated_tp1, max_tp)
            validated_tp2 = max(validated_tp2, max_tp)
        
        return validated_tp1, validated_tp2
    
    def calculate_risk_reward(self, entry: float, stop: float, tp: float, side: str) -> float:
        """Calculate risk/reward ratio"""
        if side == 'buy':
            risk = entry - stop
            reward = tp - entry
        else:
            risk = stop - entry
            reward = entry - tp
        
        if risk <= 0:
            return 0
        
        return min(reward / risk, self.config.max_risk_reward)

# ===== MAIN ENHANCED SIGNAL GENERATOR V10.1 =====

class SignalGenerator:
    """
    Enhanced Multi-Timeframe Signal Generator v10.1
    With FIXED stop losses for crypto trading
    """
    
    def __init__(self, config: EnhancedSystemConfig, exchange_manager=None):
        self.config = config
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize adaptive configuration
        self.signal_config = AdaptiveSignalConfig()
        self.validator = EnhancedSignalValidator(self.signal_config)
        
        # Initialize enhanced API intelligence
        self.intel_aggregator = EnhancedMarketIntelligenceAggregator(exchange_manager)
        
        # Performance tracking
        self.performance = PerformanceMetrics()
        self.circuit_breaker = CircuitBreaker()
        
        # Signal cache for memory optimization
        self.signal_cache = deque(maxlen=100)
        
        # Configure timeframes
        self.primary_timeframe = config.timeframe
        self.confirmation_timeframes = config.confirmation_timeframes
        
        if self.confirmation_timeframes:
            tf_minutes = {'1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440}
            sorted_tfs = sorted(self.confirmation_timeframes,
                              key=lambda x: tf_minutes.get(x, 0), reverse=True)
            self.structure_timeframe = sorted_tfs[0]
        else:
            self.structure_timeframe = '6h'
        
        # Mode configuration
        self.relaxed_mode = config.enable_signal_generation_relaxed_mode
        self.debug_mode = False
        
        self.logger.info("✅ Enhanced Signal Generator v10.1 initialized (FIXED STOP LOSSES)")
        self.logger.info(f"   Mode: {'RELAXED (Scoring)' if self.relaxed_mode else 'STRICT (All conditions)'}")
        self.logger.info(f"   Primary TF: {self.primary_timeframe}")
        self.logger.info(f"   Structure TF: {self.structure_timeframe}")
        self.logger.info(f"   Stop Distances: {self.signal_config.min_stop_distance_pct}%-{self.signal_config.max_stop_distance_pct}% (crypto-appropriate)")
        self.logger.info(f"   Adaptive parameters: ENABLED")
        self.logger.info(f"   Circuit breaker: ENABLED")
        self.logger.info(f"   API intelligence: ENHANCED with fallbacks")
    
    def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol_data: Dict,
                                   volume_entry: Dict, fibonacci_data: Dict,
                                   confluence_zones: List[Dict], timeframe: str) -> Optional[Dict]:
        """Main entry point with comprehensive analysis"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # PHASE 1: Adapt parameters to current market
            self._adapt_to_current_market(df, symbol_data)
            
            # PHASE 2: Gather Market Intelligence (with fallbacks)
            try:
                # Try async first if event loop available
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        intel = asyncio.create_task(
                            self.intel_aggregator.gather_intelligence_async(symbol)
                        )
                    else:
                        intel = loop.run_until_complete(
                            self.intel_aggregator.gather_intelligence_async(symbol)
                        )
                except:
                    # Fallback to sync
                    intel = self.intel_aggregator.gather_intelligence(symbol)
            except Exception as e:
                self.logger.warning(f"Intelligence gathering failed: {e}")
                intel = MarketIntelligence()
            
            # PHASE 3: Market regime detection
            market_regime = self._determine_market_regime(symbol_data, df)
            
            # PHASE 4: Multi-timeframe context
            mtf_context = self._get_multitimeframe_context(symbol_data, market_regime)
            if not mtf_context:
                mtf_context = self._create_fallback_context(symbol_data, market_regime)
            
            # PHASE 5: Generate signal based on mode
            signal = None
            
            if self._should_use_mtf_signals(mtf_context):
                signal = self._generate_mtf_aware_signal(
                    df, symbol_data, volume_entry, fibonacci_data,
                    confluence_zones, mtf_context, intel
                )
            
            if not signal:
                signal = self._generate_traditional_signal(
                    df, symbol_data, volume_entry, fibonacci_data,
                    confluence_zones, intel
                )
            
            if signal:
                # PHASE 6: Validate and enhance signal
                signal = self.validator.validate_and_categorize_signal(signal)
                signal = self._enhance_signal_with_analysis(
                    signal, mtf_context, df, market_regime, intel
                )
                
                # PHASE 7: Check circuit breaker
                allowed, reason = self.circuit_breaker.check_signal_allowed(
                    signal, self.performance
                )
                
                if not allowed:
                    # self.logger.warning(f"Signal blocked by circuit breaker: {reason}")
                    return None
                
                # PHASE 8: Optimize memory - create summary instead of storing DataFrame
                signal = self._optimize_signal_memory(signal, df)
                
                # Record signal for performance tracking
                self.circuit_breaker.hourly_signals.append(datetime.now())
                
                # PHASE 9: Create comprehensive analysis
                signal['analysis'] = self._create_comprehensive_analysis(
                    df, symbol_data, volume_entry, fibonacci_data,
                    confluence_zones, mtf_context
                )
                
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def _adapt_to_current_market(self, df: pd.DataFrame, symbol_data: Dict) -> None:
        """Adapt parameters to current market conditions"""
        try:
            # Calculate market metrics
            volatility = self._calculate_market_volatility(df)
            volume = symbol_data.get('volume_24h', 0)
            trend_strength = self._calculate_trend_strength_simple(df)
            
            # Get recent performance
            recent_win_rate = self.performance.recent_win_rate
            
            # Adapt configuration
            self.signal_config.adapt_to_market(
                volatility, volume, trend_strength, recent_win_rate
            )
            
            if self.debug_mode:
                self.logger.info(f"📊 Market adaptation: {self.signal_config.current_market_condition}")
                self.logger.info(f"   Volatility: {volatility:.2%}")
                self.logger.info(f"   Win rate: {recent_win_rate:.1%}")
                self.logger.info(f"   Min R/R: {self.signal_config.min_risk_reward}")
                self.logger.info(f"   Stop distance: {self.signal_config.min_stop_distance_pct}%-{self.signal_config.max_stop_distance_pct}%")
                
        except Exception as e:
            self.logger.error(f"Market adaptation error: {e}")
    
    def _generate_traditional_signal(self, df: pd.DataFrame, symbol_data: Dict,
                                   volume_entry: Dict, fibonacci_data: Dict,
                                   confluence_zones: List[Dict],
                                   intel: MarketIntelligence) -> Optional[Dict]:
        """Generate signal with STRICT MTF validation - NONE signals blocked"""
        try:
            latest = df.iloc[-1]
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Extract indicators
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            stoch_k = latest.get('stoch_rsi_k', 50)
            stoch_d = latest.get('stoch_rsi_d', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # Check for extreme volatility
            if 'atr' in df.columns:
                atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
                if atr_pct > 10:  # Skip signal in extreme volatility
                    self.logger.debug(f"❌ {symbol} - Extreme volatility {atr_pct:.1f}%, skipping")
                    return None
            
            # Initialize MTF validator if required
            mtf_validator = None
            mtf_data = {}
            
            if self.config.mtf_confirmation_required:
                mtf_validator = StrictMTFValidator(self.config.confirmation_timeframes)
                
                # Fetch MTF data early for pre-validation
                self.logger.debug(f"🔍 Fetching MTF data for {symbol}")
                for tf in self.config.confirmation_timeframes:
                    try:
                        tf_df = self.exchange_manager.fetch_ohlcv_data(symbol, tf)
                        if not tf_df.empty and len(tf_df) >= 50:
                            # Calculate indicators for MTF
                            tf_df = self._calculate_comprehensive_indicators(tf_df)
                            mtf_data[tf] = tf_df
                            self.logger.debug(f"   ✅ {tf} data fetched: {len(tf_df)} candles")
                        else:
                            self.logger.debug(f"   ⚠️ {tf} data insufficient")
                    except Exception as e:
                        self.logger.debug(f"   ❌ {tf} fetch failed: {e}")
                        continue
                
                # If no MTF data available, return None
                if not mtf_data:
                    self.logger.debug(f"❌ {symbol} - No MTF data available, blocking signal")
                    return None
            
            # Determine preliminary signal direction
            preliminary_side = None
            signal_factors = []
            
            if self.relaxed_mode:
                # ===== RELAXED MODE: SCORING SYSTEM =====
                long_score = 0
                short_score = 0
                long_factors = []
                short_factors = []
                
                # RSI scoring (weight: 2 points)
                if self.signal_config.min_rsi_for_long < rsi < self.signal_config.max_rsi_for_long:
                    long_score += 2
                    long_factors.append(f"RSI in range ({rsi:.1f})")
                elif rsi < self.signal_config.max_rsi_for_long + 5:
                    long_score += 1
                    long_factors.append(f"RSI acceptable ({rsi:.1f})")
                
                if self.signal_config.min_rsi_for_short < rsi < self.signal_config.max_rsi_for_short:
                    short_score += 2
                    short_factors.append(f"RSI in range ({rsi:.1f})")
                elif rsi > self.signal_config.min_rsi_for_short - 5:
                    short_score += 1
                    short_factors.append(f"RSI acceptable ({rsi:.1f})")
                
                # MACD scoring (weight: 1.5 points)
                if macd > macd_signal:
                    long_score += 1.5
                    long_factors.append("MACD bullish")
                else:
                    short_score += 1.5
                    short_factors.append("MACD bearish")
                
                # Stochastic scoring (weight: 1 point)
                if stoch_k > stoch_d:
                    long_score += 1
                    long_factors.append(f"Stoch bullish ({stoch_k:.1f})")
                else:
                    short_score += 1
                    short_factors.append(f"Stoch bearish ({stoch_k:.1f})")
                
                if stoch_k < self.signal_config.max_stoch_for_long:
                    long_score += 0.5
                if stoch_k > self.signal_config.min_stoch_for_short:
                    short_score += 0.5
                
                # Volume scoring (weight: 1 point)
                if volume_ratio > self.signal_config.strong_volume_threshold:
                    long_score += 1
                    short_score += 1
                    long_factors.append(f"Strong volume ({volume_ratio:.2f})")
                    short_factors.append(f"Strong volume ({volume_ratio:.2f})")
                elif volume_ratio > self.signal_config.min_volume_ratio_for_signal:
                    long_score += 0.5
                    short_score += 0.5
                
                # Price momentum bonus
                momentum = analyze_price_momentum_strength(df)
                if momentum['direction'] == 'bullish':
                    long_score += 0.5
                    long_factors.append("Bullish momentum")
                elif momentum['direction'] == 'bearish':
                    short_score += 0.5
                    short_factors.append("Bearish momentum")
                
                # Minimum score needed (tightened)
                min_score_needed = 4.5 * self.signal_config.adaptation_factor
                
                if self.debug_mode:
                    self.logger.info(f"📊 Scoring: Long={long_score:.1f}, Short={short_score:.1f} (need {min_score_needed:.1f})")
                
                if long_score >= min_score_needed and long_score > short_score:
                    preliminary_side = 'buy'
                    signal_factors = long_factors
                elif short_score >= min_score_needed and short_score > long_score:
                    preliminary_side = 'sell'
                    signal_factors = short_factors
                
            else:
                # ===== STRICT MODE: REQUIRE ALL CONDITIONS =====
                long_conditions = [
                    self.signal_config.min_rsi_for_long < rsi < self.signal_config.max_rsi_for_long,
                    macd > macd_signal,
                    stoch_k < self.signal_config.max_stoch_for_long,
                    stoch_d < self.signal_config.max_stoch_for_long,
                    volume_ratio > self.signal_config.min_volume_ratio_for_signal
                ]
                
                short_conditions = [
                    self.signal_config.min_rsi_for_short < rsi < self.signal_config.max_rsi_for_short,
                    macd < macd_signal,
                    stoch_k > self.signal_config.min_stoch_for_short,
                    stoch_d > self.signal_config.min_stoch_for_short,
                    volume_ratio > self.signal_config.min_volume_ratio_for_signal
                ]
                
                if all(long_conditions):
                    preliminary_side = 'buy'
                    signal_factors.append("All strict conditions met")
                elif all(short_conditions):
                    preliminary_side = 'sell'
                    signal_factors.append("All strict conditions met")
                else:
                    if self.debug_mode:
                        self.logger.info(f"❌ Conditions not met - Long: {sum(long_conditions)}/5, Short: {sum(short_conditions)}/5")
            
            # If no preliminary signal, return None
            if not preliminary_side:
                self.logger.debug(f"❌ {symbol} - No preliminary signal generated")
                return None
            
            # ===== STRICT MTF VALIDATION =====
            if self.config.mtf_confirmation_required and mtf_validator:
                self.logger.debug(f"🎯 Validating {preliminary_side.upper()} signal for {symbol} with MTF")
                
                # Calculate preliminary TP1 for validation
                if preliminary_side == 'buy':
                    preliminary_tp1 = current_price * 1.02  # Rough 2% estimate
                else:
                    preliminary_tp1 = current_price * 0.98
                
                # Perform MTF validation
                validation_result = mtf_validator.validate_signal_with_mtf(
                    preliminary_side, df, mtf_data, current_price, preliminary_tp1
                )
                
                # STRICT: Block if MTF status is NONE
                if validation_result.mtf_status == 'NONE':
                    self.logger.info(f"❌ {symbol} - BLOCKED: MTF status NONE | {validation_result.rejection_reason}")
                    return None
                
                # Block if validation failed
                if not validation_result.is_valid:
                    self.logger.info(f"❌ {symbol} - Failed MTF validation: {validation_result.rejection_reason}")
                    return None
                
                # Block if TP1 reachability is too low
                if validation_result.tp1_reachability_score < 0.6:
                    self.logger.info(f"❌ {symbol} - Low TP1 reachability: {validation_result.tp1_reachability_score:.1%}")
                    return None
                
                self.logger.info(f"✅ {symbol} - MTF validation passed: {validation_result.mtf_status} "
                            f"(alignment: {validation_result.alignment_score:.1%}, "
                            f"TP1 reach: {validation_result.tp1_reachability_score:.1%})")
                
                # Create MTF context from validation
                mtf_context = MultiTimeframeContext(
                    dominant_trend=validation_result.dominant_bias,
                    trend_strength=validation_result.alignment_score,
                    higher_tf_zones=validation_result.structure_analysis.get('zones', []) if validation_result.structure_analysis else [],
                    key_support=current_price * 0.95,  # Will be refined in signal generation
                    key_resistance=current_price * 1.05,
                    momentum_alignment=validation_result.alignment_score > 0.6,
                    entry_bias='long_favored' if validation_result.dominant_bias == 'bullish' else 
                            'short_favored' if validation_result.dominant_bias == 'bearish' else 'neutral',
                    confirmation_score=validation_result.alignment_score,
                    structure_timeframe=self.structure_timeframe,
                    market_regime=validation_result.dominant_bias,
                    volatility_level='medium'
                )
                
                # Generate the actual signal with MTF context
                signal = None
                if preliminary_side == 'buy':
                    signal = self._generate_long_signal(
                        symbol_data, latest, mtf_context, volume_entry,
                        confluence_zones, df, intel, use_mtf=True
                    )
                else:
                    signal = self._generate_short_signal(
                        symbol_data, latest, mtf_context, volume_entry,
                        confluence_zones, df, intel, use_mtf=True
                    )
                
                if signal:
                    # Final validation with actual TP1
                    final_validation = mtf_validator.validate_signal_with_mtf(
                        signal['side'], df, mtf_data, 
                        signal['entry_price'], signal['take_profit_1']
                    )
                    
                    # Final TP1 reachability check
                    if final_validation.tp1_reachability_score < 0.6:
                        self.logger.info(f"❌ {symbol} - Final check failed: TP1 reachability {final_validation.tp1_reachability_score:.1%}")
                        return None
                    
                    # Final MTF status check
                    if final_validation.mtf_status == 'NONE':
                        self.logger.info(f"❌ {symbol} - Final check failed: MTF status became NONE")
                        return None
                    
                    # Update signal with validation info
                    signal['mtf_status'] = final_validation.mtf_status
                    signal['mtf_validated'] = True
                    signal['mtf_validation_score'] = final_validation.alignment_score
                    signal['tp1_reachability'] = final_validation.tp1_reachability_score
                    signal['dominant_market_bias'] = final_validation.dominant_bias
                    signal['mtf_alignment_details'] = validation_result.structure_analysis
                    signal['signal_factors'] = signal_factors
                    
                    self.logger.info(f"🎯 {symbol} - Signal generated: {signal['side'].upper()} "
                                f"@ {signal['entry_price']:.6f} | "
                                f"SL: {signal['stop_loss']:.6f} | "
                                f"TP1: {signal['take_profit_1']:.6f} | "
                                f"MTF: {signal['mtf_status']} | "
                                f"TP1 Reach: {signal['tp1_reachability']:.1%}")
                    
                    return signal
                else:
                    self.logger.debug(f"❌ {symbol} - Signal generation failed after MTF validation")
                    return None
            
            # If MTF not required but we still want to block non-MTF signals
            elif self.config.mtf_confirmation_required:
                self.logger.debug(f"❌ {symbol} - MTF required but validator not available")
                return None
            
            # Fallback: Create signal without MTF (only if explicitly allowed)
            if not self.config.mtf_confirmation_required:
                # Create basic context
                mtf_context = self._create_fallback_context(symbol_data, 'unknown')
                
                if preliminary_side == 'buy':
                    return self._generate_long_signal(
                        symbol_data, latest, mtf_context, volume_entry,
                        confluence_zones, df, intel, use_mtf=False
                    )
                else:
                    return self._generate_short_signal(
                        symbol_data, latest, mtf_context, volume_entry,
                        confluence_zones, df, intel, use_mtf=False
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Traditional signal generation error for {symbol_data.get('symbol', 'unknown')}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _create_mtf_context_from_validation(self, validation_result) -> MultiTimeframeContext:
        """Create MTF context from validation result"""
        return MultiTimeframeContext(
            dominant_trend=validation_result.dominant_bias,
            trend_strength=validation_result.alignment_score,
            higher_tf_zones=[],
            key_support=0,
            key_resistance=0,
            momentum_alignment=validation_result.alignment_score > 0.6,
            entry_bias='long_favored' if validation_result.dominant_bias == 'bullish' else 'short_favored',
            confirmation_score=validation_result.alignment_score,
            structure_timeframe=self.structure_timeframe,
            market_regime=validation_result.dominant_bias,
            volatility_level='medium'
        )
    
    def _generate_long_signal(self, symbol_data: Dict, latest: pd.Series,
                                mtf_context: MultiTimeframeContext, volume_entry: Dict,
                                confluence_zones: List[Dict], df: pd.DataFrame,
                                intel: MarketIntelligence, use_mtf: bool = True) -> Optional[Dict]:
        """Generate LONG signal with FIXED stop losses"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Apply quality filters
            should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium = \
                self._apply_quality_filters(df, 'buy', symbol)
            
            if should_reject and quality_score < self.signal_config.min_quality_score_to_reject:
                return None
            
            # Check entry timing
            timing_check = self._validate_entry_timing(df, 'buy')
            if not timing_check['valid'] and timing_check['score'] < -0.5:
                return None
            
            # Estimate leverage for stop calculation (from autotrader context)
            estimated_leverage = 25  # Default assumption, could be made configurable
            
            # Calculate entry, stop, and targets
            entry_price = self._calculate_long_entry(current_price, mtf_context, volume_entry, df)
            raw_stop = self._calculate_long_stop(entry_price, mtf_context, df, estimated_leverage)
            adjusted, stop_loss = self.validator.validate_stop_loss(entry_price, raw_stop, 'buy')
            
            raw_tp1, raw_tp2 = self._calculate_long_targets(entry_price, stop_loss, mtf_context, df, intel)
            tp1, tp2 = self.validator.validate_take_profits(entry_price, stop_loss, raw_tp1, raw_tp2, 'buy')
            
            tp = tp1 if self.config.default_tp_level == 'take_profit_1' else tp2
            
            # Calculate R/R ratio
            rr_ratio = self.validator.calculate_risk_reward(entry_price, stop_loss, tp, 'buy')
            
            if rr_ratio < self.signal_config.min_risk_reward:
                if self.relaxed_mode and rr_ratio >= 1.2:
                    # Allow in relaxed mode with penalty
                    quality_score -= 1
                else:
                    return None
            
            # Calculate weighted confidence
            confidence = self._calculate_weighted_confidence(
                'buy', latest, use_mtf, mtf_context, timing_check,
                quality_score, is_premium, intel
            )
            
            if confidence < self.signal_config.min_confidence_for_signal:
                return None
            
            # Determine order type
            entry_distance_pct = abs(entry_price - current_price) / current_price
            order_type = 'limit' if entry_distance_pct > 0.01 else 'market'
            
            # Check divergence
            divergence = detect_divergence(df, 'buy')
            
            return {
                'symbol': symbol,
                'side': 'buy',
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'take_profit': tp,
                'risk_reward_ratio': rr_ratio,
                'confidence': confidence,
                'original_confidence': confidence - (self.signal_config.mtf_confidence_boost if use_mtf else 0),
                'signal_type': 'long_signal',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'order_type': order_type,
                'mtf_validated': use_mtf,
                'market_regime': mtf_context.market_regime,
                'regime_compatibility': self._assess_regime_compatibility('buy', mtf_context.market_regime),
                'entry_timing': timing_check,
                'quality_factors': enhancement_factors,
                'quality_score': quality_score,
                'is_premium_signal': is_premium,
                'divergence': divergence,
                'mtf_boost': self.signal_config.mtf_confidence_boost if use_mtf else 0,
                'stop_distance_pct': ((entry_price - stop_loss) / entry_price) * 100,
                'estimated_leverage': estimated_leverage
            }
            
        except Exception as e:
            self.logger.error(f"Long signal generation error: {e}")
            return None
    
    def _generate_short_signal(self, symbol_data: Dict, latest: pd.Series,
                                 mtf_context: MultiTimeframeContext, volume_entry: Dict,
                                 confluence_zones: List[Dict], df: pd.DataFrame,
                                 intel: MarketIntelligence, use_mtf: bool = True) -> Optional[Dict]:
        """Generate SHORT signal with FIXED stop losses"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Apply quality filters
            should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium = \
                self._apply_quality_filters(df, 'sell', symbol)
            
            if should_reject and quality_score < self.signal_config.min_quality_score_to_reject:
                return None
            
            # Check entry timing
            timing_check = self._validate_entry_timing(df, 'sell')
            if not timing_check['valid'] and timing_check['score'] < -0.5:
                return None
            
            # Estimate leverage for stop calculation
            estimated_leverage = 25  # Default assumption
            
            # Calculate entry, stop, and targets
            entry_price = self._calculate_short_entry(current_price, mtf_context, volume_entry, df)
            raw_stop = self._calculate_short_stop(entry_price, mtf_context, df, estimated_leverage)
            adjusted, stop_loss = self.validator.validate_stop_loss(entry_price, raw_stop, 'sell')
            
            raw_tp1, raw_tp2 = self._calculate_short_targets(entry_price, stop_loss, mtf_context, df, intel)
            tp1, tp2 = self.validator.validate_take_profits(entry_price, stop_loss, raw_tp1, raw_tp2, 'sell')
            
            tp = tp1 if self.config.default_tp_level == 'take_profit_1' else tp2
            
            # Calculate R/R ratio
            rr_ratio = self.validator.calculate_risk_reward(entry_price, stop_loss, tp, 'sell')
            
            if rr_ratio < self.signal_config.min_risk_reward:
                if self.relaxed_mode and rr_ratio >= 1.2:
                    quality_score -= 1
                else:
                    return None
            
            # Calculate weighted confidence
            confidence = self._calculate_weighted_confidence(
                'sell', latest, use_mtf, mtf_context, timing_check,
                quality_score, is_premium, intel
            )
            
            if confidence < self.signal_config.min_confidence_for_signal:
                return None
            
            # Determine order type
            entry_distance_pct = abs(entry_price - current_price) / current_price
            order_type = 'limit' if entry_distance_pct > 0.01 else 'market'
            
            # Check divergence
            divergence = detect_divergence(df, 'sell')
            
            return {
                'symbol': symbol,
                'side': 'sell',
                'entry_price': entry_price,
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'take_profit': tp,
                'risk_reward_ratio': rr_ratio,
                'confidence': confidence,
                'original_confidence': confidence - (self.signal_config.mtf_confidence_boost if use_mtf else 0),
                'signal_type': 'short_signal',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'order_type': order_type,
                'mtf_validated': use_mtf,
                'market_regime': mtf_context.market_regime,
                'regime_compatibility': self._assess_regime_compatibility('sell', mtf_context.market_regime),
                'entry_timing': timing_check,
                'quality_factors': enhancement_factors,
                'quality_score': quality_score,
                'is_premium_signal': is_premium,
                'divergence': divergence,
                'mtf_boost': self.signal_config.mtf_confidence_boost if use_mtf else 0,
                'stop_distance_pct': ((stop_loss - entry_price) / entry_price) * 100,
                'estimated_leverage': estimated_leverage
            }
            
        except Exception as e:
            self.logger.error(f"Short signal generation error: {e}")
            return None
    
    def _calculate_weighted_confidence(self, side: str, latest: pd.Series,
                                      use_mtf: bool, mtf_context: MultiTimeframeContext,
                                      timing_check: Dict, quality_score: float,
                                      is_premium: bool, intel: MarketIntelligence) -> float:
        """Calculate confidence with consistent weighted system"""
        
        # Base confidence starts at 50
        components = {
            'base': 50.0,
            'rsi': 0,
            'volume': 0,
            'mtf': 0,
            'timing': 0,
            'quality': 0,
            'api': 0
        }
        
        rsi = latest.get('rsi', 50)
        volume_ratio = latest.get('volume_ratio', 1)
        
        # RSI component (0-15 points)
        if side == 'buy':
            if rsi < 30:
                components['rsi'] = 15
            elif rsi < 35:
                components['rsi'] = 10
            elif rsi < 40:
                components['rsi'] = 5
        else:
            if rsi > 70:
                components['rsi'] = 15
            elif rsi > 65:
                components['rsi'] = 10
            elif rsi > 60:
                components['rsi'] = 5
        
        # Volume component (0-10 points)
        if volume_ratio > 2.0:
            components['volume'] = 10
        elif volume_ratio > 1.5:
            components['volume'] = 7
        elif volume_ratio > 1.0:
            components['volume'] = 4
        
        # MTF component (0-10 points)
        if use_mtf and mtf_context.momentum_alignment:
            components['mtf'] = self.signal_config.mtf_confidence_boost
        
        # Timing component (-5 to +5 points)
        components['timing'] = timing_check['score'] * 10
        
        # Quality component (0-10 points)
        if is_premium:
            components['quality'] = 10
        else:
            components['quality'] = max(0, quality_score * 5)
        
        # API component (0-10 points)
        if side == 'buy' and intel.fear_greed_index < 30:
            components['api'] = 8
        elif side == 'sell' and intel.fear_greed_index > 70:
            components['api'] = 8
        elif 40 < intel.fear_greed_index < 60:
            components['api'] = 4
        
        # Apply API modifier
        total_confidence = sum(components.values())
        total_confidence *= intel.api_signal_modifier
        
        # Apply market condition modifier
        total_confidence *= self.signal_config.adaptation_factor
        
        # Ensure within bounds
        confidence = min(95, max(self.signal_config.min_confidence_for_signal, total_confidence))
        
        if self.debug_mode:
            self.logger.info(f"Confidence breakdown: {components}")
            self.logger.info(f"Final confidence: {confidence:.1f}%")
        
        return confidence
    
    def _apply_quality_filters(self, df: pd.DataFrame, side: str, symbol: str) -> tuple:
        """Apply enhanced quality filters"""
        try:
            rejection_reasons = []
            enhancement_factors = []
            quality_score = 0
            
            # All existing quality checks from v9
            momentum = analyze_price_momentum_strength(df)
            divergence = check_volume_momentum_divergence(df)
            fast_setup = identify_fast_moving_setup(df, side)
            choppiness = filter_choppy_markets(df)
            
            # Momentum check
            if side == 'buy' and momentum['direction'] == 'bearish' and momentum['strength'] >= 3:
                rejection_reasons.append(f"Strong bearish momentum")
                quality_score -= 2
            elif side == 'sell' and momentum['direction'] == 'bullish' and momentum['strength'] >= 3:
                rejection_reasons.append(f"Strong bullish momentum")
                quality_score -= 2
            elif momentum['speed'] in ['fast', 'very_fast']:
                enhancement_factors.append(f"Strong {side} momentum")
                quality_score += 2
            
            # Divergence check
            if divergence['divergence'] and divergence['type'] == 'bearish_divergence' and side == 'buy':
                rejection_reasons.append(divergence['warning'])
                quality_score -= 1
            elif divergence['type'] == 'confirmed_move':
                enhancement_factors.append("Volume confirms move")
                quality_score += 1
            
            # Fast setup bonus
            if fast_setup['is_fast_setup']:
                enhancement_factors.extend(fast_setup['factors'])
                quality_score += fast_setup['score']
            
            # Choppiness filter - RELAXED for crypto
            if choppiness['is_choppy'] and choppiness['choppiness_score'] > 0.8:  # Was 0.75
                rejection_reasons.append(f"Very choppy market")
                quality_score -= 1.5
            elif choppiness['market_state'] == 'trending':
                enhancement_factors.append("Clear trending market")
                quality_score += 1
            
            # Determine if should reject (adaptive threshold)
            should_reject = len(rejection_reasons) > 0 and quality_score < self.signal_config.min_quality_score_to_reject
            is_premium = quality_score >= 3.0
            
            return should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium
            
        except Exception as e:
            self.logger.error(f"Quality filter error: {e}")
            return False, [], [], 0, False
    
    def _optimize_signal_memory(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Optimize signal memory by storing summary instead of full DataFrame"""
        
        # Create lightweight summary
        if '_chart_data' in signal:
            del signal['_chart_data']
        
        # Store only essential price data
        signal['price_summary'] = {
            'recent_high': float(df['high'].tail(20).max()),
            'recent_low': float(df['low'].tail(20).min()),
            'last_close': float(df['close'].iloc[-1]),
            'last_volume': float(df['volume'].iloc[-1]),
            'avg_volume': float(df['volume'].tail(20).mean())
        }
        
        # Store only last indicators
        latest = df.iloc[-1]
        signal['indicator_summary'] = {
            'rsi': float(latest.get('rsi', 50)),
            'macd': float(latest.get('macd', 0)),
            'macd_signal': float(latest.get('macd_signal', 0)),
            'stoch_k': float(latest.get('stoch_rsi_k', 50)),
            'volume_ratio': float(latest.get('volume_ratio', 1))
        }
        
        return signal
    
    def _enhance_signal_with_analysis(self, signal: Dict, mtf_context: MultiTimeframeContext,
                                     df: pd.DataFrame, market_regime: str,
                                     intel: MarketIntelligence) -> Dict:
        """Enhance signal with comprehensive analysis"""
        
        signal['analysis_details'] = {
            'signal_strength': self._determine_signal_strength(signal, mtf_context),
            'mtf_trend': mtf_context.dominant_trend,
            'structure_timeframe': mtf_context.structure_timeframe,
            'confirmation_score': mtf_context.confirmation_score,
            'entry_method': self._determine_entry_method(signal, mtf_context),
            'market_regime': market_regime,
            'volatility_level': mtf_context.volatility_level,
            'momentum_strength': self._analyze_momentum_strength(df),
            'api_sentiment': intel.overall_api_sentiment,
            'fear_greed': intel.fear_greed_classification,
            'funding_sentiment': intel.funding_sentiment,
            'news_sentiment': intel.news_classification
        }
        
        signal['entry_strategy'] = signal['analysis_details']['entry_method']
        signal['quality_grade'] = self._calculate_quality_grade(signal)
        
        if signal.get('mtf_validated', False):
            signal['mtf_status'] = 'MTF_VALIDATED'
        else:
            if signal['confidence'] >= 70:
                signal['mtf_status'] = 'STRONG'
            elif signal['confidence'] >= 60:
                signal['mtf_status'] = 'PARTIAL'
            else:
                signal['mtf_status'] = 'NONE'
        
        signal['market_intelligence'] = {
            'fear_greed_index': intel.fear_greed_index,
            'fear_greed_classification': intel.fear_greed_classification,
            'funding_rate': intel.funding_rate,
            'funding_sentiment': intel.funding_sentiment,
            'news_sentiment': intel.news_sentiment_score,
            'news_classification': intel.news_classification,
            'overall_api_sentiment': intel.overall_api_sentiment,
            'is_cached': intel.is_cached
        }
        
        signal['timestamp'] = pd.Timestamp.now()
        signal['version'] = 'v10.1-fixed'
        
        return signal
    
    def record_signal_result(self, signal_id: str, profitable: bool, profit_amount: float = 0) -> None:
        """Record signal result for performance tracking"""
        
        self.performance.total_signals += 1
        
        if profitable:
            self.performance.winning_signals += 1
            self.circuit_breaker.record_signal_result(True)
        else:
            self.performance.losing_signals += 1
            self.circuit_breaker.record_signal_result(False)
        
        self.performance.total_profit += profit_amount
        
        # Update recent signals
        self.performance.recent_signals.append({
            'signal_id': signal_id,
            'profitable': profitable,
            'profit': profit_amount,
            'timestamp': datetime.now()
        })
        
        # Update drawdown
        if profit_amount < 0:
            self.performance.current_drawdown += abs(profit_amount)
            self.performance.max_drawdown = max(
                self.performance.max_drawdown,
                self.performance.current_drawdown
            )
        else:
            self.performance.current_drawdown = max(
                0, self.performance.current_drawdown - profit_amount
            )
        
        # Log performance update
        if self.performance.total_signals % 10 == 0:
            self.logger.info(f"📊 Performance Update:")
            self.logger.info(f"   Total signals: {self.performance.total_signals}")
            self.logger.info(f"   Win rate: {self.performance.win_rate:.1%}")
            self.logger.info(f"   Recent win rate: {self.performance.recent_win_rate:.1%}")
            self.logger.info(f"   Total profit: {self.performance.total_profit:.2f}")
            self.logger.info(f"   Max drawdown: {self.performance.max_drawdown:.2f}")
    
    # ===== ALL REMAINING METHODS FROM V10 =====
    
    def _should_use_mtf_signals(self, mtf_context: MultiTimeframeContext) -> bool:
        """Determine if MTF signals should be used"""
        return (mtf_context.entry_bias != 'avoid' and 
                mtf_context.confirmation_score > 0.3 and
                self.exchange_manager is not None)
    
    def _generate_mtf_aware_signal(self, df: pd.DataFrame, symbol_data: Dict,
                             volume_entry: Dict, fibonacci_data: Dict,
                             confluence_zones: List[Dict], 
                             mtf_context: MultiTimeframeContext,
                             intel: MarketIntelligence) -> Optional[Dict]:
        """Generate signal with STRICT MTF awareness"""
        try:
            mtf_validator = StrictMTFValidator(self.config.confirmation_timeframes)
            
            # Only generate signals aligned with dominant bias
            if mtf_context.dominant_trend == 'bearish' and mtf_context.entry_bias != 'short_favored':
                self.logger.debug("❌ Blocking signal - market is bearish but entry bias not short")
                return None
            
            if mtf_context.dominant_trend == 'bullish' and mtf_context.entry_bias != 'long_favored':
                self.logger.debug("❌ Blocking signal - market is bullish but entry bias not long")
                return None
            
            # Generate signal based on bias
            signal = None
            latest = df.iloc[-1]
            
            if mtf_context.entry_bias == 'long_favored' and mtf_context.dominant_trend != 'bearish':
                signal = self._generate_long_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=True
                )
            elif mtf_context.entry_bias == 'short_favored' and mtf_context.dominant_trend != 'bullish':
                signal = self._generate_short_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=True
                )
            
            # Validate the generated signal
            if signal:
                # Fetch MTF data for final validation
                mtf_data = {}
                for tf in self.config.confirmation_timeframes:
                    try:
                        tf_df = self.exchange_manager.fetch_ohlcv_data(symbol_data['symbol'], tf)
                        if not tf_df.empty:
                            tf_df = self._calculate_comprehensive_indicators(tf_df)
                            mtf_data[tf] = tf_df
                    except:
                        continue
                
                # Final validation
                validation = mtf_validator.validate_signal_with_mtf(
                    signal['side'], df, mtf_data,
                    signal['entry_price'], signal['take_profit_1']
                )
                
                # STRICT: Must have at least PARTIAL status
                if validation.mtf_status == 'NONE':
                    self.logger.debug(f"❌ MTF signal blocked - status NONE")
                    return None
                
                if validation.tp1_reachability_score < 0.6:
                    self.logger.debug(f"❌ MTF signal blocked - low TP1 reachability")
                    return None
                
                # Update signal
                signal['mtf_status'] = validation.mtf_status
                signal['mtf_validated'] = True
                signal['tp1_reachability'] = validation.tp1_reachability_score
                
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"MTF signal generation error: {e}")
            return None
    
    def _validate_entry_timing(self, df: pd.DataFrame, side: str, lookback: int = 5) -> dict:
        """Validate if current timing is good for entry"""
        try:
            if len(df) < lookback:
                return {'valid': True, 'reason': 'insufficient_data', 'score': 0.5}
            
            recent = df.tail(lookback)
            latest = df.iloc[-1]
            
            recent_high = recent['high'].max()
            recent_low = recent['low'].min()
            current_price = latest['close']
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            rsi = latest.get('rsi', 50)
            stoch_k = latest.get('stoch_rsi_k', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            timing_score = 0
            reasons = []
            
            if side == 'buy':
                if price_position > 0.9:
                    timing_score -= 0.3
                    reasons.append("price_near_resistance")
                elif price_position < 0.3:
                    timing_score += 0.3
                    reasons.append("price_near_support")
                
                if rsi > self.signal_config.max_rsi_for_long:
                    timing_score -= 0.3
                    reasons.append("rsi_getting_high")
                elif rsi < 35:
                    timing_score += 0.2
                    reasons.append("rsi_oversold_good")
                
                if stoch_k > 70:
                    timing_score -= 0.2
                    reasons.append("stoch_overbought")
                elif stoch_k < 30:
                    timing_score += 0.2
                    reasons.append("stoch_oversold_good")
                    
            else:  # sell
                if price_position < 0.1:
                    timing_score -= 0.3
                    reasons.append("price_near_support")
                elif price_position > 0.7:
                    timing_score += 0.3
                    reasons.append("price_near_resistance")
                
                if rsi < self.signal_config.min_rsi_for_short:
                    timing_score -= 0.3
                    reasons.append("rsi_getting_low")
                elif rsi > 65:
                    timing_score += 0.2
                    reasons.append("rsi_overbought_good")
                
                if stoch_k < 30:
                    timing_score -= 0.2
                    reasons.append("stoch_oversold")
                elif stoch_k > 70:
                    timing_score += 0.2
                    reasons.append("stoch_overbought_good")
            
            if volume_ratio > 1.5:
                timing_score += 0.2
                reasons.append("strong_volume")
            elif volume_ratio < 0.5:
                timing_score -= 0.1
                reasons.append("weak_volume")
            
            is_valid = timing_score >= -0.3
            
            return {
                'valid': is_valid,
                'score': timing_score,
                'reasons': reasons,
                'price_position': price_position,
                'rsi': rsi,
                'stoch_k': stoch_k,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            return {'valid': True, 'reason': f'error: {str(e)}', 'score': 0}
    
    def _calculate_long_entry(self, current_price: float, mtf_context: MultiTimeframeContext,
                            volume_entry: Dict, df: pd.DataFrame = None) -> float:
        """FIXED: Calculate LONG entry WITHOUT chasing momentum"""
        try:
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                recent = df.tail(self.signal_config.price_momentum_lookback)
                price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                volatility_adjustment = (atr / current_price) * 0.5
                
                # FIXED: Use pullback entries
                if self.signal_config.use_pullback_entries:
                    if price_change > 0.01:  # Strong upward momentum
                        # Wait for pullback - set entry BELOW current price
                        base_entry = current_price * (1 - self.signal_config.entry_buffer_from_structure)
                    else:
                        base_entry = current_price * 0.999
                else:
                    # Original logic (not recommended for crypto)
                    if price_change > 0.01:
                        entry_adjustment = max(volatility_adjustment, self.signal_config.momentum_entry_adjustment)
                        base_entry = current_price * (1 + entry_adjustment)
                    elif price_change < -0.005:
                        base_entry = current_price * (1 - self.signal_config.entry_buffer_from_structure)
                    else:
                        base_entry = current_price * 0.999
            else:
                base_entry = current_price * 0.999
            
            entry_candidates = [base_entry]
            
            for zone in mtf_context.higher_tf_zones:
                if zone['type'] == 'support' and zone['price'] < current_price:
                    buffered_entry = zone['price'] * (1 + self.signal_config.entry_buffer_from_structure)
                    if buffered_entry < current_price * 1.02:
                        entry_candidates.append(buffered_entry)
            
            if volume_entry.get('confidence', 0) > 0.5:
                vol_entry = volume_entry.get('entry_price', current_price)
                if current_price * 0.99 <= vol_entry <= current_price * 1.02:
                    entry_candidates.append(vol_entry)
            
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                if price_change > 0.01 and not self.signal_config.use_pullback_entries:
                    return max(entry_candidates)
                else:
                    entry_candidates.sort()
                    return entry_candidates[len(entry_candidates)//2]
            else:
                return min(entry_candidates)
                
        except Exception:
            return current_price * 0.999
    
    def _calculate_short_entry(self, current_price: float, mtf_context: MultiTimeframeContext,
                             volume_entry: Dict, df: pd.DataFrame = None) -> float:
        """FIXED: Calculate SHORT entry WITHOUT chasing momentum"""
        try:
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                recent = df.tail(self.signal_config.price_momentum_lookback)
                price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                volatility_adjustment = (atr / current_price) * 0.5
                
                # FIXED: Use pullback entries
                if self.signal_config.use_pullback_entries:
                    if price_change < -0.01:  # Strong downward momentum
                        # Wait for pullback - set entry ABOVE current price
                        base_entry = current_price * (1 + self.signal_config.entry_buffer_from_structure)
                    else:
                        base_entry = current_price * 1.001
                else:
                    # Original logic (not recommended for crypto)
                    if price_change < -0.01:
                        entry_adjustment = max(volatility_adjustment, self.signal_config.momentum_entry_adjustment)
                        base_entry = current_price * (1 - entry_adjustment)
                    elif price_change > 0.005:
                        base_entry = current_price * (1 + self.signal_config.entry_buffer_from_structure)
                    else:
                        base_entry = current_price * 1.001
            else:
                base_entry = current_price * 1.001
            
            entry_candidates = [base_entry]
            
            for zone in mtf_context.higher_tf_zones:
                if zone['type'] == 'resistance' and zone['price'] > current_price:
                    buffered_entry = zone['price'] * (1 - self.signal_config.entry_buffer_from_structure)
                    if buffered_entry > current_price * 0.98:
                        entry_candidates.append(buffered_entry)
            
            if volume_entry.get('confidence', 0) > 0.5:
                vol_entry = volume_entry.get('entry_price', current_price)
                if current_price * 0.98 <= vol_entry <= current_price * 1.01:
                    entry_candidates.append(vol_entry)
            
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                if price_change < -0.01 and not self.signal_config.use_pullback_entries:
                    return min(entry_candidates)
                else:
                    entry_candidates.sort()
                    return entry_candidates[len(entry_candidates)//2]
            else:
                return max(entry_candidates)
                
        except Exception:
            return current_price * 1.001
    
    def _calculate_long_stop(self, entry_price: float, mtf_context: MultiTimeframeContext, 
                       df: pd.DataFrame, leverage: int = 1) -> float:
        """FIXED: Calculate LONG stop loss with PROPER crypto distances"""
        try:
            stop_candidates = []
            
            if len(df) >= 14 and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_pct = (atr / entry_price) * 100
                
                recent_changes = df['close'].pct_change().tail(20).abs()
                max_volatility = recent_changes.max()
                
                # FIXED: Proper multipliers for crypto
                if max_volatility > self.signal_config.extreme_volatility_threshold:
                    atr_multiplier = 4.0  # Very wide for extreme volatility
                elif max_volatility > self.signal_config.high_volatility_threshold:
                    atr_multiplier = 3.5
                elif max_volatility > 0.04:
                    atr_multiplier = 3.0
                elif max_volatility > 0.02:
                    atr_multiplier = 2.5
                else:
                    atr_multiplier = 2.0
                
                # Calculate ATR-based stop
                atr_stop_distance = atr * atr_multiplier
                atr_stop_pct = (atr_stop_distance / entry_price) * 100
                
                # Apply leverage adjustment
                adjusted_stop_pct = self.signal_config.get_leverage_adjusted_stop_distance(atr_stop_pct, leverage)
                
                # Ensure within bounds but don't artificially cap in high volatility
                if max_volatility > self.signal_config.high_volatility_threshold:
                    # In high volatility, let ATR determine stop without max cap
                    final_stop_pct = max(self.signal_config.min_stop_distance_pct, adjusted_stop_pct)
                else:
                    # Normal conditions - apply both min and max
                    final_stop_pct = min(self.signal_config.max_stop_distance_pct,
                                        max(self.signal_config.min_stop_distance_pct, adjusted_stop_pct))
                
                atr_stop = entry_price * (1 - final_stop_pct / 100)
                stop_candidates.append(atr_stop)
            
            # Support-based stop
            support_zones = [z for z in mtf_context.higher_tf_zones 
                        if z['type'] == 'support' and z['price'] < entry_price]
            if support_zones:
                closest_support = max(support_zones, key=lambda x: x['price'])
                structure_stop = closest_support['price'] * (1 - self.signal_config.structure_stop_buffer)
                distance_pct = ((entry_price - structure_stop) / entry_price) * 100
                
                # Only use if distance is reasonable
                if self.signal_config.min_stop_distance_pct <= distance_pct <= self.signal_config.max_stop_distance_pct * 1.5:
                    stop_candidates.append(structure_stop)
            
            # Minimum stop based on leverage
            base_min_stop_pct = self.signal_config.get_leverage_adjusted_stop_distance(
                self.signal_config.min_stop_distance_pct, leverage
            )
            min_stop = entry_price * (1 - base_min_stop_pct / 100)
            stop_candidates.append(min_stop)
            
            if stop_candidates:
                # Choose the most conservative (highest) stop
                chosen_stop = max(stop_candidates)
                return chosen_stop
            else:
                return entry_price * (1 - self.signal_config.min_stop_distance_pct / 100)
                
        except Exception:
            return entry_price * (1 - self.signal_config.min_stop_distance_pct / 100)
    
    def _calculate_short_stop(self, entry_price: float, mtf_context: MultiTimeframeContext,
                        df: pd.DataFrame, leverage: int = 1) -> float:
        """FIXED: Calculate SHORT stop loss with PROPER crypto distances"""
        try:
            stop_candidates = []
            
            if len(df) >= 14 and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_pct = (atr / entry_price) * 100
                
                recent_changes = df['close'].pct_change().tail(20).abs()
                max_volatility = recent_changes.max()
                
                # FIXED: Proper multipliers for crypto
                if max_volatility > self.signal_config.extreme_volatility_threshold:
                    atr_multiplier = 4.0
                elif max_volatility > self.signal_config.high_volatility_threshold:
                    atr_multiplier = 3.5
                elif max_volatility > 0.04:
                    atr_multiplier = 3.0
                elif max_volatility > 0.02:
                    atr_multiplier = 2.5
                else:
                    atr_multiplier = 2.0
                
                # Calculate ATR-based stop
                atr_stop_distance = atr * atr_multiplier
                atr_stop_pct = (atr_stop_distance / entry_price) * 100
                
                # Apply leverage adjustment
                adjusted_stop_pct = self.signal_config.get_leverage_adjusted_stop_distance(atr_stop_pct, leverage)
                
                # Ensure within bounds but don't artificially cap in high volatility
                if max_volatility > self.signal_config.high_volatility_threshold:
                    final_stop_pct = max(self.signal_config.min_stop_distance_pct, adjusted_stop_pct)
                else:
                    final_stop_pct = min(self.signal_config.max_stop_distance_pct,
                                        max(self.signal_config.min_stop_distance_pct, adjusted_stop_pct))
                
                atr_stop = entry_price * (1 + final_stop_pct / 100)
                stop_candidates.append(atr_stop)
            
            # Resistance-based stop
            resistance_zones = [z for z in mtf_context.higher_tf_zones 
                            if z['type'] == 'resistance' and z['price'] > entry_price]
            if resistance_zones:
                closest_resistance = min(resistance_zones, key=lambda x: x['price'])
                structure_stop = closest_resistance['price'] * (1 + self.signal_config.structure_stop_buffer)
                distance_pct = ((structure_stop - entry_price) / entry_price) * 100
                
                # Only use if distance is reasonable
                if self.signal_config.min_stop_distance_pct <= distance_pct <= self.signal_config.max_stop_distance_pct * 1.5:
                    stop_candidates.append(structure_stop)
            
            # Minimum stop based on leverage
            base_min_stop_pct = self.signal_config.get_leverage_adjusted_stop_distance(
                self.signal_config.min_stop_distance_pct, leverage
            )
            min_stop = entry_price * (1 + base_min_stop_pct / 100)
            stop_candidates.append(min_stop)
            
            if stop_candidates:
                # Choose the most conservative (lowest) stop
                chosen_stop = min(stop_candidates)
                return chosen_stop
            else:
                return entry_price * (1 + self.signal_config.min_stop_distance_pct / 100)            
                
        except Exception:
            return entry_price * (1 + self.signal_config.min_stop_distance_pct / 100)
    
    def _calculate_long_targets(self, entry_price: float, stop_loss: float,
                               mtf_context: MultiTimeframeContext, df: pd.DataFrame,
                               intel: MarketIntelligence) -> Tuple[float, float]:
        """Calculate LONG take profit targets - market-based"""
        try:
            risk = entry_price - stop_loss
            
            # TP1: Market-based
            tp1_candidates = []
            
            resistance_zones = [z for z in mtf_context.higher_tf_zones 
                              if z['type'] == 'resistance' and z['price'] > entry_price]
            
            if resistance_zones:
                nearest_resistance = min(resistance_zones, key=lambda x: x['price'])
                tp1_from_resistance = nearest_resistance['price'] * 0.995
                tp1_candidates.append(tp1_from_resistance)
            
            if len(df) >= 20:
                recent_high = df['high'].tail(20).max()
                if recent_high > entry_price:
                    tp1_from_swing = recent_high * 0.995
                    tp1_candidates.append(tp1_from_swing)
            
            if 'bb_upper' in df.columns:
                bb_upper = df['bb_upper'].iloc[-1]
                if bb_upper > entry_price:
                    tp1_candidates.append(bb_upper * 0.995)
            
            if entry_price < 1:
                round_increment = 0.01
            elif entry_price < 10:
                round_increment = 0.1
            elif entry_price < 100:
                round_increment = 1.0
            else:
                round_increment = 10.0
            
            next_round = ((entry_price // round_increment) + 1) * round_increment
            if next_round > entry_price * 1.005:
                tp1_candidates.append(next_round)
            
            min_tp1_distance = entry_price * (1 + self.signal_config.min_tp_distance_pct)
            valid_tp1_candidates = [tp for tp in tp1_candidates if tp >= min_tp1_distance]
            
            if valid_tp1_candidates:
                tp1 = min(valid_tp1_candidates)
            else:
                tp1 = entry_price * (1 + max(self.signal_config.min_tp_distance_pct * 2, 0.025))
            
            max_tp1_distance = entry_price * (1 + 0.08)
            tp1 = min(tp1, max_tp1_distance)
            
            # TP2: Market-based
            tp2_candidates = []
            
            deeper_resistances = [z for z in mtf_context.higher_tf_zones 
                                 if z['type'] == 'resistance' and z['price'] > tp1 * 1.02]
            
            if deeper_resistances:
                for resistance in sorted(deeper_resistances, key=lambda x: x['price'])[:2]:
                    tp2_from_resistance = resistance['price'] * 0.995
                    if tp2_from_resistance > tp1 * 1.03:
                        tp2_candidates.append(tp2_from_resistance)
            
            if len(df) >= 50:
                major_high = df['high'].tail(50).max()
                if major_high > tp1 * 1.05:
                    tp2_from_major_swing = major_high * 0.995
                    tp2_candidates.append(tp2_from_major_swing)
            
            if len(df) >= 30:
                recent_low = df['low'].tail(30).min()
                recent_high = df['high'].tail(30).max()
                fib_range = recent_high - recent_low
                
                fib_1618 = entry_price + (fib_range * 0.618)
                if fib_1618 > tp1 * 1.04:
                    tp2_candidates.append(fib_1618)
                
                fib_2618 = entry_price + (fib_range * 1.0)
                if fib_2618 > tp1 * 1.05 and fib_2618 < entry_price * 1.15:
                    tp2_candidates.append(fib_2618)
            
            if entry_price < 1:
                major_round_increment = 0.05
            elif entry_price < 10:
                major_round_increment = 0.5
            elif entry_price < 100:
                major_round_increment = 5.0
            else:
                major_round_increment = 50.0
            
            major_round = ((entry_price // major_round_increment) + 2) * major_round_increment
            if major_round > tp1 * 1.05:
                tp2_candidates.append(major_round)
            
            if intel.overall_api_sentiment > 65:
                sentiment_tp2 = tp1 * 1.08
                tp2_candidates.append(sentiment_tp2)
            
            if tp2_candidates:
                reasonable_tp2 = [tp for tp in tp2_candidates 
                                 if tp > tp1 * 1.04 and tp < entry_price * (1 + self.signal_config.max_tp_distance_pct)]
                
                if reasonable_tp2:
                    reasonable_tp2.sort()
                    tp2 = reasonable_tp2[len(reasonable_tp2) // 2]
                else:
                    tp2 = tp1 * 1.06
            else:
                tp2 = tp1 * 1.06
            
            tp2 = max(tp2, tp1 * 1.04)
            max_tp = entry_price * (1 + self.signal_config.max_tp_distance_pct)
            tp2 = min(tp2, max_tp)
            
            return tp1, tp2
            
        except Exception:
            risk = entry_price - stop_loss
            tp1 = entry_price * 1.03
            tp2 = entry_price * 1.06
            return tp1, tp2
    
    def _calculate_short_targets(self, entry_price: float, stop_loss: float,
                                mtf_context: MultiTimeframeContext, df: pd.DataFrame,
                                intel: MarketIntelligence) -> Tuple[float, float]:
        """Calculate SHORT take profit targets - market-based"""
        try:
            risk = stop_loss - entry_price
            
            # TP1: Market-based
            tp1_candidates = []
            
            support_zones = [z for z in mtf_context.higher_tf_zones 
                           if z['type'] == 'support' and z['price'] < entry_price]
            
            if support_zones:
                nearest_support = max(support_zones, key=lambda x: x['price'])
                tp1_from_support = nearest_support['price'] * 1.005
                tp1_candidates.append(tp1_from_support)
            
            if len(df) >= 20:
                recent_low = df['low'].tail(20).min()
                if recent_low < entry_price:
                    tp1_from_swing = recent_low * 1.005
                    tp1_candidates.append(tp1_from_swing)
            
            if 'bb_lower' in df.columns:
                bb_lower = df['bb_lower'].iloc[-1]
                if bb_lower < entry_price:
                    tp1_candidates.append(bb_lower * 1.005)
            
            if entry_price < 1:
                round_increment = 0.01
            elif entry_price < 10:
                round_increment = 0.1
            elif entry_price < 100:
                round_increment = 1.0
            else:
                round_increment = 10.0
            
            next_round = (entry_price // round_increment) * round_increment
            if next_round < entry_price * 0.995:
                tp1_candidates.append(next_round)
            
            min_tp1_distance = entry_price * (1 - self.signal_config.min_tp_distance_pct)
            valid_tp1_candidates = [tp for tp in tp1_candidates if tp <= min_tp1_distance]
            
            if valid_tp1_candidates:
                tp1 = max(valid_tp1_candidates)
            else:
                tp1 = entry_price * (1 - max(self.signal_config.min_tp_distance_pct * 2, 0.025))
            
            max_tp1_distance = entry_price * (1 - 0.08)
            tp1 = max(tp1, max_tp1_distance)
            
            # TP2: Market-based
            tp2_candidates = []
            
            deeper_supports = [z for z in mtf_context.higher_tf_zones 
                             if z['type'] == 'support' and z['price'] < tp1 * 0.98]
            
            if deeper_supports:
                for support in sorted(deeper_supports, key=lambda x: x['price'], reverse=True)[:2]:
                    tp2_from_support = support['price'] * 1.005
                    if tp2_from_support < tp1 * 0.97:
                        tp2_candidates.append(tp2_from_support)
            
            if len(df) >= 50:
                major_low = df['low'].tail(50).min()
                if major_low < tp1 * 0.95:
                    tp2_from_major_swing = major_low * 1.005
                    tp2_candidates.append(tp2_from_major_swing)
            
            if len(df) >= 30:
                recent_low = df['low'].tail(30).min()
                recent_high = df['high'].tail(30).max()
                fib_range = recent_high - recent_low
                
                fib_1618 = entry_price - (fib_range * 0.618)
                if fib_1618 < tp1 * 0.96:
                    tp2_candidates.append(fib_1618)
                
                fib_2618 = entry_price - (fib_range * 1.0)
                if fib_2618 < tp1 * 0.95 and fib_2618 > entry_price * 0.85:
                    tp2_candidates.append(fib_2618)
            
            if entry_price < 1:
                major_round_increment = 0.05
            elif entry_price < 10:
                major_round_increment = 0.5
            elif entry_price < 100:
                major_round_increment = 5.0
            else:
                major_round_increment = 50.0
            
            major_round = ((entry_price // major_round_increment) - 2) * major_round_increment
            if major_round < tp1 * 0.95:
                tp2_candidates.append(major_round)
            
            if intel.overall_api_sentiment < 35:
                sentiment_tp2 = tp1 * 0.92
                tp2_candidates.append(sentiment_tp2)
            
            if tp2_candidates:
                reasonable_tp2 = [tp for tp in tp2_candidates 
                                 if tp < tp1 * 0.96 and tp > entry_price * (1 - self.signal_config.max_tp_distance_pct)]
                
                if reasonable_tp2:
                    reasonable_tp2.sort(reverse=True)
                    tp2 = reasonable_tp2[len(reasonable_tp2) // 2]
                else:
                    tp2 = tp1 * 0.94
            else:
                tp2 = tp1 * 0.94
            
            tp2 = min(tp2, tp1 * 0.96)
            max_tp = entry_price * (1 - self.signal_config.max_tp_distance_pct)
            tp2 = max(tp2, max_tp)
            
            return tp1, tp2
            
        except Exception:
            risk = stop_loss - entry_price
            tp1 = entry_price * 0.97
            tp2 = entry_price * 0.94
            return tp1, tp2
    
    def _determine_market_regime(self, symbol_data: Dict, df: pd.DataFrame) -> str:
        """Determine current market regime for the symbol"""
        try:
            price_change_24h = symbol_data.get('price_change_24h', 0)
            volume_24h = symbol_data.get('volume_24h', 0)
            
            if len(df) >= 20:
                recent_changes = df['close'].pct_change().tail(20) * 100
                volatility = recent_changes.std()
                
                if volatility > 8:
                    return 'volatile'
            
            if price_change_24h > 8:
                return 'trending_up'
            elif price_change_24h > 3:
                if len(df) >= 10:
                    recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
                    if recent_trend > 2:
                        return 'trending_up'
                return 'ranging'
            elif price_change_24h < -8:
                return 'trending_down'
            elif price_change_24h < -3:
                if len(df) >= 10:
                    recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
                    if recent_trend < -2:
                        return 'trending_down'
                return 'ranging'
            else:
                if len(df) >= 20:
                    price_range = (df['high'].tail(20).max() - df['low'].tail(20).min()) / df['close'].iloc[-1]
                    if price_range < 0.15:
                        return 'ranging'
                
                return 'ranging'
            
        except Exception as e:
            self.logger.error(f"Market regime detection error: {e}")
            return 'ranging'
    
    def _get_multitimeframe_context(self, symbol_data: Dict, market_regime: str) -> Optional[MultiTimeframeContext]:
        """Get multi-timeframe context analysis"""
        try:
            if not self.exchange_manager:
                return self._create_fallback_context(symbol_data, market_regime)
            
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            structure_analysis = self._analyze_structure_timeframe(symbol, current_price)
            if not structure_analysis:
                return self._create_fallback_context(symbol_data, market_regime)
                
            confirmation_analysis = self._analyze_confirmation_timeframes(symbol, current_price)
            
            entry_bias = self._determine_entry_bias_with_regime(
                structure_analysis, confirmation_analysis, current_price, market_regime
            )
            
            confirmation_score = self._calculate_confirmation_score(
                structure_analysis, confirmation_analysis
            )
            
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
            structure_df = self.exchange_manager.fetch_ohlcv_data(symbol, self.structure_timeframe)
            if structure_df.empty or len(structure_df) < 30:
                return None
                
            structure_df = self._calculate_comprehensive_indicators(structure_df)
            latest = structure_df.iloc[-1]
            
            trend_data = self._analyze_trend_from_df(structure_df, current_price)
            
            key_zones = self._identify_key_zones(structure_df, current_price)
            recent_high = structure_df['high'].tail(20).max()
            recent_low = structure_df['low'].tail(20).min()
            
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
                    continue
                    
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
            traditional_bias = self._determine_entry_bias(structure_analysis, confirmation_analysis, current_price)
            
            struct_trend = structure_analysis['trend']
            
            if market_regime == 'trending_up':
                if traditional_bias == 'long_favored':
                    return 'long_favored'
                elif traditional_bias == 'short_favored':
                    if struct_trend == 'bearish' and structure_analysis['strength'] > 0.7:
                        return 'short_favored'
                    else:
                        return 'neutral'
                else:
                    return traditional_bias
                    
            elif market_regime == 'trending_down':
                if traditional_bias == 'short_favored':
                    return 'short_favored'
                elif traditional_bias == 'long_favored':
                    if struct_trend in ['bullish', 'strong_bullish'] and structure_analysis['strength'] > 0.7:
                        return 'long_favored'
                    else:
                        return 'neutral'
                else:
                    return traditional_bias
                    
            elif market_regime == 'volatile':
                if traditional_bias in ['long_favored', 'short_favored']:
                    confirmation_count = len(confirmation_analysis)
                    strong_confirmations = sum(1 for tf_data in confirmation_analysis.values() 
                                             if tf_data.get('strength', 0) > 0.6)
                    
                    if strong_confirmations >= max(1, confirmation_count // 2):
                        return traditional_bias
                    else:
                        return 'neutral'
                else:
                    return traditional_bias
                    
            elif market_regime == 'ranging':
                if traditional_bias == 'avoid':
                    return 'neutral'
                else:
                    return traditional_bias
                    
            else:
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
            
            near_major_resistance = any(
                zone['type'] == 'resistance' and zone['distance_pct'] < 0.02 
                for zone in key_zones if zone['price'] > current_price
            )
            near_major_support = any(
                zone['type'] == 'support' and zone['distance_pct'] < 0.02 
                for zone in key_zones if zone['price'] < current_price
            )
            
            confirmation_bullish = 0
            confirmation_bearish = 0
            total_confirmations = len(confirmation_analysis)
            
            for tf_data in confirmation_analysis.values():
                if tf_data['trend'] in ['bullish', 'strong_bullish']:
                    confirmation_bullish += 1
                elif tf_data['trend'] == 'bearish':
                    confirmation_bearish += 1
            
            if struct_trend == 'strong_bullish' and struct_strength > 0.75:
                if near_major_resistance:
                    return 'neutral'
                elif structure_analysis['momentum_bullish']:
                    return 'long_favored'
                else:
                    return 'neutral'
            
            elif struct_trend == 'bullish':
                if near_major_support and confirmation_bullish > confirmation_bearish:
                    return 'long_favored'
                elif near_major_resistance:
                    return 'short_favored'
                else:
                    return 'neutral'
            
            elif struct_trend == 'bearish':
                if near_major_resistance:
                    return 'short_favored'
                elif near_major_support:
                    return 'neutral'
                else:
                    return 'short_favored' if not structure_analysis['momentum_bullish'] else 'neutral'
            
            elif struct_trend == 'neutral':
                if near_major_support and confirmation_bullish > confirmation_bearish:
                    return 'long_favored'
                elif near_major_resistance and confirmation_bearish > confirmation_bullish:
                    return 'short_favored'
                else:
                    return 'neutral'
            
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
            
            sma_20 = latest.get('sma_20', current_price)
            sma_50 = latest.get('sma_50', current_price)
            ema_12 = latest.get('ema_12', current_price)
            ema_26 = latest.get('ema_26', current_price)
            
            price_above_sma20 = current_price > sma_20
            price_above_sma50 = current_price > sma_50
            sma20_above_sma50 = sma_20 > sma_50
            ema_bullish = ema_12 > ema_26
            
            bullish_signals = sum([price_above_sma20, price_above_sma50, sma20_above_sma50, ema_bullish])
            
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
            
            for i in range(2, len(recent_candles) - 2):
                high = recent_candles.iloc[i]['high']
                low = recent_candles.iloc[i]['low']
                
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
            
            zones.sort(key=lambda x: x['distance_pct'])
            return zones[:8]
            
        except Exception:
            return []
    
    def _calculate_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            rsi_min = df['rsi'].rolling(window=14).min()
            rsi_max = df['rsi'].rolling(window=14).max()
            stoch_rsi = (df['rsi'] - rsi_min) / (rsi_max - rsi_min) * 100
            df['stoch_rsi_k'] = stoch_rsi.rolling(window=3).mean()
            df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            df['volume_avg'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_avg']
            
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
            
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_strength = 1 if macd > macd_signal else 0
            
            rsi = latest.get('rsi', 50)
            rsi_momentum = 0.5
            if 30 < rsi < 70:
                rsi_momentum = 1
            elif rsi > 70:
                rsi_momentum = 0.3
            elif rsi < 30:
                rsi_momentum = 0.3
            
            price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
            price_momentum = min(1.0, abs(price_change_5) * 10)
            
            volume_ratio = latest.get('volume_ratio', 1)
            volume_momentum = min(1.0, volume_ratio / 2)
            
            total_momentum = (macd_strength * 0.3 + rsi_momentum * 0.3 + 
                            price_momentum * 0.25 + volume_momentum * 0.15)
            
            return max(0.1, min(1.0, total_momentum))
            
        except Exception as e:
            self.logger.error(f"Momentum strength analysis error: {e}")
            return 0.5
    
    def _determine_signal_strength(self, signal: Dict, mtf_context: MultiTimeframeContext) -> str:
        """Determine overall signal strength"""
        confidence = signal.get('confidence', 0)
        rr_ratio = signal.get('risk_reward_ratio', 0)
        mtf_score = mtf_context.confirmation_score
        quality_score = signal.get('quality_score', 0)
        
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
        
        if abs(entry - current) / current > 0.003:
            if entry > current and signal['side'] == 'buy':
                return 'momentum_chase'
            elif entry < current and signal['side'] == 'sell':
                return 'momentum_chase'
        
        for zone in mtf_context.higher_tf_zones:
            if abs(zone['price'] - entry) / entry < 0.005:
                if zone['type'] == 'support':
                    return 'support_bounce'
                else:
                    return 'resistance_rejection'
        
        if abs(entry - mtf_context.key_support) / entry < 0.005:
            return 'key_support_bounce'
        elif abs(entry - mtf_context.key_resistance) / entry < 0.005:
            return 'key_resistance_rejection'
        
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
        """Calculate signal quality grade"""
        confidence = signal.get('confidence', 0)
        rr_ratio = signal.get('risk_reward_ratio', 0)
        volume_24h = signal.get('volume_24h', 0)
        quality_score = signal.get('quality_score', 0)
        is_premium = signal.get('is_premium_signal', False)
        
        score = 0
        
        if is_premium:
            score += 20
        
        score += quality_score * 10
        
        if confidence >= 80:
            score += 35
        elif confidence >= 70:
            score += 25
        elif confidence >= 60:
            score += 15
        elif confidence >= 50:
            score += 10
        elif confidence >= 45:
            score += 5
        
        if rr_ratio >= 3.5:
            score += 25
        elif rr_ratio >= 2.5:
            score += 20
        elif rr_ratio >= 2:
            score += 15
        elif rr_ratio >= 1.5:
            score += 10
        elif rr_ratio >= 1.3:
            score += 5
        
        if volume_24h >= 10_000_000:
            score += 20
        elif volume_24h >= 5_000_000:
            score += 15
        elif volume_24h >= 1_000_000:
            score += 10
        elif volume_24h >= 500_000:
            score += 5
        
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
    
    def _calculate_market_volatility(self, df: pd.DataFrame) -> float:
        """Calculate market volatility"""
        if len(df) < 20:
            return 0.05
        
        returns = df['close'].pct_change().tail(20)
        return returns.std()
    
    def _calculate_trend_strength_simple(self, df: pd.DataFrame) -> float:
        """Simple trend strength calculation"""
        if len(df) < 20:
            return 0
        
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        return abs(price_change)
    
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
    
    def rank_opportunities_with_mtf(self, signals: List[Dict], dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """Enhanced ranking system with quality scoring"""
        try:
            opportunities = []
            
            for signal in signals:
                mtf_validated = signal.get('mtf_validated', False)
                confidence = signal.get('confidence', 0)
                rr_ratio = signal.get('risk_reward_ratio', 0)
                volume_24h = signal.get('volume_24h', 0)
                quality_grade = signal.get('quality_grade', 'C')
                quality_score = signal.get('quality_score', 0)
                is_premium = signal.get('is_premium_signal', False)
                
                api_sentiment = signal.get('market_intelligence', {}).get('overall_api_sentiment', 50)
                api_bonus = 0
                if api_sentiment > 70 or api_sentiment < 30:
                    api_bonus = 500
                
                if is_premium:
                    base_priority = 10000
                elif mtf_validated:
                    base_priority = 7000
                else:
                    base_priority = 4000
                
                quality_bonus = {
                    'A+': 2000, 'A': 1600, 'A-': 1200,
                    'B+': 800, 'B': 400, 'B-': 200,
                    'C+': 100, 'C': 0
                }.get(quality_grade, 0)
                
                quality_score_bonus = int(quality_score * 200)
                
                quality_factors = signal.get('quality_factors', [])
                if any('breakout' in factor or 'momentum' in factor for factor in quality_factors):
                    fast_signal_bonus = 500
                else:
                    fast_signal_bonus = 0
                
                priority = (base_priority + quality_bonus + quality_score_bonus + fast_signal_bonus + api_bonus +
                          int(confidence * 10) + 
                          int(min(rr_ratio * 100, 500)) +
                          int(min(volume_24h / 100000, 100)))
                
                signal['priority'] = priority
                signal['ranking_details'] = {
                    'mtf_validated': mtf_validated,
                    'is_premium': is_premium,
                    'quality_grade': quality_grade,
                    'quality_score': quality_score,
                    'fast_signal': fast_signal_bonus > 0,
                    'api_extreme': api_bonus > 0,
                    'final_priority': priority
                }
                
                opportunities.append(signal)
            
            opportunities.sort(key=lambda x: x['priority'], reverse=True)
            
            if opportunities:
                self.logger.info(f"🎯 Top signals after ranking:")
                for i, opp in enumerate(opportunities[:5]):
                    details = opp['ranking_details']
                    self.logger.info(f"   {i+1}. {opp['symbol']} - Priority: {details['final_priority']} "
                                   f"(Premium: {details['is_premium']}, Grade: {details['quality_grade']}, "
                                   f"Fast: {details['fast_signal']}, API: {details['api_extreme']})")
            
            return opportunities[:self.config.charts_per_batch]
            
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return signals
    
    # ===== COMPATIBILITY METHODS =====
    
    def create_technical_summary(self, df: pd.DataFrame, latest: pd.Series = None) -> Dict:
        """Create technical analysis summary"""
        try:
            if latest is None:
                latest = df.iloc[-1]
            
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
            
            atr = latest.get('atr', latest['close'] * 0.02)
            volatility_pct = (atr / latest['close']) * 100
            volume_ratio = latest.get('volume_ratio', 1)
            volume_trend = self.get_volume_trend(df)
            
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
            
            price_change_5 = (latest['close'] - recent_30.iloc[-5]['close']) / recent_30.iloc[-5]['close']
            price_change_15 = (latest['close'] - recent_30.iloc[-15]['close']) / recent_30.iloc[-15]['close']
            price_change_30 = (latest['close'] - recent_30.iloc[0]['close']) / recent_30.iloc[0]['close']
            
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            ema_12 = latest.get('ema_12', latest['close'])
            ema_26 = latest.get('ema_26', latest['close'])
            
            ma_alignment_score = 0
            if latest['close'] > sma_20 > sma_50:
                ma_alignment_score += 3
            elif latest['close'] > sma_20:
                ma_alignment_score += 1
            elif latest['close'] < sma_20 < sma_50:
                ma_alignment_score -= 3
            elif latest['close'] < sma_20:
                ma_alignment_score -= 1
            
            if ema_12 > ema_26:
                ma_alignment_score += 1
            else:
                ma_alignment_score -= 1
            
            bullish_candles = len(recent_30[recent_30['close'] > recent_30['open']])
            consistency = bullish_candles / len(recent_30)
            
            momentum_alignment = 0
            if price_change_5 > 0 and price_change_15 > 0 and price_change_30 > 0:
                momentum_alignment = 1
            elif price_change_5 < 0 and price_change_15 < 0 and price_change_30 < 0:
                momentum_alignment = -1
            else:
                momentum_alignment = 0
            
            base_strength = (abs(price_change_15) + abs(ma_alignment_score) / 6 + 
                           abs(consistency - 0.5) * 2) / 3
            
            if momentum_alignment != 0:
                base_strength *= 1.2
            
            strength = min(1.0, base_strength)
            
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
            
            if body_size < 0.003:
                patterns.append('doji')
            elif body_size > 0.02:
                patterns.append('strong_body')
            
            if full_range > 0 and lower_shadow / full_range > 0.6 and body_size < full_range * 0.3:
                patterns.append('hammer')
            elif full_range > 0 and upper_shadow / full_range > 0.6 and body_size < full_range * 0.3:
                patterns.append('shooting_star')
            
            recent_lows = recent_10['low'].min()
            recent_highs = recent_10['high'].max()
            
            if latest['low'] <= recent_lows * 1.002:
                patterns.append('support_test')
            if latest['high'] >= recent_highs * 0.998:
                patterns.append('resistance_test')
            
            closes = recent_10['close'].values
            momentum = (closes[-1] - closes[0]) / closes[0] if closes[0] != 0 else 0
            
            pattern_strength = 0
            if 'hammer' in patterns or 'shooting_star' in patterns:
                pattern_strength += 0.3
            if 'support_test' in patterns or 'resistance_test' in patterns:
                pattern_strength += 0.2
            if 'strong_body' in patterns:
                pattern_strength += 0.2
            
            momentum_strength = min(0.5, abs(momentum) * 10)
            total_strength = min(1.0, pattern_strength + momentum_strength)
            
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
            
            atr = latest.get('atr', current_price * 0.02)
            volatility = atr / current_price
            
            base_risk = volatility * 2.0
            total_risk = max(0.1, min(1.0, base_risk))
            
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
                'distance_risk': 0,
                'risk_level': risk_level,
                'mtf_validated': False,
                'market_regime': 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return {'total_risk_score': 0.5, 'risk_level': 'Medium'}
    
    def _assess_volatility_risk(self, df: pd.DataFrame) -> Dict:
        """Assess volatility risk for position sizing adjustments"""
        try:
            if len(df) < 20:
                return {'level': 'unknown', 'multiplier': 1.0}
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            atr_pct = (atr / df['close'].iloc[-1]) * 100
            
            recent_returns = df['close'].pct_change().tail(10) * 100
            return_volatility = recent_returns.std()
            
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

# ===== FACTORY FUNCTION =====

def create_mtf_signal_generator(config: EnhancedSystemConfig, exchange_manager):
    """Factory function to create the enhanced signal generator v10.1"""
    return SignalGenerator(config, exchange_manager)

# ===== EXPORTS =====

__all__ = [
    'SignalGenerator',
    'AdaptiveSignalConfig',
    'SignalQualityTier',
    'MarketCondition',
    'PerformanceMetrics',
    'CircuitBreaker',
    'EnhancedSignalValidator',
    'MarketIntelligence',
    'MultiTimeframeContext',
    'create_mtf_signal_generator',
    # Include all the quality filter functions
    'analyze_price_momentum_strength',
    'check_volume_momentum_divergence',
    'identify_fast_moving_setup',
    'filter_choppy_markets',
    'calculate_momentum_adjusted_entry',
    'calculate_dynamic_stop_loss',
    'detect_divergence',
    'check_near_support_resistance'
]

__version__ = "10.1.0"
__features__ = [
    "✅ FIXED: Stop losses adjusted from 0.05-0.10% to 1.0-5.0% for crypto",
    "✅ FIXED: Entry logic to avoid chasing momentum (pullback entries)",
    "✅ FIXED: Tightened signal requirements (4.5 score vs 3.0)",
    "✅ NEW: Leverage awareness for stop calculation",
    "✅ NEW: Extreme volatility handling (15%+ daily moves)",
    "✅ ENHANCED: Volatility-based stop multipliers without artificial caps",
    "✅ All v10.0 features retained:",
    "   - Adaptive parameter adjustment",
    "   - Dual-mode: Strict vs Relaxed",
    "   - Circuit breaker for drawdown protection",
    "   - Intelligent API fallbacks",
    "   - Memory optimization",
    "   - Performance monitoring"
]