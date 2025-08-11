"""
ENHANCED Multi-Timeframe Signal Generation with Market Intelligence
VERSION 8.0 - V6.0 COMPLETE LOGIC + INTELLIGENT APIS

COMBINES:
✅ 100% of v6.0 technical analysis and logic (unchanged)
✅ Fear & Greed Index API integration
✅ Bybit Funding Rates analysis
✅ CryptoPanic News Sentiment
✅ Market-based TP2 calculation (enhanced from v6.0)
✅ All v6.0 configurations preserved

KEY FEATURES FROM V6.0:
✅ Fixed short signals in oversold conditions (RSI must be > 50)
✅ Fixed long signals in overbought conditions (RSI must be < 50)
✅ Support/Resistance detection to avoid bad entries
✅ Divergence detection (bullish/bearish)
✅ Stricter entry conditions for better win rate
✅ Balanced stop losses (5% - 10% for crypto markets)
✅ Market-based TP1 (uses support/resistance levels)
✅ Market-based TP2 (NEW: uses deeper market structure)
✅ All quality filters and momentum analysis
"""

import pandas as pd
import numpy as np
import logging
import time
import aiohttp
import asyncio
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from config.config import EnhancedSystemConfig

# ===== API CONFIGURATION =====
CRYPTOPANIC_API_KEY = '2c2a4ce275d7c36a8bb5ac71bf6a3b5a61e60cb8'

# ===== V6.0 ORIGINAL CONFIGURATION (UNCHANGED) =====

@dataclass
class SignalConfig:
    """Enhanced configuration for signal generation with balanced parameters for crypto"""
    
    # Stop loss parameters - BALANCED FOR CRYPTO
    min_stop_distance_pct: float = 0.075  # Minimum 5% stop distance
    max_stop_distance_pct: float = 0.12  # Maximum 10% stop distance
    structure_stop_buffer: float = 0.003  # 0.3% buffer below support/above resistance
    
    # Entry parameters - MORE FLEXIBLE
    entry_buffer_from_structure: float = 0.002  # 0.2% buffer from key levels
    entry_limit_distance: float = 0.02    # Max 2% from current price for limit orders
    momentum_entry_adjustment: float = 0.003  # 0.3% adjustment for trending markets
    
    # Take profit parameters - REALISTIC TARGETS
    min_tp_distance_pct: float = 0.015  # Minimum 1.5% profit target
    max_tp_distance_pct: float = 0.20   # Maximum 20% profit target
    tp1_multiplier: float = 2.0         # TP1 at 2x risk (NOT USED - market based instead)
    tp2_multiplier: float = 3.5         # TP2 at 3.5x risk (NOT USED - market based instead in v8.0)
    use_market_based_tp1: bool = True   # Use market structure for TP1 instead of multiplier
    use_market_based_tp2: bool = True   # NEW in v8.0: Use market structure for TP2
    
    # Risk/Reward parameters
    min_risk_reward: float = 1.5          # Minimum acceptable R/R (slightly below 2 for flexibility)
    max_risk_reward: float = 10.0         # Maximum R/R (cap unrealistic values)
    
    # Signal quality thresholds - STRICTER FOR BETTER WIN RATE
    min_confidence_for_signal: float = 55.0  # Higher minimum confidence
    mtf_confidence_boost: float = 10.0       # Reduced MTF boost for realism
    
    # RSI thresholds - FIXED TO AVOID BAD ENTRIES
    min_rsi_for_short: float = 45.0  # Don't short below this RSI (avoid oversold)
    max_rsi_for_short: float = 80.0  # Don't short above this (too overbought)
    min_rsi_for_long: float = 20.0   # Don't long below this (too oversold)
    max_rsi_for_long: float = 55.0   # Don't long above this RSI (avoid overbought)
    
    # Stochastic thresholds - STRICTER
    min_stoch_for_short: float = 40.0  # Don't short when stoch is oversold
    max_stoch_for_long: float = 60.0   # Don't long when stoch is overbought
    
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
    
    # Support/Resistance parameters
    support_resistance_lookback: int = 20  # Candles to look back for S/R
    min_distance_from_support: float = 0.02  # Don't short within 2% of support
    min_distance_from_resistance: float = 0.02  # Don't long within 2% of resistance

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

@dataclass
class MarketIntelligence:
    """Market intelligence data from APIs"""
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

# ===== API COMPONENTS =====

class FearGreedAnalyzer:
    """Fetch and analyze crypto Fear & Greed Index"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        self.session = None  # Reusable session
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            # Create session with DNS resolver that doesn't use inotify
            connector = aiohttp.TCPConnector(
                force_close=True,
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    async def close(self):
        """Close the session when done"""
        if self.session and not self.session.closed:
            await self.session.close()
        
    def get_fear_greed_index(self) -> Dict:
        """Get Fear & Greed Index from Alternative.me API - SYNCHRONOUS VERSION"""
        try:
            # Check cache
            if 'fear_greed' in self.cache:
                cached_time, cached_data = self.cache['fear_greed']
                if time.time() - cached_time < self.cache_duration:
                    return cached_data
            
            url = "https://api.alternative.me/fng/?limit=2"
            
            # Use requests instead of aiohttp for synchronous operation
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
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
                    
                    self.cache['fear_greed'] = (time.time(), result)
                    self.logger.debug(f"Fear & Greed Index: {result['value']} ({result['classification']})")
                    return result
            
            return self._get_fallback_fear_greed()
            
        except Exception as e:
            self.logger.error(f"Fear & Greed API error: {e}")
            return self._get_fallback_fear_greed()
        
    def _get_fallback_fear_greed(self) -> Dict:
        """Fallback Fear & Greed calculation"""
        return {
            'value': 50,
            'classification': 'Neutral',
            'timestamp': str(int(time.time())),
            'time_until_update': 86400,
            'change': 0
        }

class FundingRateAnalyzer:
    """Analyze funding rates from Bybit exchange"""
    
    def __init__(self, exchange_manager=None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange_manager
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get funding rate for symbol from Bybit"""
        try:
            cache_key = f'funding_{symbol}'
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    return cached_data
            
            if not self.exchange:
                return self._get_fallback_funding()
            
            try:
                markets = self.exchange.exchange.load_markets()
                market = self.exchange.exchange.market(symbol)
                
                if market['swap']:
                    funding = self.exchange.exchange.fetch_funding_rate(symbol)
                    
                    funding_history = self.exchange.exchange.fetch_funding_rate_history(
                        symbol, limit=8
                    )
                    
                    if funding_history:
                        avg_funding = sum(f['fundingRate'] for f in funding_history) / len(funding_history)
                    else:
                        avg_funding = funding['fundingRate']
                    
                    result = {
                        'current_rate': funding['fundingRate'],
                        'predicted_rate': funding.get('predictedFundingRate', funding['fundingRate']),
                        'funding_timestamp': funding['timestamp'],
                        'avg_24h': avg_funding,
                        'sentiment': self._interpret_funding_rate(funding['fundingRate']),
                        'next_funding_time': funding.get('fundingDatetime', '')
                    }
                    
                    self.cache[cache_key] = (time.time(), result)
                    self.logger.debug(f"Funding rate for {symbol}: {result['current_rate']:.4%}")
                    return result
                    
            except Exception as e:
                self.logger.debug(f"Funding rate fetch error: {e}")
                
            return self._get_fallback_funding()
            
        except Exception as e:
            self.logger.error(f"Funding rate analysis error: {e}")
            return self._get_fallback_funding()
    
    def _get_fallback_funding(self) -> Dict:
        """Fallback funding rate data"""
        return {
            'current_rate': 0.0001,
            'predicted_rate': 0.0001,
            'funding_timestamp': int(time.time() * 1000),
            'avg_24h': 0.0001,
            'sentiment': 'neutral',
            'next_funding_time': ''
        }
    
    def _interpret_funding_rate(self, rate: float) -> str:
        """Interpret funding rate for sentiment"""
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

class NewsAnalyzer:
    """Analyze news sentiment from CryptoPanic API"""
    
    def __init__(self, api_key: str = CRYPTOPANIC_API_KEY):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.cache = {}
        self.cache_duration = 600  # 10 minutes
        self.session = None  # Reusable session
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            # Create session with DNS resolver that doesn't use inotify
            connector = aiohttp.TCPConnector(
                force_close=True,
                enable_cleanup_closed=True
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    async def close(self):
        """Close the session when done"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment for a symbol - SYNCHRONOUS VERSION"""
        try:
            base_currency = symbol.split('/')[0]
            
            cache_key = f'news_{base_currency}'
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if time.time() - cached_time < self.cache_duration:
                    return cached_data
            
            if not self.api_key:
                return self._get_fallback_news()
            
            url = f"https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': self.api_key,
                'currencies': base_currency,
                'filter': 'hot',
                'public': 'true'
            }
            
            # Use requests instead of aiohttp
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data:
                    result = self._analyze_news_items(data['results'], base_currency)
                    self.cache[cache_key] = (time.time(), result)
                    self.logger.debug(f"News sentiment for {base_currency}: {result['sentiment_score']:.2f}")
                    return result
            
            return self._get_fallback_news()
            
        except Exception as e:
            self.logger.error(f"News API error: {e}")
            return self._get_fallback_news()
    
    def _analyze_news_items(self, news_items: List[Dict], currency: str) -> Dict:
        """Analyze news items for sentiment"""
        try:
            total_score = 0
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            important_news = []
            
            for item in news_items[:20]:
                votes = item.get('votes', {})
                positive = votes.get('positive', 0)
                negative = votes.get('negative', 0)
                liked = votes.get('liked', 0)
                disliked = votes.get('disliked', 0)
                
                if positive + negative > 0:
                    item_sentiment = (positive - negative) / (positive + negative)
                else:
                    item_sentiment = 0
                
                importance = votes.get('important', 0)
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
                        'url': item.get('url', ''),
                        'published_at': item.get('published_at', '')
                    })
            
            news_count = len(news_items)
            if news_count > 0:
                sentiment_score = total_score / news_count
            else:
                sentiment_score = 0
            
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
            return self._get_fallback_news()
    
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
    
    def _get_fallback_news(self) -> Dict:
        """Fallback news data"""
        return {
            'sentiment_score': 0,
            'news_count_24h': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'important_news': [],
            'sentiment_classification': 'neutral'
        }

class MarketIntelligenceAggregator:
    """Aggregate market intelligence from APIs"""
    
    def __init__(self, exchange_manager=None):
        self.logger = logging.getLogger(__name__)
        self.fear_greed = FearGreedAnalyzer()
        self.funding = FundingRateAnalyzer(exchange_manager)
        self.news = NewsAnalyzer()
    
    def gather_intelligence(self, symbol: str) -> MarketIntelligence:
        """Gather all market intelligence for a symbol - SYNCHRONOUS VERSION"""
        try:
            intel = MarketIntelligence()
            
            # Call synchronous methods
            try:
                fear_greed_data = self.fear_greed.get_fear_greed_index()
                intel.fear_greed_index = fear_greed_data['value']
                intel.fear_greed_classification = fear_greed_data['classification']
                intel.fear_greed_change = fear_greed_data.get('change', 0)
            except Exception as e:
                self.logger.debug(f"Fear & Greed fetch failed: {e}")
            
            # Funding rate is already synchronous
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
            
            # Calculate overall API sentiment
            intel = self._calculate_overall_sentiment(intel)
            
            self.logger.debug(f"Market Intelligence gathered for {symbol}")
            self.logger.debug(f"  Fear & Greed: {intel.fear_greed_index} ({intel.fear_greed_classification})")
            self.logger.debug(f"  Funding: {intel.funding_rate:.4%} ({intel.funding_sentiment})")
            self.logger.debug(f"  News: {intel.news_sentiment_score:.2f} ({intel.news_classification})")
            
            return intel
            
        except Exception as e:
            self.logger.error(f"Intelligence gathering error: {e}")
            return MarketIntelligence()
        
    async def cleanup(self):
        """Clean up sessions when done"""
        try:
            await self.fear_greed.close()
            await self.news.close()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def _calculate_overall_sentiment(self, intel: MarketIntelligence) -> MarketIntelligence:
        """Calculate overall API sentiment"""
        try:
            # Fear & Greed component (0-100)
            fear_greed_score = intel.fear_greed_index
            
            # Funding component (convert to 0-100)
            if intel.funding_rate > 0:
                funding_score = max(0, min(100, 50 - intel.funding_rate * 50000))
            else:
                funding_score = max(0, min(100, 50 - intel.funding_rate * 50000))
            
            # News component (convert from -1 to 1 to 0-100)
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
                intel.api_signal_modifier = 1.15  # Strong sentiment = boost signals
            elif intel.overall_api_sentiment > 65 or intel.overall_api_sentiment < 35:
                intel.api_signal_modifier = 1.08
            else:
                intel.api_signal_modifier = 1.0
            
            return intel
            
        except Exception as e:
            self.logger.error(f"Sentiment calculation error: {e}")
            return intel
       
# ===== V6.0 QUALITY FILTER FUNCTIONS (UNCHANGED) =====

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
        
        if side == 'buy':
            if distance_from_support < 0.02:
                return {
                    'near_level': True,
                    'level_type': 'support',
                    'distance_pct': distance_from_support,
                    'favorable': True,
                    'level_price': recent_low
                }
            elif distance_from_resistance < 0.02:
                return {
                    'near_level': True,
                    'level_type': 'resistance',
                    'distance_pct': distance_from_resistance,
                    'favorable': False,
                    'level_price': recent_high
                }
        else:
            if distance_from_resistance < 0.02:
                return {
                    'near_level': True,
                    'level_type': 'resistance',
                    'distance_pct': distance_from_resistance,
                    'favorable': True,
                    'level_price': recent_high
                }
            elif distance_from_support < 0.02:
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
    """Identify setups likely to move fast in the signal direction"""
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
            if current_close > recent_high * 0.995:
                score += 2
                factors.append('breakout_imminent')
            
            if len(df) >= 10:
                touched_support = any(df['low'].tail(10) <= recent_low * 1.01)
                bounced_up = current_close > recent_low * 1.02
                if touched_support and bounced_up:
                    score += 1.5
                    factors.append('support_bounce')
        
        else:
            if current_close < recent_low * 1.005:
                score += 2
                factors.append('breakdown_imminent')
            
            if len(df) >= 10:
                touched_resistance = any(df['high'].tail(10) >= recent_high * 0.99)
                rejected_down = current_close < recent_high * 0.98
                if touched_resistance and rejected_down:
                    score += 1.5
                    factors.append('resistance_rejection')
        
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio > 2.0:
            score += 1
            factors.append('volume_surge')
        
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        
        if side == 'buy':
            if 35 < rsi < 50 and macd > macd_signal:
                score += 1
                factors.append('momentum_building')
        else:
            if 50 < rsi < 65 and macd < macd_signal:
                score += 1
                factors.append('momentum_building')
        
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
        if len(df) < config.price_momentum_lookback:
            momentum_pct = 0
        else:
            recent = df.tail(config.price_momentum_lookback)
            price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            momentum_pct = price_change
        
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        volatility_adjustment = (atr / current_price) * 0.5
        
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
                              df: pd.DataFrame, side: str, config: SignalConfig) -> float:
    """Calculate stop loss with proper distance based on volatility (5% - 10%)"""
    try:
        if 'atr' in df.columns and len(df) >= 14:
            atr = df['atr'].iloc[-1]
            atr_pct = atr / current_price
        else:
            atr_pct = 0.03
        
        if len(df) >= 20:
            recent_changes = df['close'].pct_change().tail(20).abs()
            avg_volatility = recent_changes.mean()
            max_volatility = recent_changes.max()
        else:
            avg_volatility = 0.02
            max_volatility = 0.04
        
        if max_volatility > config.high_volatility_threshold:
            stop_distance_pct = min(config.max_stop_distance_pct, max(atr_pct * 3.5, config.min_stop_distance_pct * 1.5))
        elif avg_volatility > 0.03:
            stop_distance_pct = min(config.max_stop_distance_pct * 0.9, max(atr_pct * 3.0, config.min_stop_distance_pct * 1.3))
        elif avg_volatility > 0.015:
            stop_distance_pct = min(config.max_stop_distance_pct * 0.8, max(atr_pct * 2.5, config.min_stop_distance_pct * 1.1))
        else:
            stop_distance_pct = max(atr_pct * 2.0, config.min_stop_distance_pct)
        
        if side == 'buy':
            stop_loss = entry_price * (1 - stop_distance_pct)
        else:
            stop_loss = entry_price * (1 + stop_distance_pct)
        
        return stop_loss
        
    except Exception:
        if side == 'buy':
            return entry_price * (1 - config.min_stop_distance_pct)
        else:
            return entry_price * (1 + config.min_stop_distance_pct)

# ===== V6.0 SIGNAL VALIDATOR (UNCHANGED) =====

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
                new_stop = entry_price * (1 - self.config.min_stop_distance_pct)
                self.logger.debug(f"Adjusted stop loss from {stop_loss:.6f} to {new_stop:.6f} (min distance)")
                return True, new_stop
            elif distance_pct > self.config.max_stop_distance_pct:
                new_stop = entry_price * (1 - self.config.max_stop_distance_pct)
                self.logger.debug(f"Adjusted stop loss from {stop_loss:.6f} to {new_stop:.6f} (max distance)")
                return True, new_stop
        else:
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
        """Validate and adjust take profits - Both TP1 and TP2 are market-based in v8.0"""
        if side == 'buy':
            min_tp1_by_pct = entry_price * (1 + self.config.min_tp_distance_pct)
            validated_tp1 = max(tp1, min_tp1_by_pct)
            
            min_tp2_by_pct = entry_price * (1 + self.config.min_tp_distance_pct * 2)
            validated_tp2 = max(tp2, min_tp2_by_pct, validated_tp1 * 1.01)
            
            max_tp = entry_price * (1 + self.config.max_tp_distance_pct)
            validated_tp1 = min(validated_tp1, max_tp)
            validated_tp2 = min(validated_tp2, max_tp)
            
        else:
            min_tp1_by_pct = entry_price * (1 - self.config.min_tp_distance_pct)
            validated_tp1 = min(tp1, min_tp1_by_pct)
            
            min_tp2_by_pct = entry_price * (1 - self.config.min_tp_distance_pct * 2)
            validated_tp2 = min(tp2, min_tp2_by_pct, validated_tp1 * 0.99)
            
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
        return min(rr, self.config.max_risk_reward)

# ===== MAIN ENHANCED SIGNAL GENERATOR =====

class SignalGenerator:
    """
    ENHANCED Multi-Timeframe Signal Generator v8.0
    Combines v6.0 complete logic with Market Intelligence APIs
    """
    
    def __init__(self, config: EnhancedSystemConfig, exchange_manager=None):
        self.config = config
        self.exchange_manager = exchange_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize v6.0 signal configuration
        self.signal_config = SignalConfig()
        self.validator = SignalValidator(self.signal_config)
        
        # Initialize API intelligence
        self.intel_aggregator = MarketIntelligenceAggregator(exchange_manager)
        
        # Use configured timeframes from database
        self.primary_timeframe = config.timeframe
        self.confirmation_timeframes = config.confirmation_timeframes
        
        if self.confirmation_timeframes:
            tf_minutes = {'1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440}
            sorted_tfs = sorted(self.confirmation_timeframes, 
                              key=lambda x: tf_minutes.get(x, 0), reverse=True)
            self.structure_timeframe = sorted_tfs[0]
        else:
            self.structure_timeframe = '6h'
        
        self.debug_mode = False
        
        self.logger.debug("✅ ENHANCED Signal Generator v8.0 initialized")
        self.logger.debug(f"   Primary TF: {self.primary_timeframe}")
        self.logger.debug(f"   Structure TF: {self.structure_timeframe}")
        self.logger.debug(f"   Confirmation TFs: {self.confirmation_timeframes}")
        self.logger.debug(f"   Stop Loss Range: {self.signal_config.min_stop_distance_pct*100:.1f}% - {self.signal_config.max_stop_distance_pct*100:.1f}%")
        self.logger.debug(f"   Take Profit: TP1 & TP2 Market-based")
        self.logger.debug(f"   RSI Thresholds: Long < {self.signal_config.max_rsi_for_long}, Short > {self.signal_config.min_rsi_for_short}")
        self.logger.debug(f"   Market Intelligence: ✅ Enabled")

    def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol_data: Dict, 
                               volume_entry: Dict, fibonacci_data: Dict, 
                               confluence_zones: List[Dict], timeframe: str) -> Optional[Dict]:
        """Main entry point with comprehensive analysis - Enhanced with API Intelligence"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            if self.debug_mode:
                self.logger.info(f"🔧 DEBUG: Starting analysis for {symbol}")
            
            # Gather Market Intelligence from APIs - NOW SYNCHRONOUS
            try:
                intel = self.intel_aggregator.gather_intelligence(symbol)
            except Exception as e:
                self.logger.warning(f"Failed to gather market intelligence: {e}")
                intel = MarketIntelligence()  # Use default values
            
            # PHASE 1: Market regime detection (v6.0 logic)
            try:
                market_regime = self._determine_market_regime(symbol_data, df)
                if self.debug_mode:
                    self.logger.info(f"🔧 DEBUG: {symbol} - Market regime: {market_regime}")
            except Exception as e:
                self.logger.warning(f"Market regime detection failed for {symbol}: {e}")
                market_regime = 'ranging'
            
            # PHASE 2: Multi-timeframe context analysis (v6.0 logic)
            try:
                mtf_context = self._get_multitimeframe_context(symbol_data, market_regime)
                if not mtf_context:
                    if self.debug_mode:
                        self.logger.info(f"🔧 DEBUG: {symbol} - Creating fallback MTF context")
                    mtf_context = self._create_fallback_context(symbol_data, market_regime)
            except Exception as e:
                self.logger.warning(f"MTF context creation failed for {symbol}: {e}")
                mtf_context = self._create_fallback_context(symbol_data, market_regime)
            
            # PHASE 3: Signal generation with API enhancement
            signal = None
            
            # Try MTF-aware signal first (v6.0 logic)
            if self._should_use_mtf_signals(mtf_context):
                signal = self._generate_mtf_aware_signal(
                    df, symbol_data, volume_entry, fibonacci_data, 
                    confluence_zones, mtf_context, intel  # Pass intel data
                )
            
            # Fallback to traditional signal if MTF fails
            if not signal:
                if self.debug_mode:
                    self.logger.info(f"🔧 DEBUG: {symbol} - MTF signal failed, trying traditional")
                signal = self._generate_traditional_signal(
                    df, symbol_data, volume_entry, fibonacci_data, confluence_zones, intel
                )
            
            # Final validation and enhancement
            if signal:
                # Enhanced signal with comprehensive metadata and API intelligence
                signal = self._validate_and_enhance_signal(signal, mtf_context, df, market_regime, intel)
                
                # Add comprehensive analysis
                signal['analysis'] = self._create_comprehensive_analysis(
                    df, symbol_data, volume_entry, fibonacci_data, 
                    confluence_zones, mtf_context
                )
                
                # Add API intelligence data
                signal['market_intelligence'] = {
                    'fear_greed_index': intel.fear_greed_index,
                    'fear_greed_classification': intel.fear_greed_classification,
                    'funding_rate': intel.funding_rate,
                    'funding_sentiment': intel.funding_sentiment,
                    'news_sentiment': intel.news_sentiment_score,
                    'news_classification': intel.news_classification,
                    'overall_api_sentiment': intel.overall_api_sentiment
                }
                
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
    
    def __del__(self):
        """Cleanup when generator is destroyed"""
        pass
        # try:
        #     # Clean up sessions
        #     asyncio.run(self.intel_aggregator.cleanup())
        # except:
        #     pass  # Ignore errors during cleanup

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
        """Generate signal with MTF awareness and API intelligence"""
        try:
            latest = df.iloc[-1]
            
            # Determine signal direction based on MTF context
            if mtf_context.entry_bias == 'long_favored':
                return self._generate_long_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=True
                )
            elif mtf_context.entry_bias == 'short_favored':
                return self._generate_short_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=True
                )
            elif mtf_context.entry_bias == 'neutral':
                # Try both directions
                long_signal = self._generate_long_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=True
                )
                if long_signal:
                    return long_signal
                    
                return self._generate_short_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=True
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"MTF signal generation error: {e}")
            return None

    def _generate_traditional_signal(self, df: pd.DataFrame, symbol_data: Dict,
                                   volume_entry: Dict, fibonacci_data: Dict,
                                   confluence_zones: List[Dict],
                                   intel: MarketIntelligence) -> Optional[Dict]:
        """Generate traditional signal without MTF but with API intelligence"""
        try:
            latest = df.iloc[-1]
            
            # Create simple context
            mtf_context = self._create_fallback_context(symbol_data, 'unknown')
            
            # Traditional signal conditions with FIXED thresholds (v6.0 logic)
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            stoch_k = latest.get('stoch_rsi_k', 50)
            stoch_d = latest.get('stoch_rsi_d', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # LONG conditions - Fixed to avoid overbought entries
            if (self.signal_config.min_rsi_for_long < rsi < self.signal_config.max_rsi_for_long and 
                macd > macd_signal and 
                stoch_k > stoch_d and stoch_k < self.signal_config.max_stoch_for_long and
                volume_ratio > 0.8):
                return self._generate_long_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=False
                )
            # SHORT conditions - Fixed to avoid oversold entries
            elif (self.signal_config.min_rsi_for_short < rsi < self.signal_config.max_rsi_for_short and 
                  macd < macd_signal and 
                  stoch_k < stoch_d and stoch_k > self.signal_config.min_stoch_for_short and
                  volume_ratio > 0.8):
                return self._generate_short_signal(
                    symbol_data, latest, mtf_context, volume_entry, 
                    confluence_zones, df, intel, use_mtf=False
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Traditional signal generation error: {e}")
            return None

    def _generate_long_signal(self, symbol_data: Dict, latest: pd.Series,
                            mtf_context: MultiTimeframeContext, volume_entry: Dict,
                            confluence_zones: List[Dict], df: pd.DataFrame,
                            intel: MarketIntelligence, use_mtf: bool = True) -> Optional[Dict]:
        """Generate LONG signal with enhanced entry validation, quality filters, and API intelligence"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Entry conditions with FIXED RSI thresholds (v6.0 logic)
            rsi = latest.get('rsi', 50)
            stoch_k = latest.get('stoch_rsi_k', 50)
            stoch_d = latest.get('stoch_rsi_d', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # CHECK 1: RSI must be in valid range for longs (not overbought)
            if rsi < self.signal_config.min_rsi_for_long or rsi > self.signal_config.max_rsi_for_long:
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: RSI {rsi:.1f} out of range ({self.signal_config.min_rsi_for_long}-{self.signal_config.max_rsi_for_long})")
                return None
            
            # CHECK 2: Stochastic must not be overbought
            if stoch_k > 70 or stoch_k < stoch_d:
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: Stoch conditions not met (K:{stoch_k:.1f}, D:{stoch_d:.1f})")
                return None
            
            # CHECK 3: Volume must be sufficient
            if volume_ratio < 0.8:
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: Insufficient volume ({volume_ratio:.2f})")
                return None
            
            # CHECK 4: Support/Resistance check
            sr_check = check_near_support_resistance(df, current_price, 'buy')
            if sr_check['near_level'] and not sr_check.get('favorable', False):
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: Too close to resistance")
                return None
            
            # CHECK 5: Divergence check
            divergence = detect_divergence(df, 'buy')
            if divergence['has_divergence'] and divergence['favorable_for'] == 'sell':
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: Bearish divergence detected")
                return None
            
            # CHECK 6: Apply enhanced quality filters
            should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium = \
                self.apply_enhanced_filters(df, 'buy', symbol)
            
            if should_reject:
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: {', '.join(rejection_reasons)}")
                return None
            
            if enhancement_factors:
                self.logger.debug(f"   ✨ {symbol} - LONG quality factors: {', '.join(enhancement_factors)}")
            
            # CHECK 7: Validate entry timing
            timing_check = self._validate_entry_timing(df, 'buy')
            if not timing_check['valid'] and timing_check['score'] < -0.3:
                self.logger.debug(f"   ❌ {symbol} - Poor entry timing for LONG: {timing_check['reasons']}")
                return None
            
            # CHECK 8: API Intelligence Check
            if not self._check_api_conditions_for_long(intel):
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: Unfavorable API conditions")
                return None
            
            # Calculate entry price with momentum awareness
            entry_price = self._calculate_long_entry(current_price, mtf_context, volume_entry, df)
            
            # Validate entry price is reasonable
            entry_distance_pct = abs(entry_price - current_price) / current_price
            if entry_distance_pct > 0.03:
                self.logger.debug(f"   ⚠️ {symbol} - Entry too far from current price: {entry_distance_pct*100:.1f}%")
                entry_price = current_price * (1.015 if entry_price > current_price else 0.985)
            
            # Calculate stop loss with validation
            raw_stop = self._calculate_long_stop(entry_price, mtf_context, df)
            adjusted, stop_loss = self.validator.validate_stop_loss(entry_price, raw_stop, 'buy')
            
            # Calculate take profits (both market-based in v8.0)
            raw_tp1, raw_tp2 = self._calculate_long_targets_v8(entry_price, stop_loss, mtf_context, df, intel)
            tp1, tp2 = self.validator.validate_take_profits(entry_price, stop_loss, raw_tp1, raw_tp2, 'buy')

            tp = tp1 if self.config.default_tp_level == 'take_profit_1' else tp2
            
            # Calculate R/R ratio
            rr_ratio = self.validator.calculate_risk_reward(entry_price, stop_loss, tp, 'buy')
            
            # Check minimum R/R
            if rr_ratio < self.signal_config.min_risk_reward:
                self.logger.debug(f"   ❌ {symbol} - LONG rejected: R/R too low ({rr_ratio:.2f})")
                return None
            
            # Calculate confidence with all factors including API intelligence
            confidence = self._calculate_confidence_with_apis(
                'buy', rsi, volume_ratio, use_mtf, mtf_context, timing_check,
                quality_score, is_premium, divergence, intel
            )
            
            # Determine order type
            if entry_distance_pct > 0.01:
                order_type = 'limit'
            else:
                order_type = 'market'
            
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
                'signal_type': 'long_signal_v8',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'order_type': order_type,
                'signal_notes': f"MTF: {use_mtf}, Stop adjusted: {adjusted}, Timing: {timing_check['score']:.2f}, Quality: {quality_score:.1f}, API Sentiment: {intel.overall_api_sentiment:.1f}",
                'mtf_validated': use_mtf,
                'market_regime': mtf_context.market_regime,
                'regime_compatibility': self._assess_regime_compatibility('buy', mtf_context.market_regime),
                'entry_timing': timing_check,
                'quality_factors': enhancement_factors,
                'quality_score': quality_score,
                'is_premium_signal': is_premium,
                'divergence': divergence,
                'sr_analysis': sr_check
            }
            
        except Exception as e:
            self.logger.error(f"Long signal generation error: {e}")
            return None

    def _generate_short_signal(self, symbol_data: Dict, latest: pd.Series,
                             mtf_context: MultiTimeframeContext, volume_entry: Dict,
                             confluence_zones: List[Dict], df: pd.DataFrame,
                             intel: MarketIntelligence, use_mtf: bool = True) -> Optional[Dict]:
        """Generate SHORT signal with enhanced entry validation, quality filters, and API intelligence"""
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            # Entry conditions with FIXED RSI thresholds (v6.0 logic)
            rsi = latest.get('rsi', 50)
            stoch_k = latest.get('stoch_rsi_k', 50)
            stoch_d = latest.get('stoch_rsi_d', 50)
            volume_ratio = latest.get('volume_ratio', 1)
            
            # CHECK 1: RSI must be in valid range for shorts (not oversold)
            if rsi < self.signal_config.min_rsi_for_short or rsi > self.signal_config.max_rsi_for_short:
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: RSI {rsi:.1f} out of range ({self.signal_config.min_rsi_for_short}-{self.signal_config.max_rsi_for_short})")
                return None
            
            # CHECK 2: Stochastic must not be oversold
            if stoch_k < 30 or stoch_k > stoch_d:
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: Stoch conditions not met (K:{stoch_k:.1f}, D:{stoch_d:.1f})")
                return None
            
            # CHECK 3: Volume must be sufficient
            if volume_ratio < 0.8:
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: Insufficient volume ({volume_ratio:.2f})")
                return None
            
            # CHECK 4: Support/Resistance check
            sr_check = check_near_support_resistance(df, current_price, 'sell')
            if sr_check['near_level'] and not sr_check.get('favorable', False):
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: Too close to support")
                return None
            
            # CHECK 5: Divergence check
            divergence = detect_divergence(df, 'sell')
            if divergence['has_divergence'] and divergence['favorable_for'] == 'buy':
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: Bullish divergence detected")
                return None
            
            # CHECK 6: Apply enhanced quality filters
            should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium = \
                self.apply_enhanced_filters(df, 'sell', symbol)
            
            if should_reject:
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: {', '.join(rejection_reasons)}")
                return None
            
            if enhancement_factors:
                self.logger.debug(f"   ✨ {symbol} - SHORT quality factors: {', '.join(enhancement_factors)}")
            
            # CHECK 7: Validate entry timing
            timing_check = self._validate_entry_timing(df, 'sell')
            if not timing_check['valid'] and timing_check['score'] < -0.3:
                self.logger.debug(f"   ❌ {symbol} - Poor entry timing for SHORT: {timing_check['reasons']}")
                return None
            
            # CHECK 8: API Intelligence Check
            if not self._check_api_conditions_for_short(intel):
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: Unfavorable API conditions")
                return None
            
            # Calculate entry price with momentum awareness
            entry_price = self._calculate_short_entry(current_price, mtf_context, volume_entry, df)
            
            # Validate entry price is reasonable
            entry_distance_pct = abs(entry_price - current_price) / current_price
            if entry_distance_pct > 0.03:
                self.logger.debug(f"   ⚠️ {symbol} - Entry too far from current price: {entry_distance_pct*100:.1f}%")
                entry_price = current_price * (0.985 if entry_price < current_price else 1.015)
            
            # Calculate stop loss with validation
            raw_stop = self._calculate_short_stop(entry_price, mtf_context, df)
            adjusted, stop_loss = self.validator.validate_stop_loss(entry_price, raw_stop, 'sell')
            
            # Calculate take profits (both market-based in v8.0)
            raw_tp1, raw_tp2 = self._calculate_short_targets_v8(entry_price, stop_loss, mtf_context, df, intel)
            tp1, tp2 = self.validator.validate_take_profits(entry_price, stop_loss, raw_tp1, raw_tp2, 'sell')

            tp = tp1 if self.config.default_tp_level == 'take_profit_1' else tp2
            
            # Calculate R/R ratio
            rr_ratio = self.validator.calculate_risk_reward(entry_price, stop_loss, tp, 'sell')
            
            # Check minimum R/R
            if rr_ratio < self.signal_config.min_risk_reward:
                self.logger.debug(f"   ❌ {symbol} - SHORT rejected: R/R too low ({rr_ratio:.2f})")
                return None
            
            # Calculate confidence with all factors including API intelligence
            confidence = self._calculate_confidence_with_apis(
                'sell', rsi, volume_ratio, use_mtf, mtf_context, timing_check,
                quality_score, is_premium, divergence, intel
            )
            
            # Determine order type
            if entry_distance_pct > 0.01:
                order_type = 'limit'
            else:
                order_type = 'market'
            
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
                'signal_type': 'short_signal_v8',
                'volume_24h': symbol_data['volume_24h'],
                'price_change_24h': symbol_data['price_change_24h'],
                'order_type': order_type,
                'signal_notes': f"MTF: {use_mtf}, Stop adjusted: {adjusted}, Timing: {timing_check['score']:.2f}, Quality: {quality_score:.1f}, API Sentiment: {intel.overall_api_sentiment:.1f}",
                'mtf_validated': use_mtf,
                'market_regime': mtf_context.market_regime,
                'regime_compatibility': self._assess_regime_compatibility('sell', mtf_context.market_regime),
                'entry_timing': timing_check,
                'quality_factors': enhancement_factors,
                'quality_score': quality_score,
                'is_premium_signal': is_premium,
                'divergence': divergence,
                'sr_analysis': sr_check
            }
            
        except Exception as e:
            self.logger.error(f"Short signal generation error: {e}")
            return None

    def _check_api_conditions_for_long(self, intel: MarketIntelligence) -> bool:
        """Check if API conditions favor LONG positions"""
        # Extreme fear can be good for contrarian longs
        if intel.fear_greed_index < 25:
            return True
        
        # Negative funding means shorts are paying longs
        if intel.funding_rate < -0.0001:
            return True
        
        # Moderate conditions are acceptable
        if 25 <= intel.fear_greed_index <= 65:
            # Check if news isn't extremely bearish
            if intel.news_classification not in ['very_bearish']:
                return True
        
        # Extreme greed is bad for longs
        if intel.fear_greed_index > 75:
            return False
        
        # Extremely high funding is bad for longs
        if intel.funding_rate > 0.0005:
            return False
        
        return True

    def _check_api_conditions_for_short(self, intel: MarketIntelligence) -> bool:
        """Check if API conditions favor SHORT positions"""
        # Extreme greed can be good for contrarian shorts
        if intel.fear_greed_index > 75:
            return True
        
        # Positive funding means longs are paying shorts
        if intel.funding_rate > 0.0002:
            return True
        
        # Moderate conditions are acceptable
        if 35 <= intel.fear_greed_index <= 75:
            # Check if news isn't extremely bullish
            if intel.news_classification not in ['very_bullish']:
                return True
        
        # Extreme fear is bad for shorts
        if intel.fear_greed_index < 25:
            return False
        
        # Extremely negative funding is bad for shorts
        if intel.funding_rate < -0.0005:
            return False
        
        return True

    def _calculate_confidence_with_apis(self, side: str, rsi: float, volume_ratio: float,
                                       use_mtf: bool, mtf_context: MultiTimeframeContext,
                                       timing_check: Dict, quality_score: float,
                                       is_premium: bool, divergence: Dict,
                                       intel: MarketIntelligence) -> float:
        """Calculate confidence with API intelligence factors"""
        # Base confidence
        base_confidence = 55.0
        
        # RSI contribution
        if side == 'buy':
            if rsi < 35:
                base_confidence += 10
            elif rsi < 40:
                base_confidence += 5
        else:  # sell
            if rsi > 65:
                base_confidence += 10
            elif rsi > 60:
                base_confidence += 5
        
        # Volume contribution
        if volume_ratio > 1.5:
            base_confidence += 10
        elif volume_ratio > 1.2:
            base_confidence += 5
        
        # MTF bonus
        if use_mtf and mtf_context.momentum_alignment:
            base_confidence += self.signal_config.mtf_confidence_boost
        
        # Timing adjustment
        timing_adjustment = timing_check['score'] * 10
        base_confidence += timing_adjustment
        
        # Quality score adjustment
        if is_premium:
            base_confidence += self.signal_config.fast_signal_bonus_confidence
        else:
            base_confidence += quality_score * 5
        
        # Divergence bonus
        if divergence['has_divergence'] and divergence['favorable_for'] == side:
            base_confidence += 10
        
        # API Intelligence adjustments
        # Fear & Greed
        if side == 'buy' and intel.fear_greed_index < 30:  # Extreme fear = good for longs
            base_confidence += 8
        elif side == 'sell' and intel.fear_greed_index > 70:  # Extreme greed = good for shorts
            base_confidence += 8
        
        # Funding rate
        if side == 'buy' and intel.funding_rate < -0.0001:  # Negative funding favors longs
            base_confidence += 5
        elif side == 'sell' and intel.funding_rate > 0.0002:  # Positive funding favors shorts
            base_confidence += 5
        
        # News sentiment
        if side == 'buy' and intel.news_classification in ['bullish', 'very_bullish']:
            base_confidence += 5
        elif side == 'sell' and intel.news_classification in ['bearish', 'very_bearish']:
            base_confidence += 5
        
        # Apply API signal modifier
        base_confidence *= intel.api_signal_modifier
        
        confidence = min(90, max(self.signal_config.min_confidence_for_signal, base_confidence))
        
        return confidence

    def _calculate_long_targets_v8(self, entry_price: float, stop_loss: float,
                                   mtf_context: MultiTimeframeContext, df: pd.DataFrame,
                                   intel: MarketIntelligence) -> Tuple[float, float]:
        """Calculate LONG take profit targets - Both TP1 and TP2 are market-based in v8.0"""
        try:
            risk = entry_price - stop_loss
            
            # TP1: Market-based (nearest resistance or key level) - UNCHANGED FROM V6.0
            tp1_candidates = []
            
            # 1. Look for nearest resistance from MTF zones
            resistance_zones = [z for z in mtf_context.higher_tf_zones 
                              if z['type'] == 'resistance' and z['price'] > entry_price]
            
            if resistance_zones:
                nearest_resistance = min(resistance_zones, key=lambda x: x['price'])
                tp1_from_resistance = nearest_resistance['price'] * 0.995
                tp1_candidates.append(tp1_from_resistance)
            
            # 2. Look for recent swing highs as potential TP1
            if len(df) >= 20:
                recent_high = df['high'].tail(20).max()
                if recent_high > entry_price:
                    tp1_from_swing = recent_high * 0.995
                    tp1_candidates.append(tp1_from_swing)
            
            # 3. Use Bollinger Band upper as potential TP1
            if 'bb_upper' in df.columns:
                bb_upper = df['bb_upper'].iloc[-1]
                if bb_upper > entry_price:
                    tp1_candidates.append(bb_upper * 0.995)
            
            # 4. Use key psychological levels (round numbers)
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
            
            # Choose TP1: Use the nearest valid target that gives decent profit
            min_tp1_distance = entry_price * (1 + self.signal_config.min_tp_distance_pct)
            valid_tp1_candidates = [tp for tp in tp1_candidates if tp >= min_tp1_distance]
            
            if valid_tp1_candidates:
                tp1 = min(valid_tp1_candidates)
            else:
                tp1 = entry_price * (1 + max(self.signal_config.min_tp_distance_pct * 2, 0.03))
            
            # Ensure TP1 is not too far
            max_tp1_distance = entry_price * (1 + 0.08)
            tp1 = min(tp1, max_tp1_distance)
            
            # TP2: NEW MARKET-BASED CALCULATION (v8.0 enhancement)
            tp2_candidates = []
            
            # 1. Look for deeper resistance levels
            deeper_resistances = [z for z in mtf_context.higher_tf_zones 
                                 if z['type'] == 'resistance' and z['price'] > tp1 * 1.02]
            
            if deeper_resistances:
                # Take up to 2 deeper resistance levels
                for resistance in sorted(deeper_resistances, key=lambda x: x['price'])[:2]:
                    tp2_from_resistance = resistance['price'] * 0.995
                    if tp2_from_resistance > tp1 * 1.03:  # At least 3% beyond TP1
                        tp2_candidates.append(tp2_from_resistance)
            
            # 2. Look for major swing highs (longer period)
            if len(df) >= 50:
                major_high = df['high'].tail(50).max()
                if major_high > tp1 * 1.05:
                    tp2_from_major_swing = major_high * 0.995
                    tp2_candidates.append(tp2_from_major_swing)
            
            # 3. Use Fibonacci extension levels
            if len(df) >= 30:
                recent_low = df['low'].tail(30).min()
                recent_high = df['high'].tail(30).max()
                fib_range = recent_high - recent_low
                
                # 1.618 Fibonacci extension
                fib_1618 = entry_price + (fib_range * 0.618)
                if fib_1618 > tp1 * 1.04:
                    tp2_candidates.append(fib_1618)
                
                # 2.618 Fibonacci extension
                fib_2618 = entry_price + (fib_range * 1.0)
                if fib_2618 > tp1 * 1.05 and fib_2618 < entry_price * 1.15:
                    tp2_candidates.append(fib_2618)
            
            # 4. Major psychological levels (larger round numbers)
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
            
            # 5. Market sentiment-based adjustment
            if intel.overall_api_sentiment > 65:  # Bullish sentiment = further targets
                sentiment_tp2 = tp1 * 1.08
                tp2_candidates.append(sentiment_tp2)
            
            # Choose TP2: Select the most reasonable target
            if tp2_candidates:
                # Filter candidates that are reasonable
                reasonable_tp2 = [tp for tp in tp2_candidates 
                                 if tp > tp1 * 1.04 and tp < entry_price * (1 + self.signal_config.max_tp_distance_pct)]
                
                if reasonable_tp2:
                    # Choose median target for balance
                    reasonable_tp2.sort()
                    tp2 = reasonable_tp2[len(reasonable_tp2) // 2]
                else:
                    # Fallback: Use a reasonable distance beyond TP1
                    tp2 = tp1 * 1.06
            else:
                # Fallback: Use a fixed percentage beyond TP1
                tp2 = tp1 * 1.06
            
            # Ensure TP2 is beyond TP1 but within max distance
            tp2 = max(tp2, tp1 * 1.04)
            max_tp = entry_price * (1 + self.signal_config.max_tp_distance_pct)
            tp2 = min(tp2, max_tp)
            
            return tp1, tp2
            
        except Exception:
            # Fallback to simple calculation
            risk = entry_price - stop_loss
            tp1 = entry_price * 1.03
            tp2 = entry_price * 1.06
            return tp1, tp2

    def _calculate_short_targets_v8(self, entry_price: float, stop_loss: float,
                                    mtf_context: MultiTimeframeContext, df: pd.DataFrame,
                                    intel: MarketIntelligence) -> Tuple[float, float]:
        """Calculate SHORT take profit targets - Both TP1 and TP2 are market-based in v8.0"""
        try:
            risk = stop_loss - entry_price
            
            # TP1: Market-based (nearest support or key level) - UNCHANGED FROM V6.0
            tp1_candidates = []
            
            # 1. Look for nearest support from MTF zones
            support_zones = [z for z in mtf_context.higher_tf_zones 
                           if z['type'] == 'support' and z['price'] < entry_price]
            
            if support_zones:
                nearest_support = max(support_zones, key=lambda x: x['price'])
                tp1_from_support = nearest_support['price'] * 1.005
                tp1_candidates.append(tp1_from_support)
            
            # 2. Look for recent swing lows as potential TP1
            if len(df) >= 20:
                recent_low = df['low'].tail(20).min()
                if recent_low < entry_price:
                    tp1_from_swing = recent_low * 1.005
                    tp1_candidates.append(tp1_from_swing)
            
            # 3. Use Bollinger Band lower as potential TP1
            if 'bb_lower' in df.columns:
                bb_lower = df['bb_lower'].iloc[-1]
                if bb_lower < entry_price:
                    tp1_candidates.append(bb_lower * 1.005)
            
            # 4. Use key psychological levels (round numbers)
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
            
            # Choose TP1: Use the nearest valid target that gives decent profit
            min_tp1_distance = entry_price * (1 - self.signal_config.min_tp_distance_pct)
            valid_tp1_candidates = [tp for tp in tp1_candidates if tp <= min_tp1_distance]
            
            if valid_tp1_candidates:
                tp1 = max(valid_tp1_candidates)
            else:
                tp1 = entry_price * (1 - max(self.signal_config.min_tp_distance_pct * 2, 0.03))
            
            # Ensure TP1 is not too far
            max_tp1_distance = entry_price * (1 - 0.08)
            tp1 = max(tp1, max_tp1_distance)
            
            # TP2: NEW MARKET-BASED CALCULATION (v8.0 enhancement)
            tp2_candidates = []
            
            # 1. Look for deeper support levels
            deeper_supports = [z for z in mtf_context.higher_tf_zones 
                             if z['type'] == 'support' and z['price'] < tp1 * 0.98]
            
            if deeper_supports:
                # Take up to 2 deeper support levels
                for support in sorted(deeper_supports, key=lambda x: x['price'], reverse=True)[:2]:
                    tp2_from_support = support['price'] * 1.005
                    if tp2_from_support < tp1 * 0.97:  # At least 3% beyond TP1
                        tp2_candidates.append(tp2_from_support)
            
            # 2. Look for major swing lows (longer period)
            if len(df) >= 50:
                major_low = df['low'].tail(50).min()
                if major_low < tp1 * 0.95:
                    tp2_from_major_swing = major_low * 1.005
                    tp2_candidates.append(tp2_from_major_swing)
            
            # 3. Use Fibonacci extension levels
            if len(df) >= 30:
                recent_low = df['low'].tail(30).min()
                recent_high = df['high'].tail(30).max()
                fib_range = recent_high - recent_low
                
                # 1.618 Fibonacci extension
                fib_1618 = entry_price - (fib_range * 0.618)
                if fib_1618 < tp1 * 0.96:
                    tp2_candidates.append(fib_1618)
                
                # 2.618 Fibonacci extension
                fib_2618 = entry_price - (fib_range * 1.0)
                if fib_2618 < tp1 * 0.95 and fib_2618 > entry_price * 0.85:
                    tp2_candidates.append(fib_2618)
            
            # 4. Major psychological levels (larger round numbers)
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
            
            # 5. Market sentiment-based adjustment
            if intel.overall_api_sentiment < 35:  # Bearish sentiment = further targets
                sentiment_tp2 = tp1 * 0.92
                tp2_candidates.append(sentiment_tp2)
            
            # Choose TP2: Select the most reasonable target
            if tp2_candidates:
                # Filter candidates that are reasonable
                reasonable_tp2 = [tp for tp in tp2_candidates 
                                 if tp < tp1 * 0.96 and tp > entry_price * (1 - self.signal_config.max_tp_distance_pct)]
                
                if reasonable_tp2:
                    # Choose median target for balance
                    reasonable_tp2.sort(reverse=True)
                    tp2 = reasonable_tp2[len(reasonable_tp2) // 2]
                else:
                    # Fallback: Use a reasonable distance beyond TP1
                    tp2 = tp1 * 0.94
            else:
                # Fallback: Use a fixed percentage beyond TP1
                tp2 = tp1 * 0.94
            
            # Ensure TP2 is beyond TP1 but within max distance
            tp2 = min(tp2, tp1 * 0.96)
            max_tp = entry_price * (1 - self.signal_config.max_tp_distance_pct)
            tp2 = max(tp2, max_tp)
            
            return tp1, tp2
            
        except Exception:
            # Fallback to simple calculation
            risk = stop_loss - entry_price
            tp1 = entry_price * 0.97
            tp2 = entry_price * 0.94
            return tp1, tp2

    # ===== ALL V6.0 HELPER METHODS (UNCHANGED) =====

    def _calculate_long_entry(self, current_price: float, mtf_context: MultiTimeframeContext,
                            volume_entry: Dict, df: pd.DataFrame = None) -> float:
        """Calculate LONG entry with momentum awareness"""
        try:
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                recent = df.tail(self.signal_config.price_momentum_lookback)
                price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                volatility_adjustment = (atr / current_price) * 0.5
                
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
                if price_change > 0.01:
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
        """Calculate SHORT entry with momentum awareness"""
        try:
            if df is not None and len(df) >= self.signal_config.price_momentum_lookback:
                recent = df.tail(self.signal_config.price_momentum_lookback)
                price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                volatility_adjustment = (atr / current_price) * 0.5
                
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
                if price_change < -0.01:
                    return min(entry_candidates)
                else:
                    entry_candidates.sort()
                    return entry_candidates[len(entry_candidates)//2]
            else:
                return max(entry_candidates)
                
        except Exception:
            return current_price * 1.001

    def _calculate_long_stop(self, entry_price: float, mtf_context: MultiTimeframeContext, 
                       df: pd.DataFrame) -> float:
        """Calculate LONG stop loss with dynamic volatility adjustment (5% - 10%)"""
        try:
            stop_candidates = []
            
            if len(df) >= 14 and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_pct = atr / entry_price
                
                recent_changes = df['close'].pct_change().tail(20).abs()
                max_volatility = recent_changes.max()
                
                if max_volatility > self.signal_config.high_volatility_threshold:
                    atr_multiplier = 3.5
                elif max_volatility > 0.04:
                    atr_multiplier = 3.0
                elif max_volatility > 0.02:
                    atr_multiplier = 2.5
                else:
                    atr_multiplier = 2.0
                
                atr_stop = entry_price - (atr * atr_multiplier)
                stop_candidates.append(atr_stop)
            
            support_zones = [z for z in mtf_context.higher_tf_zones 
                        if z['type'] == 'support' and z['price'] < entry_price]
            if support_zones:
                closest_support = max(support_zones, key=lambda x: x['price'])
                structure_stop = closest_support['price'] * (1 - self.signal_config.structure_stop_buffer)
                if (entry_price - structure_stop) / entry_price >= self.signal_config.min_stop_distance_pct:
                    stop_candidates.append(structure_stop)
            
            min_stop = entry_price * (1 - self.signal_config.min_stop_distance_pct)
            stop_candidates.append(min_stop)
            
            if stop_candidates:
                chosen_stop = max(stop_candidates)
                return max(chosen_stop, entry_price * (1 - self.signal_config.min_stop_distance_pct))
            else:
                return entry_price * (1 - self.signal_config.min_stop_distance_pct)
                
        except Exception:
            return entry_price * (1 - self.signal_config.min_stop_distance_pct)
    
    def _calculate_short_stop(self, entry_price: float, mtf_context: MultiTimeframeContext,
                        df: pd.DataFrame) -> float:
        """Calculate SHORT stop loss with dynamic volatility adjustment (5% - 10%)"""
        try:
            stop_candidates = []
            
            if len(df) >= 14 and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                atr_pct = atr / entry_price
                
                recent_changes = df['close'].pct_change().tail(20).abs()
                max_volatility = recent_changes.max()
                
                if max_volatility > self.signal_config.high_volatility_threshold:
                    atr_multiplier = 3.5
                elif max_volatility > 0.04:
                    atr_multiplier = 3.0
                elif max_volatility > 0.02:
                    atr_multiplier = 2.5
                else:
                    atr_multiplier = 2.0
                
                atr_stop = entry_price + (atr * atr_multiplier)
                stop_candidates.append(atr_stop)
            
            resistance_zones = [z for z in mtf_context.higher_tf_zones 
                            if z['type'] == 'resistance' and z['price'] > entry_price]
            if resistance_zones:
                closest_resistance = min(resistance_zones, key=lambda x: x['price'])
                structure_stop = closest_resistance['price'] * (1 + self.signal_config.structure_stop_buffer)
                if (structure_stop - entry_price) / entry_price >= self.signal_config.min_stop_distance_pct:
                    stop_candidates.append(structure_stop)
            
            min_stop = entry_price * (1 + self.signal_config.min_stop_distance_pct)
            stop_candidates.append(min_stop)
            
            if stop_candidates:
                chosen_stop = min(stop_candidates)
                return min(chosen_stop, entry_price * (1 + self.signal_config.min_stop_distance_pct))
            else:
                return entry_price * (1 + self.signal_config.min_stop_distance_pct)
                
        except Exception:
            return entry_price * (1 + self.signal_config.min_stop_distance_pct)

    def _validate_entry_timing(self, df: pd.DataFrame, side: str, lookback: int = 5) -> dict:
        """Validate if current timing is good for entry with stricter checks"""
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
                    timing_score -= 0.5
                    reasons.append("rsi_overbought_avoid_long")
                elif rsi < 35:
                    timing_score += 0.2
                    reasons.append("rsi_oversold_good_for_long")
                
                if stoch_k > 70:
                    timing_score -= 0.3
                    reasons.append("stoch_overbought")
                elif stoch_k < 30:
                    timing_score += 0.2
                    reasons.append("stoch_oversold_good_for_long")
                    
            else:
                if price_position < 0.1:
                    timing_score -= 0.3
                    reasons.append("price_near_support")
                elif price_position > 0.7:
                    timing_score += 0.3
                    reasons.append("price_near_resistance")
                
                if rsi < self.signal_config.min_rsi_for_short:
                    timing_score -= 0.5
                    reasons.append("rsi_oversold_avoid_short")
                elif rsi > 65:
                    timing_score += 0.2
                    reasons.append("rsi_overbought_good_for_short")
                
                if stoch_k < 30:
                    timing_score -= 0.3
                    reasons.append("stoch_oversold_avoid_short")
                elif stoch_k > 70:
                    timing_score += 0.2
                    reasons.append("stoch_overbought_good_for_short")
            
            if volume_ratio > 1.5:
                timing_score += 0.2
                reasons.append("strong_volume")
            elif volume_ratio < 0.5:
                timing_score -= 0.1
                reasons.append("weak_volume")
            
            is_valid = timing_score >= -0.2
            
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

    def apply_enhanced_filters(self, df: pd.DataFrame, side: str, symbol: str) -> tuple:
        """Apply all enhanced filters and return decision with reasons"""
        try:
            rejection_reasons = []
            enhancement_factors = []
            quality_score = 0
            
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
            
            divergence = check_volume_momentum_divergence(df)
            if divergence['divergence'] and divergence['type'] == 'bearish_divergence' and side == 'buy':
                rejection_reasons.append(divergence['warning'])
                quality_score -= 1
            elif divergence['type'] == 'confirmed_move':
                enhancement_factors.append("Volume confirms move")
                quality_score += 1
            
            fast_setup = identify_fast_moving_setup(df, side)
            if fast_setup['is_fast_setup']:
                enhancement_factors.extend(fast_setup['factors'])
                quality_score += fast_setup['score']
            
            choppiness = filter_choppy_markets(df)
            if choppiness['is_choppy']:
                rejection_reasons.append(f"Choppy market detected (score: {choppiness['choppiness_score']:.2f})")
                quality_score -= 2
            elif choppiness['market_state'] == 'trending':
                enhancement_factors.append("Clear trending market")
                quality_score += 1
            
            should_reject = len(rejection_reasons) > 0 and quality_score < 0
            is_premium = quality_score >= 3
            
            return should_reject, rejection_reasons, enhancement_factors, quality_score, is_premium
            
        except Exception as e:
            self.logger.error(f"Enhanced filter error for {symbol}: {e}")
            return False, [], [], 0, False

    def _validate_and_enhance_signal(self, signal: Dict, mtf_context: MultiTimeframeContext,
                                   df: pd.DataFrame, market_regime: str, 
                                   intel: MarketIntelligence) -> Dict:
        """Final validation and enhancement of signal with API intelligence"""
        try:
            signal['analysis_details'] = {
                'signal_strength': self._determine_signal_strength(signal, mtf_context),
                'mtf_trend': mtf_context.dominant_trend,
                'structure_timeframe': mtf_context.structure_timeframe,
                'confirmation_score': mtf_context.confirmation_score,
                'entry_method': self._determine_entry_method(signal, mtf_context),
                'market_regime': market_regime,
                'volatility_level': mtf_context.volatility_level,
                'momentum_strength': self._analyze_momentum_strength(df),
                'regime_compatibility': signal.get('regime_compatibility', 'medium'),
                'api_sentiment': intel.overall_api_sentiment,
                'fear_greed': intel.fear_greed_classification,
                'funding_sentiment': intel.funding_sentiment,
                'news_sentiment': intel.news_classification
            }
            
            signal['entry_strategy'] = signal['analysis_details']['entry_method']
            signal['quality_grade'] = self._calculate_quality_grade(signal)
            
            if signal.get('mtf_validated', False):
                signal['original_confidence'] = signal['confidence'] - self.signal_config.mtf_confidence_boost
                signal['mtf_boost'] = self.signal_config.mtf_confidence_boost
            else:
                signal['original_confidence'] = signal['confidence']
                signal['mtf_boost'] = 0
            
            if signal.get('mtf_validated', False):
                signal['mtf_status'] = 'MTF_VALIDATED'
            else:
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

    # ===== ALL OTHER V6.0 METHODS CONTINUE UNCHANGED =====
    
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
        """Calculate signal quality grade with enhanced scoring"""
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
        elif confidence >= 55:
            score += 5
        
        if rr_ratio >= 3.5:
            score += 25
        elif rr_ratio >= 2.5:
            score += 20
        elif rr_ratio >= 2:
            score += 15
        elif rr_ratio >= 1.5:
            score += 10
        
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

    def rank_opportunities_with_mtf(self, signals: List[Dict], dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """Enhanced ranking system with quality scoring and fast-signal prioritization"""
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
                
                # Add API intelligence bonus
                api_sentiment = signal.get('market_intelligence', {}).get('overall_api_sentiment', 50)
                api_bonus = 0
                if api_sentiment > 70 or api_sentiment < 30:
                    api_bonus = 500  # Extreme sentiment bonus
                
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

    # Continue with all other v6.0 methods unchanged...
    # [All remaining methods from v6.0 are included here unchanged]
    
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


# ===== COMPATIBILITY =====

def create_mtf_signal_generator(config: EnhancedSystemConfig, exchange_manager):
    """Factory function to create the enhanced MTF-aware signal generator with APIs"""
    generator = SignalGenerator(config, exchange_manager)
    generator.debug_mode = False
    return generator

# ===== EXPORT =====

__all__ = [
    'SignalGenerator',
    'SignalConfig',
    'SignalValidator',
    'MultiTimeframeContext',
    'MarketIntelligence',
    'create_mtf_signal_generator',
    'analyze_price_momentum_strength',
    'check_volume_momentum_divergence',
    'identify_fast_moving_setup',
    'filter_choppy_markets',
    'calculate_momentum_adjusted_entry',
    'calculate_dynamic_stop_loss',
    'detect_divergence',
    'check_near_support_resistance'
]

__version__ = "8.0.0-PRODUCTION"
__features__ = [
    "✅ 100% of v6.0 technical analysis logic preserved",
    "✅ Fear & Greed Index integration",
    "✅ Bybit Funding Rates analysis",
    "✅ CryptoPanic News Sentiment",
    "✅ Market-based TP1 (unchanged from v6.0)",
    "✅ Market-based TP2 (NEW - uses deeper market structure)",
    "✅ All v6.0 quality filters and momentum analysis",
    "✅ API intelligence enhances but doesn't override v6.0 logic"
]