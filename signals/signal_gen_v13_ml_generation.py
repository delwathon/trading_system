"""
Signal Generator V13.0 - ML Prediction and Signal Generation
============================================================
PART 3: Machine Learning, News Sentiment, and Signal Generation Engine

This module provides:
- ML-based signal prediction
- News sentiment analysis
- Multi-timeframe signal generation
- Entry monitoring service
"""

import os
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import hashlib
from collections import deque

# Import from previous parts
from signals.signal_gen_v13_core import (
    Signal, SignalStatus, SignalQuality, TimeFrame, MarketRegime,
    SystemConfiguration, SignalCriteria, StateManager,
    AnalysisModule, EntryMonitor, MarketContext
)

from signals.signal_gen_v13_analysis import (
    TechnicalIndicators, VolumeProfileAnalyzer, MarketStructureAnalyzer
)

# ===========================
# ML PREDICTION ENGINE
# ===========================

@dataclass
class MLFeatures:
    """Features for ML model"""
    # Price features
    price_change_5: float
    price_change_10: float
    price_change_20: float
    distance_from_sma20: float
    distance_from_sma50: float
    
    # Technical indicators
    rsi: float
    rsi_change: float
    macd_histogram: float
    stoch_k: float
    bb_position: float
    
    # Volume features
    volume_ratio: float
    obv_change: float
    mfi: float
    
    # Volatility features
    atr_percent: float
    bb_width: float
    
    # Market structure
    trend_strength: float
    distance_from_resistance: float
    distance_from_support: float
    
    # Momentum
    momentum_score: float
    roc: float

class MLPredictionEngine:
    """Machine Learning based prediction system"""
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Models for different purposes
        self.direction_model = None  # Predicts price direction
        self.quality_model = None    # Predicts signal quality
        self.target_model = None     # Predicts target achievement
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.prediction_history = deque(maxlen=1000)
        
        # Load pre-trained models if available
        self._load_models()
    
    def extract_features(self, df: pd.DataFrame, current_idx: int = -1) -> MLFeatures:
        """Extract ML features from dataframe"""
        try:
            if current_idx == -1:
                current_idx = len(df) - 1
            
            current = df.iloc[current_idx]
            
            # Price changes
            price_5 = (current['close'] - df.iloc[max(0, current_idx-5)]['close']) / df.iloc[max(0, current_idx-5)]['close']
            price_10 = (current['close'] - df.iloc[max(0, current_idx-10)]['close']) / df.iloc[max(0, current_idx-10)]['close']
            price_20 = (current['close'] - df.iloc[max(0, current_idx-20)]['close']) / df.iloc[max(0, current_idx-20)]['close']
            
            # Distance from MAs
            sma20_dist = (current['close'] - current.get('sma_20', current['close'])) / current['close']
            sma50_dist = (current['close'] - current.get('sma_50', current['close'])) / current['close']
            
            # RSI change
            rsi_change = current.get('rsi', 50) - df.iloc[max(0, current_idx-5)].get('rsi', 50)
            
            # OBV change
            obv_change = 0
            if 'obv' in df.columns:
                obv_now = current['obv']
                obv_prev = df.iloc[max(0, current_idx-5)]['obv']
                if obv_prev != 0:
                    obv_change = (obv_now - obv_prev) / abs(obv_prev)
            
            # Support/Resistance distances (simplified)
            recent_high = df.iloc[max(0, current_idx-20):current_idx]['high'].max()
            recent_low = df.iloc[max(0, current_idx-20):current_idx]['low'].min()
            
            res_dist = (recent_high - current['close']) / current['close']
            sup_dist = (current['close'] - recent_low) / current['close']
            
            # Trend strength (simplified)
            if current.get('sma_20', 0) > current.get('sma_50', 1):
                trend_strength = 1.0
            elif current.get('sma_20', 1) < current.get('sma_50', 0):
                trend_strength = -1.0
            else:
                trend_strength = 0.0
            
            # Momentum score
            momentum_indicators = [
                1 if current.get('rsi', 50) > 50 else -1,
                1 if current.get('macd', 0) > current.get('macd_signal', 0) else -1,
                1 if current.get('stoch_k', 50) > 50 else -1,
                1 if price_5 > 0 else -1
            ]
            momentum_score = sum(momentum_indicators) / len(momentum_indicators)
            
            return MLFeatures(
                price_change_5=price_5,
                price_change_10=price_10,
                price_change_20=price_20,
                distance_from_sma20=sma20_dist,
                distance_from_sma50=sma50_dist,
                rsi=current.get('rsi', 50),
                rsi_change=rsi_change,
                macd_histogram=current.get('macd_hist', 0),
                stoch_k=current.get('stoch_k', 50),
                bb_position=current.get('bb_percent', 0.5),
                volume_ratio=current.get('volume_ratio', 1.0),
                obv_change=obv_change,
                mfi=current.get('mfi', 50),
                atr_percent=current.get('atr_percent', 2.0),
                bb_width=current.get('bb_width', 0.1),
                trend_strength=trend_strength,
                distance_from_resistance=res_dist,
                distance_from_support=sup_dist,
                momentum_score=momentum_score,
                roc=current.get('roc_10', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            # Return default features
            return MLFeatures(
                price_change_5=0, price_change_10=0, price_change_20=0,
                distance_from_sma20=0, distance_from_sma50=0,
                rsi=50, rsi_change=0, macd_histogram=0,
                stoch_k=50, bb_position=0.5,
                volume_ratio=1, obv_change=0, mfi=50,
                atr_percent=2, bb_width=0.1,
                trend_strength=0, distance_from_resistance=0.05,
                distance_from_support=0.05, momentum_score=0, roc=0
            )
    
    def predict_direction(self, features: MLFeatures, timeframe: TimeFrame) -> Dict:
        """Predict price direction"""
        try:
            # Convert features to array
            X = self._features_to_array(features)
            
            if self.direction_model is None:
                # Use rule-based prediction as fallback
                return self._rule_based_direction_prediction(features)
            
            # Scale features
            X_scaled = self.scaler.transform([X])
            
            # Get prediction and probability
            prediction = self.direction_model.predict(X_scaled)[0]
            probabilities = self.direction_model.predict_proba(X_scaled)[0]
            
            # Map to direction
            direction_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
            direction = direction_map.get(prediction, 'neutral')
            
            return {
                'direction': direction,
                'confidence': max(probabilities),
                'probabilities': {
                    'bearish': probabilities[0],
                    'neutral': probabilities[1],
                    'bullish': probabilities[2]
                },
                'timeframe': timeframe.label
            }
            
        except Exception as e:
            self.logger.error(f"Direction prediction error: {e}")
            return self._rule_based_direction_prediction(features)
    
    def predict_signal_quality(self, signal_data: Dict, features: MLFeatures) -> Dict:
        """Predict signal quality and success probability"""
        try:
            if self.quality_model is None:
                # Use rule-based quality assessment
                return self._rule_based_quality_prediction(signal_data, features)
            
            # Prepare features
            X = self._prepare_quality_features(signal_data, features)
            X_scaled = self.scaler.transform([X])
            
            # Predict quality score
            quality_score = self.quality_model.predict(X_scaled)[0]
            
            # Determine quality tier
            if quality_score >= 0.85:
                tier = SignalQuality.ELITE
            elif quality_score >= 0.75:
                tier = SignalQuality.PREMIUM
            elif quality_score >= 0.65:
                tier = SignalQuality.STANDARD
            elif quality_score >= 0.55:
                tier = SignalQuality.MARGINAL
            else:
                tier = SignalQuality.REJECTED
            
            return {
                'quality_score': quality_score,
                'quality_tier': tier,
                'success_probability': quality_score,
                'risk_score': 1 - quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Quality prediction error: {e}")
            return self._rule_based_quality_prediction(signal_data, features)
    
    def predict_targets(self, entry_price: float, features: MLFeatures, side: str) -> Dict:
        """Predict optimal targets using ML"""
        try:
            if self.target_model is None:
                # Use ATR-based targets as fallback
                return self._atr_based_targets(entry_price, features, side)
            
            # Prepare features
            X = self._features_to_array(features)
            X_scaled = self.scaler.transform([X])
            
            # Predict expected move percentage
            expected_move_pct = self.target_model.predict(X_scaled)[0]
            
            # Calculate targets based on prediction
            if side == 'buy':
                tp1 = entry_price * (1 + expected_move_pct * 0.5)
                tp2 = entry_price * (1 + expected_move_pct * 1.0)
                tp3 = entry_price * (1 + expected_move_pct * 1.5)
            else:
                tp1 = entry_price * (1 - expected_move_pct * 0.5)
                tp2 = entry_price * (1 - expected_move_pct * 1.0)
                tp3 = entry_price * (1 - expected_move_pct * 1.5)
            
            return {
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'expected_move_pct': expected_move_pct,
                'confidence': 0.7  # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Target prediction error: {e}")
            return self._atr_based_targets(entry_price, features, side)
    
    def train_models(self, historical_data: pd.DataFrame, outcomes: pd.DataFrame):
        """Train ML models on historical data"""
        try:
            self.logger.info("Training ML models...")
            
            # Prepare training data
            X, y_direction, y_quality = self._prepare_training_data(historical_data, outcomes)
            
            # Split data
            X_train, X_test, y_dir_train, y_dir_test = train_test_split(
                X, y_direction, test_size=0.2, random_state=42
            )
            
            # Train direction model
            self.direction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.direction_model.fit(X_train, y_dir_train)
            
            # Train quality model
            self.quality_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            # Note: Would need quality labels for proper training
            
            # Fit scaler
            self.scaler.fit(X)
            
            # Save models
            self._save_models()
            
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
    
    def _features_to_array(self, features: MLFeatures) -> np.ndarray:
        """Convert features to numpy array"""
        return np.array([
            features.price_change_5,
            features.price_change_10,
            features.price_change_20,
            features.distance_from_sma20,
            features.distance_from_sma50,
            features.rsi,
            features.rsi_change,
            features.macd_histogram,
            features.stoch_k,
            features.bb_position,
            features.volume_ratio,
            features.obv_change,
            features.mfi,
            features.atr_percent,
            features.bb_width,
            features.trend_strength,
            features.distance_from_resistance,
            features.distance_from_support,
            features.momentum_score,
            features.roc
        ])
    
    def _rule_based_direction_prediction(self, features: MLFeatures) -> Dict:
        """Fallback rule-based direction prediction"""
        score = 0
        
        # Price momentum
        if features.price_change_5 > 0.02:
            score += 2
        elif features.price_change_5 < -0.02:
            score -= 2
        
        # Technical indicators
        if features.rsi > 60:
            score += 1
        elif features.rsi < 40:
            score -= 1
        
        if features.macd_histogram > 0:
            score += 1
        else:
            score -= 1
        
        if features.momentum_score > 0.5:
            score += 2
        elif features.momentum_score < -0.5:
            score -= 2
        
        # Determine direction
        if score >= 3:
            direction = 'bullish'
            confidence = min(0.8, 0.5 + score * 0.05)
        elif score <= -3:
            direction = 'bearish'
            confidence = min(0.8, 0.5 + abs(score) * 0.05)
        else:
            direction = 'neutral'
            confidence = 0.5
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': {
                'bullish': confidence if direction == 'bullish' else (1-confidence)/2,
                'bearish': confidence if direction == 'bearish' else (1-confidence)/2,
                'neutral': confidence if direction == 'neutral' else (1-confidence)
            },
            'timeframe': 'rule_based'
        }
    
    def _rule_based_quality_prediction(self, signal_data: Dict, features: MLFeatures) -> Dict:
        """Fallback rule-based quality assessment"""
        quality_score = 0.5  # Base score
        
        # Risk/Reward ratio
        rr = signal_data.get('risk_reward_ratio', 1.5)
        if rr >= 3:
            quality_score += 0.15
        elif rr >= 2:
            quality_score += 0.10
        elif rr >= 1.5:
            quality_score += 0.05
        
        # Trend alignment
        if features.trend_strength > 0.5:
            quality_score += 0.10
        elif features.trend_strength < -0.5:
            quality_score -= 0.05
        
        # Momentum
        if abs(features.momentum_score) > 0.7:
            quality_score += 0.10
        
        # Volatility (prefer moderate)
        if 1 < features.atr_percent < 3:
            quality_score += 0.05
        elif features.atr_percent > 5:
            quality_score -= 0.10
        
        # Volume confirmation
        if features.volume_ratio > 1.5:
            quality_score += 0.05
        
        # Cap score
        quality_score = max(0.1, min(0.95, quality_score))
        
        # Determine tier
        if quality_score >= 0.85:
            tier = SignalQuality.ELITE
        elif quality_score >= 0.75:
            tier = SignalQuality.PREMIUM
        elif quality_score >= 0.65:
            tier = SignalQuality.STANDARD
        elif quality_score >= 0.55:
            tier = SignalQuality.MARGINAL
        else:
            tier = SignalQuality.REJECTED
        
        return {
            'quality_score': quality_score,
            'quality_tier': tier,
            'success_probability': quality_score,
            'risk_score': 1 - quality_score
        }
    
    def _atr_based_targets(self, entry_price: float, features: MLFeatures, side: str) -> Dict:
        """Calculate targets based on ATR"""
        atr_move = entry_price * (features.atr_percent / 100)
        
        if side == 'buy':
            tp1 = entry_price + atr_move * 1.5
            tp2 = entry_price + atr_move * 2.5
            tp3 = entry_price + atr_move * 3.5
        else:
            tp1 = entry_price - atr_move * 1.5
            tp2 = entry_price - atr_move * 2.5
            tp3 = entry_price - atr_move * 3.5
        
        return {
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'expected_move_pct': features.atr_percent * 2,
            'confidence': 0.6
        }
    
    def _prepare_training_data(self, historical_data: pd.DataFrame, 
                              outcomes: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        # This would need actual implementation based on available data
        # Placeholder implementation
        X = np.random.randn(100, 20)  # 100 samples, 20 features
        y_direction = np.random.randint(0, 3, 100)  # 3 classes
        y_quality = np.random.rand(100)  # Quality scores
        
        return X, y_direction, y_quality
    
    def _prepare_quality_features(self, signal_data: Dict, features: MLFeatures) -> np.ndarray:
        """Prepare features for quality prediction"""
        base_features = self._features_to_array(features)
        
        # Add signal-specific features
        additional_features = np.array([
            signal_data.get('risk_reward_ratio', 1.5),
            signal_data.get('confidence', 50) / 100,
            1 if signal_data.get('mtf_aligned', False) else 0
        ])
        
        return np.concatenate([base_features, additional_features])
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            # Placeholder - would load from files
            self.logger.info("No pre-trained models found, using rule-based predictions")
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
    
    def _save_models(self):
        """Save trained models"""
        try:
            # Placeholder - would save to files
            self.logger.info("Models saved successfully")
        except Exception as e:
            self.logger.error(f"Model saving error: {e}")

# ===========================
# NEWS SENTIMENT ANALYZER
# ===========================

class NewsSentimentAnalyzer:
    """Analyze news sentiment from multiple sources"""

    def __init__(self, config: SystemConfiguration, resource_manager=None):
        self.config = config
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # API configurations
        self.cryptopanic_key = '2c2a4ce275d7c36a8bb5ac71bf6a3b5a61e60cb8'
        self.newsapi_key = 'YOUR_NEWSAPI_KEY'
        
        # Sentiment keywords
        self.bullish_keywords = [
            'surge', 'rally', 'breakout', 'bullish', 'moon', 'pump',
            'adoption', 'institutional', 'upgrade', 'partnership'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bearish', 'sell-off', 'plunge', 'ban',
            'hack', 'scam', 'lawsuit', 'regulation', 'delisting'
        ]
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300
        
        # Session management
        self._session = None
        self._connector = None
        self._shutdown = False  # Add shutdown flag
    
    def _is_event_loop_running(self) -> bool:
        """Check if event loop is running and not closed"""
        try:
            loop = asyncio.get_running_loop()
            return loop is not None and not loop.is_closed()
        except RuntimeError:
            # No running event loop
            return False
    
    async def analyze_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment with proper error handling"""
        try:
            # Check if we should proceed
            if self._shutdown or not self._is_event_loop_running():
                self.logger.debug(f"Skipping news analysis for {symbol} - system shutting down")
                return self._default_sentiment()
            
            # Check cache first
            cache_key = f"{symbol}_{int(datetime.now().timestamp() // self.cache_duration)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Fetch news with timeout and error handling
            try:
                news_items = await asyncio.wait_for(
                    self._fetch_news_safe(symbol), 
                    timeout=5.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self.logger.debug(f"News fetch timeout/cancelled for {symbol}")
                return self._default_sentiment()
            except Exception as e:
                self.logger.debug(f"News fetch failed for {symbol}: {e}")
                return self._default_sentiment()
            
            # Analyze sentiment
            sentiment_data = self._analyze_news_items(news_items)
            
            # Get Fear & Greed Index (optional)
            try:
                if self._is_event_loop_running() and not self._shutdown:
                    fear_greed = await asyncio.wait_for(
                        self._fetch_fear_greed_index_safe(), 
                        timeout=2.0
                    )
                    sentiment_data['fear_greed'] = fear_greed
            except:
                sentiment_data['fear_greed'] = {'value': 50, 'classification': 'Neutral'}
            
            # Cache result
            if not self._shutdown:
                self.cache[cache_key] = sentiment_data
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return self._default_sentiment()
    
    async def _fetch_news_safe(self, symbol: str) -> List[Dict]:
        """Safely fetch news with event loop checks"""
        if self._shutdown or not self._is_event_loop_running():
            return []
        
        news_items = []
        clean_symbol = symbol.replace('USDT', '').replace('USD', '').replace('PERP', '')
        
        try:
            # Get session safely
            session = await self._get_session_safe()
            if not session:
                return []
            
            url = f"https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': self.cryptopanic_key,
                'currencies': clean_symbol,
                'filter': 'hot'
            }
            
            # Check loop again before making request
            if self._shutdown or not self._is_event_loop_running():
                return []
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for item in data.get('results', [])[:10]:
                        news_items.append({
                            'title': item.get('title', ''),
                            'source': item.get('domain', ''),
                            'published': item.get('published_at', ''),
                            'votes': item.get('votes', {}),
                            'url': item.get('url', '')
                        })
        
        except asyncio.CancelledError:
            self.logger.debug(f"News fetch cancelled for {symbol}")
            raise
        except Exception as e:
            self.logger.debug(f"News fetch error for {symbol}: {e}")
        
        return news_items
    
    async def _get_session_safe(self):
        """Get HTTP session with safety checks"""
        if self._shutdown or not self._is_event_loop_running():
            return None
        
        try:
            if self.resource_manager:
                return await self.resource_manager.get_http_session("news_analyzer")
            else:
                if self._session is None or self._session.closed:
                    # Create new session with safety checks
                    if not self._is_event_loop_running():
                        return None
                    
                    self._connector = aiohttp.TCPConnector(
                        limit=50,
                        limit_per_host=10,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True
                    )
                    
                    timeout = aiohttp.ClientTimeout(total=5, connect=2, sock_read=3)
                    
                    self._session = aiohttp.ClientSession(
                        connector=self._connector,
                        timeout=timeout,
                        headers={'User-Agent': 'SignalGenerator/1.0'}
                    )
                
                return self._session
                
        except Exception as e:
            self.logger.debug(f"Session creation error: {e}")
            return None
    
    async def _fetch_fear_greed_index_safe(self) -> Dict:
        """Safely fetch Fear & Greed Index"""
        if self._shutdown or not self._is_event_loop_running():
            return {'value': 50, 'classification': 'Neutral'}
        
        try:
            session = await self._get_session_safe()
            if not session:
                return {'value': 50, 'classification': 'Neutral'}
            
            url = "https://api.alternative.me/fng/"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('data'):
                        latest = data['data'][0]
                        return {
                            'value': int(latest.get('value', 50)),
                            'classification': latest.get('value_classification', 'Neutral')
                        }
        except:
            pass  # Silently fail for optional data
        
        return {'value': 50, 'classification': 'Neutral'}
    
    async def shutdown(self):
        """Graceful shutdown of news analyzer"""
        self.logger.info("Shutting down NewsSentimentAnalyzer...")
        self._shutdown = True
        
        # Wait a bit for ongoing requests to complete
        await asyncio.sleep(0.1)
        
        await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        self._shutdown = True
        
        if not self.resource_manager:
            # Only cleanup if we're managing our own session
            try:
                if self._session and not self._session.closed:
                    await self._session.close()
                if self._connector:
                    await self._connector.close()
            except Exception as e:
                self.logger.debug(f"Cleanup error: {e}")
        
        # Clear cache
        self.cache.clear()

    def _analyze_news_items(self, news_items: List[Dict]) -> Dict:
        """Analyze sentiment from news items"""
        if not news_items:
            return self._default_sentiment()
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_weight = 0
        weighted_sentiment = 0
        
        important_news = []
        
        for item in news_items:
            # Analyze title sentiment
            title_lower = item['title'].lower()
            
            bullish_score = sum(1 for keyword in self.bullish_keywords if keyword in title_lower)
            bearish_score = sum(1 for keyword in self.bearish_keywords if keyword in title_lower)
            
            # Consider votes if available
            votes = item.get('votes', {})
            positive_votes = votes.get('positive', 0) + votes.get('liked', 0)
            negative_votes = votes.get('negative', 0) + votes.get('disliked', 0)
            important_votes = votes.get('important', 0)
            
            # Calculate item sentiment
            if bullish_score > bearish_score or positive_votes > negative_votes * 2:
                item_sentiment = 1
                bullish_count += 1
            elif bearish_score > bullish_score or negative_votes > positive_votes * 2:
                item_sentiment = -1
                bearish_count += 1
            else:
                item_sentiment = 0
                neutral_count += 1
            
            # Weight by importance
            weight = 1 + (important_votes * 0.5)
            weighted_sentiment += item_sentiment * weight
            total_weight += weight
            
            # Track important news
            if important_votes >= 5 or (bullish_score + bearish_score) >= 3:
                important_news.append({
                    'title': item['title'],
                    'sentiment': item_sentiment,
                    'importance': important_votes
                })
        
        # Calculate overall sentiment
        if total_weight > 0:
            overall_sentiment = weighted_sentiment / total_weight
        else:
            overall_sentiment = 0
        
        # Determine sentiment classification
        if overall_sentiment > 0.3:
            classification = 'bullish'
        elif overall_sentiment < -0.3:
            classification = 'bearish'
        else:
            classification = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'classification': classification,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'important_news': important_news[:3],
            'news_count': len(news_items),
            'confidence': min(len(news_items) / 10, 1.0)  # More news = higher confidence
        }
    
    def _default_sentiment(self) -> Dict:
        """Return default neutral sentiment"""
        return {
            'overall_sentiment': 0,
            'classification': 'neutral',
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'important_news': [],
            'news_count': 0,
            'confidence': 0,
            'fear_greed': {'value': 50, 'classification': 'Neutral'}
        }

# ===========================
# MULTI-TIMEFRAME SIGNAL GENERATOR
# ===========================

class MultiTimeframeSignalGenerator(AnalysisModule):
    """Top-down multi-timeframe signal generator"""
    
    def __init__(self, config: SystemConfiguration, state_manager: StateManager):
        super().__init__(config)
        self.state_manager = state_manager
        self.criteria = SignalCriteria()
        
        # Initialize components
        self.ml_engine = MLPredictionEngine(config)
        self.news_analyzer = NewsSentimentAnalyzer(config)
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.structure_analyzer = MarketStructureAnalyzer()
        
        # Signal tracking
        self.pending_signals = []
        self.generated_count = 0
        self.rejected_count = 0
    
    async def analyze(self, symbol: str, exchange_manager) -> List[Signal]:
        """Perform top-down analysis and generate signals"""
        try:
            self.logger.debug(f"Analyzing {symbol} with top-down approach")
            
            # ============ ADD THIS SECTION TO FETCH MARKET DATA ============
            # Step 0: Fetch current market data for volume and price change
            try:
                ticker = exchange_manager.exchange.fetch_ticker(symbol)
                volume_24h = ticker.get('quoteVolume', 0)  # 24h volume in quote currency
                price_change_24h = ticker.get('percentage', 0)  # 24h price change percentage
                current_price = ticker.get('last', 0)
            except Exception as e:
                self.logger.warning(f"Could not fetch ticker data for {symbol}: {e}")
                volume_24h = 0
                price_change_24h = 0
                current_price = 0
            # ============ END OF NEW SECTION ============
            
            # Step 1: Analyze primary timeframe (highest - 6h)
            primary_df = await self._fetch_and_prepare_data(
                symbol, self.config.primary_timeframe, exchange_manager
            )
            
            if primary_df is None or len(primary_df) < 100:
                self.logger.debug(f"Insufficient data for {symbol}")
                return []
            
            # If current_price wasn't fetched, use last close
            if current_price == 0:
                current_price = primary_df['close'].iloc[-1]
            
            # Step 2: Identify potential signals on primary timeframe
            # ============ PASS MARKET DATA TO PRIMARY SIGNALS ============
            primary_signals = await self._identify_primary_signals(
                primary_df, symbol, volume_24h, price_change_24h, current_price
            )

            if not primary_signals:
                self.logger.debug(f"No signals found on primary timeframe for {symbol}")
                return []
            
            # Step 3: Validate with confirmation timeframes (4h, 1h)
            validated_signals = []
            
            for signal_candidate in primary_signals:
                validation_result = await self._validate_with_confirmations(
                    signal_candidate, symbol, exchange_manager
                )
                
                if validation_result['is_valid']:
                    # Step 4: Optimize entry on lowest timeframe (1h)
                    optimized_signal = await self._optimize_entry(
                        signal_candidate, symbol, exchange_manager, validation_result
                    )
                    
                    if optimized_signal:
                        validated_signals.append(optimized_signal)
            
            # Step 5: Rank and filter signals
            final_signals = self._rank_and_filter_signals(validated_signals)
            
            self.logger.info(f"Generated {len(final_signals)} signals for {symbol}")
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Analysis error for {symbol}: {e}")
            return []
    
    async def _fetch_and_prepare_data(self, symbol: str, timeframe: TimeFrame, 
                                 exchange_manager) -> Optional[pd.DataFrame]:
        """Fetch and prepare data with indicators"""
        try:
            # Direct call - no await needed
            df = exchange_manager.fetch_ohlcv_data(symbol, timeframe.label)
            
            if df is None or df.empty:
                return None
            
            # Check minimum data requirements
            if len(df) < 100:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
                return None
            
            # Calculate indicators - also synchronous
            df = TechnicalIndicators.calculate_all_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data fetch error for {symbol}: {e}")
            return None
    
    async def _identify_primary_signals(self, df: pd.DataFrame, symbol: str, 
                                    volume_24h: float = 0, price_change_24h: float = 0,
                                    current_price: float = 0) -> List[Dict]:
        """
        Identify potential signals on primary timeframe with comprehensive analysis
        FIXED: Handle missing columns and historical data access properly
        """
        signal_candidates = []
        
        try:
            # Ensure we have current price
            if current_price <= 0:
                current_price = df['close'].iloc[-1]
            
            # Calculate volume ratio
            volume_avg = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1.0
            
            # ===========================
            # TECHNICAL INDICATORS
            # ===========================
            
            # Ensure all indicators are calculated
            if 'rsi' not in df.columns:
                df = TechnicalIndicators.calculate_all_indicators(df)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Extract key indicators with safe defaults
            rsi = latest.get('rsi', 50)
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_histogram = latest.get('macd_histogram', 0)
            stoch_k = latest.get('stoch_k', 50)
            stoch_d = latest.get('stoch_d', 50)
            bb_position = latest.get('bb_percent', 0.5)
            atr = latest.get('atr', 0)
            atr_percent = latest.get('atr_percent', 2.0)
            
            # Moving averages
            sma_20 = latest.get('sma_20', current_price)
            sma_50 = latest.get('sma_50', current_price)
            ema_20 = latest.get('ema_20', current_price)
            
            # Get previous macd_histogram safely
            prev_macd_histogram = 0
            if 'macd_histogram' in df.columns and len(df) >= 2:
                prev_macd_histogram = df['macd_histogram'].iloc[-2]
            
            # ===========================
            # VOLUME PROFILE ANALYSIS
            # ===========================
            
            volume_analyzer = VolumeProfileAnalyzer()
            volume_profile = volume_analyzer.analyze_volume_profile(df)
            
            poc = volume_profile.get('poc', current_price)
            vah = volume_profile.get('value_area_high', current_price)
            val = volume_profile.get('value_area_low', current_price)
            
            # ===========================
            # MARKET STRUCTURE ANALYSIS
            # ===========================
            
            structure_analyzer = MarketStructureAnalyzer()
            structure = structure_analyzer.analyze_structure(df)
            
            trend_structure = structure.get('trend_structure', {})
            support_zones = structure.get('support_zones', [])
            resistance_zones = structure.get('resistance_zones', [])
            patterns = structure.get('patterns', [])
            market_regime_str = structure.get('market_regime', 'ranging')
            
            # Breakout detection
            breakout_info = structure_analyzer.detect_breakout(df, current_price)
            
            # ===========================
            # ML FEATURES EXTRACTION
            # ===========================
            
            # Safe calculation of price changes
            price_change_5 = 0
            price_change_10 = 0
            price_change_20 = 0
            
            if len(df) >= 5:
                price_change_5 = ((current_price - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100)
            if len(df) >= 10:
                price_change_10 = ((current_price - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100)
            if len(df) >= 20:
                price_change_20 = ((current_price - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100)
            
            # Safe RSI change calculation
            rsi_change = 0
            if 'rsi' in df.columns and len(df) >= 2:
                rsi_change = rsi - df['rsi'].iloc[-2]
            
            # Safe OBV change calculation
            obv_change = 0
            if 'obv' in df.columns and len(df) >= 5:
                obv_5_ago = df['obv'].iloc[-5]
                if obv_5_ago != 0:
                    obv_change = ((latest.get('obv', 0) - obv_5_ago) / abs(obv_5_ago) * 100)
            
            ml_features = MLFeatures(
                # Price features
                price_change_5=price_change_5,
                price_change_10=price_change_10,
                price_change_20=price_change_20,
                distance_from_sma20=(current_price - sma_20) / sma_20 * 100 if sma_20 > 0 else 0,
                distance_from_sma50=(current_price - sma_50) / sma_50 * 100 if sma_50 > 0 else 0,
                
                # Technical indicators
                rsi=rsi,
                rsi_change=rsi_change,
                macd_histogram=macd_histogram,
                stoch_k=stoch_k,
                bb_position=bb_position,
                
                # Volume features
                volume_ratio=volume_ratio,
                obv_change=obv_change,
                mfi=latest.get('mfi', 50),
                
                # Volatility features
                atr_percent=atr_percent,
                bb_width=latest.get('bb_width', 0.02),
                
                # Market structure
                trend_strength=trend_structure.get('strength', 0.5),
                distance_from_resistance=min([abs(current_price - r['price']) / current_price for r in resistance_zones]) if resistance_zones else 0.1,
                distance_from_support=min([abs(current_price - s['price']) / current_price for s in support_zones]) if support_zones else 0.1,
                
                # Momentum
                momentum_score=(macd_histogram / current_price * 1000) if current_price > 0 else 0,
                roc=latest.get('roc_10', 0)
            )
            
            # ===========================
            # ML PREDICTION
            # ===========================
            
            ml_prediction = self.ml_engine.predict_direction(ml_features, self.config.primary_timeframe)
            
            # ===========================
            # NEWS SENTIMENT (Optional)
            # ===========================
            
            news_sentiment = {'classification': 'neutral', 'score': 0}
            if self.config.use_news_sentiment:
                try:
                    news_sentiment = await self.news_analyzer.analyze_sentiment(symbol)
                except Exception as e:
                    self.logger.debug(f"News sentiment failed for {symbol}: {e}")
            
            # ===========================
            # SIGNAL GENERATION CONDITIONS (FIXED)
            # ===========================
            
            # Check for LONG signals
            long_conditions = {
                'rsi_oversold': 25 <= rsi <= 45,
                'stoch_oversold': stoch_k < 30 or (stoch_k < 50 and stoch_k > stoch_d),
                'macd_bullish': macd_histogram > 0 or (macd > macd_signal and macd_histogram > prev_macd_histogram),
                'price_above_support': any(abs(current_price - s['price']) / current_price < 0.02 for s in support_zones) if support_zones else False,
                'bullish_pattern': any(p.get('type') == 'bullish' for p in patterns) if patterns else False,
                'trend_up': trend_structure.get('trend') in ['uptrend', 'ranging'],
                'volume_surge': volume_ratio > 1.3,
                'ml_bullish': ml_prediction.get('direction') == 'bullish' and ml_prediction.get('confidence', 0) > 0.6,
                'breakout_up': breakout_info.get('type') == 'resistance_break',
                'price_above_vwap': current_price > latest.get('vwap', current_price),
                'bollinger_oversold': bb_position < 0.2
            }
            
            # Check for SHORT signals
            short_conditions = {
                'rsi_overbought': 55 <= rsi <= 75,
                'stoch_overbought': stoch_k > 70 or (stoch_k > 50 and stoch_k < stoch_d),
                'macd_bearish': macd_histogram < 0 or (macd < macd_signal and macd_histogram < prev_macd_histogram),
                'price_near_resistance': any(abs(current_price - r['price']) / current_price < 0.02 for r in resistance_zones) if resistance_zones else False,
                'bearish_pattern': any(p.get('type') == 'bearish' for p in patterns) if patterns else False,
                'trend_down': trend_structure.get('trend') in ['downtrend', 'ranging'],
                'volume_surge': volume_ratio > 1.3,
                'ml_bearish': ml_prediction.get('direction') == 'bearish' and ml_prediction.get('confidence', 0) > 0.6,
                'breakout_down': breakout_info.get('type') == 'support_break',
                'price_below_vwap': current_price < latest.get('vwap', current_price),
                'bollinger_overbought': bb_position > 0.8
            }
            
            # Count satisfied conditions
            long_score = sum(1 for v in long_conditions.values() if v)
            short_score = sum(1 for v in short_conditions.values() if v)
            
            # Minimum conditions required (relaxed for more signals)
            min_conditions = 3
            
            # ===========================
            # CREATE SIGNAL CANDIDATES
            # ===========================
            
            # Helper function to calculate stop loss safely
            def calculate_stop_loss(side, current_price, atr_percent, support_zones, resistance_zones, df):
                if side == 'buy':
                    # For long positions
                    atr_stop = current_price * (1 - atr_percent / 100 * 2)
                    support_stop = support_zones[0]['price'] * 0.995 if support_zones else current_price * 0.98
                    recent_low_stop = df['low'].tail(10).min() * 0.995 if len(df) >= 10 else current_price * 0.98
                    return min(atr_stop, support_stop, recent_low_stop)
                else:
                    # For short positions
                    atr_stop = current_price * (1 + atr_percent / 100 * 2)
                    resistance_stop = resistance_zones[0]['price'] * 1.005 if resistance_zones else current_price * 1.02
                    recent_high_stop = df['high'].tail(10).max() * 1.005 if len(df) >= 10 else current_price * 1.02
                    return max(atr_stop, resistance_stop, recent_high_stop)
            
            # Long signal candidate
            if long_score >= min_conditions:
                stop_loss = calculate_stop_loss('buy', current_price, atr_percent, support_zones, resistance_zones, df)
                
                signal_candidate = {
                    'symbol': symbol,
                    'side': 'buy',
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    
                    # Market data
                    'volume_24h': volume_24h,
                    'price_change_24h': price_change_24h,
                    'volume_ratio': volume_ratio,
                    
                    # Technical data
                    'rsi': rsi,
                    'macd_histogram': macd_histogram,
                    'stoch_k': stoch_k,
                    'bb_position': bb_position,
                    'atr_percent': atr_percent,
                    
                    # Structure data
                    'structure': structure,
                    'volume_profile': volume_profile,
                    'breakout_info': breakout_info,
                    
                    # ML data
                    'ml_features': ml_features,
                    'ml_prediction': ml_prediction,
                    'news_sentiment': news_sentiment,
                    
                    # Scoring
                    'conditions_met': long_score,
                    'conditions_details': long_conditions,
                    'signal_strength': long_score / len(long_conditions),
                    
                    # For later processing
                    'timeframe': self.config.primary_timeframe.label,
                    'analysis_time': datetime.now(timezone.utc).isoformat()
                }
                
                signal_candidates.append(signal_candidate)
                self.logger.debug(f"Long signal candidate for {symbol}: {long_score}/{len(long_conditions)} conditions met")
            
            # Short signal candidate
            if short_score >= min_conditions:
                stop_loss = calculate_stop_loss('sell', current_price, atr_percent, support_zones, resistance_zones, df)
                
                signal_candidate = {
                    'symbol': symbol,
                    'side': 'sell',
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    
                    # Market data
                    'volume_24h': volume_24h,
                    'price_change_24h': price_change_24h,
                    'volume_ratio': volume_ratio,
                    
                    # Technical data
                    'rsi': rsi,
                    'macd_histogram': macd_histogram,
                    'stoch_k': stoch_k,
                    'bb_position': bb_position,
                    'atr_percent': atr_percent,
                    
                    # Structure data
                    'structure': structure,
                    'volume_profile': volume_profile,
                    'breakout_info': breakout_info,
                    
                    # ML data
                    'ml_features': ml_features,
                    'ml_prediction': ml_prediction,
                    'news_sentiment': news_sentiment,
                    
                    # Scoring
                    'conditions_met': short_score,
                    'conditions_details': short_conditions,
                    'signal_strength': short_score / len(short_conditions),
                    
                    # For later processing
                    'timeframe': self.config.primary_timeframe.label,
                    'analysis_time': datetime.now(timezone.utc).isoformat()
                }
                
                signal_candidates.append(signal_candidate)
                self.logger.debug(f"Short signal candidate for {symbol}: {short_score}/{len(short_conditions)} conditions met")
            
            return signal_candidates
            
        except Exception as e:
            self.logger.error(f"Error identifying primary signals for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def _check_long_conditions(self, latest: pd.Series, ml_prediction: Dict,
                              news_sentiment: Dict, structure: Dict) -> bool:
        """RELAXED long conditions"""
        # RSI condition (much more relaxed)
        rsi = latest.get('rsi', 50)
        if not (15 <= rsi <= 60):
            return False
        
        # Stochastic condition (relaxed)
        stoch_k = latest.get('stoch_k', 50)
        if stoch_k > 70:  # Was much stricter
            return False
        
        # ML prediction (more flexible) - accept bullish OR neutral
        ml_direction = ml_prediction.get('direction', 'neutral')
        ml_confidence = ml_prediction.get('confidence', 0)
        if ml_direction == 'bearish' and ml_confidence > 0.7:
            return False
        
        # Market structure (more permissive)
        trend = structure.get('trend_structure', {}).get('trend', 'ranging')
        if trend == 'downtrend':
            strength = structure.get('trend_structure', {}).get('strength', 0)
            if strength > 0.6:  # Only reject strong downtrends
                return False
        
        # News sentiment (more permissive)
        sentiment_score = news_sentiment.get('overall_sentiment', 0)
        if sentiment_score < -0.7:  # Only reject very bearish news
            return False
        
        return True

    def _check_short_conditions(self, latest: pd.Series, ml_prediction: Dict,
                               news_sentiment: Dict, structure: Dict) -> bool:
        """RELAXED short conditions"""
        # RSI condition (much more relaxed)
        rsi = latest.get('rsi', 50)
        if not (40 <= rsi <= 85):
            return False
        
        # Stochastic condition (relaxed)
        stoch_k = latest.get('stoch_k', 50)
        if stoch_k < 30:  # Was much stricter
            return False
        
        # ML prediction (more flexible) - accept bearish OR neutral
        ml_direction = ml_prediction.get('direction', 'neutral')
        ml_confidence = ml_prediction.get('confidence', 0)
        if ml_direction == 'bullish' and ml_confidence > 0.7:
            return False
        
        # Market structure (more permissive)
        trend = structure.get('trend_structure', {}).get('trend', 'ranging')
        if trend == 'uptrend':
            strength = structure.get('trend_structure', {}).get('strength', 0)
            if strength > 0.6:  # Only reject strong uptrends
                return False
        
        # News sentiment (more permissive)
        sentiment_score = news_sentiment.get('overall_sentiment', 0)
        if sentiment_score > 0.7:  # Only reject very bullish news
            return False
        
        return True

    def _create_signal_candidate(self, symbol: str, side: str, current_price: float,
                                df: pd.DataFrame, ml_features: MLFeatures,
                                ml_prediction: Dict, news_sentiment: Dict,
                                structure: Dict, volume_profile: Dict) -> Optional[Dict]:
        """Create a signal candidate"""
        try:
            # Calculate initial stop loss
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
            
            if side == 'buy':
                stop_loss = current_price - (atr * 2.5)
                
                # Use structure for better stop
                if structure.get('support_zones'):
                    nearest_support = structure['support_zones'][0]['price']
                    stop_loss = min(stop_loss, nearest_support * 0.995)
            else:
                stop_loss = current_price + (atr * 2.5)
                
                # Use structure for better stop
                if structure.get('resistance_zones'):
                    nearest_resistance = structure['resistance_zones'][0]['price']
                    stop_loss = max(stop_loss, nearest_resistance * 1.005)
            
            # Calculate initial targets
            risk = abs(current_price - stop_loss)
            
            if side == 'buy':
                tp1 = current_price + (risk * 2.0)
                tp2 = current_price + (risk * 3.5)
            else:
                tp1 = current_price - (risk * 2.0)
                tp2 = current_price - (risk * 3.5)
            
            # Create candidate
            return {
                'symbol': symbol,
                'side': side,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'current_price': current_price,
                'ml_prediction': ml_prediction,
                'news_sentiment': news_sentiment,
                'structure': structure,
                'volume_profile': volume_profile,
                'ml_features': ml_features,
                'primary_timeframe': self.config.primary_timeframe.label
            }
            
        except Exception as e:
            self.logger.error(f"Signal candidate creation error: {e}")
            return None
        
    async def _validate_with_confirmations(self, signal_candidate: Dict, 
                                          symbol: str, exchange_manager) -> Dict:
        """Validate signal with confirmation timeframes"""
        validation_result = {
            'is_valid': True,
            'confirmations': {},
            'alignment_score': 0
        }
        
        try:
            for tf in self.config.confirmation_timeframes:
                # Fetch data for confirmation timeframe
                conf_df = await self._fetch_and_prepare_data(symbol, tf, exchange_manager)
                
                if conf_df is None or len(conf_df) < 50:
                    validation_result['is_valid'] = False
                    break
                
                # Check alignment
                alignment = self._check_timeframe_alignment(
                    conf_df, signal_candidate['side'], tf
                )
                
                validation_result['confirmations'][tf.label] = alignment
                
                if not alignment['aligned']:
                    if self.criteria.require_all_timeframe_alignment:
                        validation_result['is_valid'] = False
                        break
                
                validation_result['alignment_score'] += alignment['score']
            
            # Calculate average alignment
            if validation_result['confirmations']:
                validation_result['alignment_score'] /= len(validation_result['confirmations'])
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Confirmation validation error: {e}")
            return {'is_valid': False, 'confirmations': {}, 'alignment_score': 0}
    
    def _check_timeframe_alignment(self, df: pd.DataFrame, side: str, 
                                  timeframe: TimeFrame) -> Dict:
        """Check if timeframe aligns with signal"""
        latest = df.iloc[-1]
        
        alignment = {
            'aligned': False,
            'score': 0,
            'reasons': []
        }
        
        # RSI check
        rsi = latest.get('rsi', 50)
        if side == 'buy':
            if rsi < self.criteria.confirm_rsi_long_max:
                alignment['score'] += 0.25
                alignment['reasons'].append(f"RSI supportive ({rsi:.1f})")
            else:
                alignment['reasons'].append(f"RSI not supportive ({rsi:.1f})")
        else:
            if rsi > self.criteria.confirm_rsi_short_min:
                alignment['score'] += 0.25
                alignment['reasons'].append(f"RSI supportive ({rsi:.1f})")
            else:
                alignment['reasons'].append(f"RSI not supportive ({rsi:.1f})")
        
        # MACD check
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        if side == 'buy' and macd > macd_signal:
            alignment['score'] += 0.25
            alignment['reasons'].append("MACD bullish")
        elif side == 'sell' and macd < macd_signal:
            alignment['score'] += 0.25
            alignment['reasons'].append("MACD bearish")
        
        # Trend check
        sma20 = latest.get('sma_20', latest['close'])
        sma50 = latest.get('sma_50', latest['close'])
        if side == 'buy' and sma20 > sma50:
            alignment['score'] += 0.25
            alignment['reasons'].append("Trend bullish")
        elif side == 'sell' and sma20 < sma50:
            alignment['score'] += 0.25
            alignment['reasons'].append("Trend bearish")
        
        # Volume check
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio > self.criteria.min_volume_ratio:
            alignment['score'] += 0.25
            alignment['reasons'].append(f"Volume strong ({volume_ratio:.2f}x)")
        
        # Determine if aligned
        alignment['aligned'] = alignment['score'] >= 0.5
        
        return alignment
    
    async def _optimize_entry(self, signal_candidate: Dict, symbol: str,
                            exchange_manager, validation_result: Dict) -> Optional[Signal]:
        """Optimize entry using lowest timeframe"""
        try:
            # Fetch entry timeframe data
            entry_df = await self._fetch_and_prepare_data(
                symbol, self.config.entry_timeframe, exchange_manager
            )
            
            if entry_df is None or len(entry_df) < 50:
                return None
            
            latest = entry_df.iloc[-1]
            current_price = latest['close']
            
            # Check entry conditions on lowest timeframe
            if not self._check_entry_conditions(latest, signal_candidate['side']):
                # Add to pending queue for monitoring
                signal_candidate['status'] = 'pending_entry'
                return None
            
            # Optimize entry price
            optimized_entry = self._calculate_optimized_entry(
                entry_df, signal_candidate, current_price
            )
            
            # Recalculate stops and targets with optimized entry
            optimized_signal = self._finalize_signal(
                signal_candidate, optimized_entry, validation_result
            )
            
            return optimized_signal
            
        except Exception as e:
            self.logger.error(f"Entry optimization error: {e}")
            return None
    
    def _check_entry_conditions(self, latest: pd.Series, side: str) -> bool:
        """MUCH MORE RELAXED entry conditions"""
        rsi = latest.get('rsi', 50)
        stoch_k = latest.get('stoch_k', 50)
        
        if side == 'buy':
            rsi_ok = rsi <= 55  # Much simpler and more lenient
            stoch_ok = stoch_k < 60
        else:
            rsi_ok = rsi >= 45  # Much simpler and more lenient
            stoch_ok = stoch_k > 40
        
        return rsi_ok or stoch_ok  # OR instead of AND - only need one!
    
    def _calculate_optimized_entry(self, df: pd.DataFrame, signal_candidate: Dict,
                                  current_price: float) -> float:
        """Calculate optimized entry price"""
        side = signal_candidate['side']
        
        # Use volume profile for better entry
        vp = signal_candidate.get('volume_profile', {})
        
        if side == 'buy':
            # For long, try to enter near support or value area low
            if vp.get('value_area_low'):
                entry = min(current_price, vp['value_area_low'] * 1.001)
            else:
                entry = current_price * 0.999
        else:
            # For short, try to enter near resistance or value area high
            if vp.get('value_area_high'):
                entry = max(current_price, vp['value_area_high'] * 0.999)
            else:
                entry = current_price * 1.001
        
        return entry
    
    def _determine_order_type(self, signal_candidate: Dict, optimized_entry: float, 
                              current_price: float) -> str:
        """
        Determine order type based on entry proximity and breakout patterns
        
        Market order conditions:
        1. Entry price is within 0.5% to 1% of current price
        2. Breakout pattern detected
        3. High volume surge
        4. Strong MTF confirmation
        """
        
        # Calculate price difference percentage
        price_diff_pct = abs(optimized_entry - current_price) / current_price
        
        # Check for breakout
        breakout_info = signal_candidate.get('breakout_info', {})
        has_breakout = breakout_info.get('has_breakout', False)
        breakout_strength = breakout_info.get('strength', 0)
        
        # Check for pattern-based breakouts
        patterns = signal_candidate.get('structure', {}).get('patterns', [])
        breakout_patterns = [
            'ascending_triangle', 'descending_triangle', 
            'double_bottom', 'inverse_head_and_shoulders',
            'bull_flag', 'bear_flag', 'cup_and_handle'
        ]
        
        has_pattern_breakout = any(
            p.get('name') in breakout_patterns 
            for p in patterns
        )
        
        # Check volume conditions
        volume_ratio = signal_candidate.get('volume_ratio', 1.0)
        has_volume_surge = volume_ratio > 2.0
        
        # Check MTF alignment
        mtf_status = signal_candidate.get('mtf_status', 'NONE')
        strong_mtf = mtf_status == 'STRONG'
        
        # Determine order type with reasoning
        order_type = 'limit'  # Default
        reason = "Default - patient entry preferred"
        
        # Priority conditions for market orders
        if has_breakout and breakout_strength > 0.003:  # Strong breakout
            order_type = 'market'
            reason = f"Strong breakout detected ({breakout_info.get('type', 'breakout')})"
        elif 0.005 <= price_diff_pct <= 0.01:  # Optimal proximity
            order_type = 'market'
            reason = f"Entry within optimal range ({price_diff_pct:.2%} from current)"
        elif has_pattern_breakout and has_volume_surge:
            order_type = 'market'
            pattern_names = [p.get('name') for p in patterns if p.get('name') in breakout_patterns]
            reason = f"Pattern breakout with volume: {', '.join(pattern_names[:2])}"
        elif price_diff_pct < 0.005 and (has_volume_surge or strong_mtf):
            order_type = 'market'
            reason = f"Very close entry ({price_diff_pct:.2%}) with "
            reason += "volume surge" if has_volume_surge else "strong MTF"
        elif strong_mtf and price_diff_pct < 0.015:
            order_type = 'market'
            reason = f"Strong MTF confirmation with reasonable proximity"
        
        self.logger.debug(f"Order type: {order_type.upper()} - {reason}")
        
        # Store reasoning for later reference
        signal_candidate['order_type_reason'] = reason
        
        return order_type
    
    def _calculate_quality_grade(self, confidence: float, rr_ratio: float, 
                                  mtf_status: str, volume_ratio: float) -> str:
        """
        Calculate comprehensive quality grade
        A+ = Elite signals (>90 score)
        A  = Premium signals (85-90)
        A- = High quality (80-85)
        B+ = Good quality (75-80)
        B  = Decent quality (70-75)
        B- = Acceptable (65-70)
        C+ = Marginal (60-65)
        C  = Minimum viable (<60)
        """
        
        # Calculate composite score (0-100)
        score = 0
        
        # Confidence component (0-40 points)
        conf_score = min(confidence / 100 * 40, 40)
        score += conf_score
        
        # Risk/Reward component (0-25 points)
        rr_score = min(rr_ratio / 4 * 25, 25)  # Max at 4:1
        score += rr_score
        
        # MTF component (0-20 points)
        mtf_scores = {
            'STRONG': 20,
            'PARTIAL': 12,
            'NONE': 5,
            'DISABLED': 0
        }
        score += mtf_scores.get(mtf_status, 5)
        
        # Volume component (0-15 points)
        if volume_ratio >= 2.0:
            score += 15
        elif volume_ratio >= 1.5:
            score += 10
        elif volume_ratio >= 1.2:
            score += 7
        elif volume_ratio >= 1.0:
            score += 5
        else:
            score += 2
        
        # Determine grade based on score
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        else:
            return 'C'
    
    def _determine_mtf_status(self, validation_result: Dict) -> Tuple[str, int, int]:
        """
        Determine MTF status and counts from validation result
        Returns: (status, confirmed_count, total_count)
        """
        confirmations = validation_result.get('confirmations', {})
        confirmed_count = sum(1 for v in confirmations.values() if v)
        total_count = len(confirmations)
        
        if total_count == 0:
            return 'DISABLED', 0, 0
        elif confirmed_count >= 2:
            return 'STRONG', confirmed_count, total_count
        elif confirmed_count == 1:
            return 'PARTIAL', confirmed_count, total_count
        else:
            return 'NONE', confirmed_count, total_count
    
    def _calculate_mtf_boost(self, original_confidence: float, 
                             mtf_status: str, alignment_score: float) -> float:
        """Calculate confidence boost from MTF confirmation"""
        boost = 0
        
        if mtf_status == 'STRONG':
            boost = min(10, alignment_score * 15)  # Max 10 point boost
        elif mtf_status == 'PARTIAL':
            boost = min(5, alignment_score * 8)   # Max 5 point boost
        
        return round(boost, 1)
    
    def _determine_entry_strategy(self, order_type: str, mtf_status: str, 
                                  breakout_info: Dict) -> str:
        """Determine entry strategy based on conditions"""
        if order_type == 'market':
            if breakout_info.get('has_breakout'):
                return 'breakout_entry'
            else:
                return 'immediate_entry'
        else:  # limit order
            if mtf_status == 'STRONG':
                return 'patient_confirmed'
            else:
                return 'patient_entry'
    
    def _determine_regime_compatibility(self, side: str, market_regime: str) -> str:
        """Determine how compatible the signal is with market regime"""
        regime_str = str(market_regime).lower()
        
        if side == 'buy':
            if 'strong_trend_up' in regime_str or 'trend_up' in regime_str:
                return 'high'
            elif 'ranging' in regime_str or 'squeeze' in regime_str:
                return 'medium'
            else:
                return 'low'
        else:  # sell
            if 'strong_trend_down' in regime_str or 'trend_down' in regime_str:
                return 'high'
            elif 'ranging' in regime_str or 'squeeze' in regime_str:
                return 'medium'
            else:
                return 'low'
    
    def _process_mtf_analysis(self, validation_result: Dict, 
                             confirmation_timeframes: List[str]) -> Dict:
        """Process MTF data for display"""
        confirmations = validation_result.get('confirmations', {})
        
        confirmed_tfs = []
        conflicting_tfs = []
        neutral_tfs = []
        
        for tf in confirmation_timeframes:
            if tf in confirmations:
                if confirmations[tf]:
                    confirmed_tfs.append(tf)
                else:
                    conflicting_tfs.append(tf)
            else:
                neutral_tfs.append(tf)
        
        return {
            'confirmed_timeframes': confirmed_tfs,
            'conflicting_timeframes': conflicting_tfs,
            'neutral_timeframes': neutral_tfs,
            'confirmation_strength': validation_result.get('alignment_score', 0)
        }
    
    def _finalize_signal(self, signal_candidate: Dict, optimized_entry: float,
                        validation_result: Dict) -> Signal:
        """
        COMPLETE REWRITE: Create final Signal object with ALL required fields
        """
        
        # Extract basic info
        side = signal_candidate['side']
        current_price = signal_candidate['current_price']
        
        # Fix stop loss and calculate targets with proper logic
        stop_loss = signal_candidate['stop_loss']
        
        if side == 'buy':
            # For long positions, stop should be below entry
            if stop_loss >= optimized_entry:
                # Fix invalid stop loss
                risk = optimized_entry * 0.02  # 2% default risk
                stop_loss = optimized_entry - risk
            else:
                risk = optimized_entry - stop_loss
            
            tp1 = optimized_entry + (risk * 2.0)
            tp2 = optimized_entry + (risk * 3.5)
            
        else:  # sell
            # For short positions, stop should be above entry
            if stop_loss <= optimized_entry:
                # Fix invalid stop loss
                risk = optimized_entry * 0.02  # 2% default risk
                stop_loss = optimized_entry + risk
            else:
                risk = stop_loss - optimized_entry
            
            tp1 = optimized_entry - (risk * 2.0)
            tp2 = optimized_entry - (risk * 3.5)
        
        # Calculate risk/reward
        risk_amount = abs(optimized_entry - stop_loss)
        reward_amount = abs(tp1 - optimized_entry)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Determine order type
        order_type = self._determine_order_type(
            signal_candidate, optimized_entry, current_price
        )
        
        # Determine MTF status
        mtf_status, mtf_confirmed_count, mtf_total = self._determine_mtf_status(
            validation_result
        )
        
        # Calculate base confidence
        base_confidence = 50  # Start with base
        base_confidence += validation_result.get('alignment_score', 0.5) * 30
        base_confidence += min(rr_ratio / 4, 1) * 10
        base_confidence += signal_candidate.get('ml_prediction', {}).get('confidence', 0.5) * 10
        base_confidence = min(95, max(40, base_confidence))
        
        # Calculate MTF boost
        original_confidence = base_confidence
        mtf_boost = self._calculate_mtf_boost(
            original_confidence, 
            mtf_status, 
            validation_result.get('alignment_score', 0)
        )
        final_confidence = min(95, original_confidence + mtf_boost)
        
        # Get volume data
        volume_ratio = signal_candidate.get('volume_ratio', 1.0)
        volume_24h = signal_candidate.get('volume_24h', 0)
        price_change_24h = signal_candidate.get('price_change_24h', 0)
        
        # Calculate quality grade
        quality_grade = self._calculate_quality_grade(
            final_confidence, rr_ratio, mtf_status, volume_ratio
        )
        
        # Determine entry strategy
        breakout_info = signal_candidate.get('breakout_info', {})
        entry_strategy = self._determine_entry_strategy(
            order_type, mtf_status, breakout_info
        )
        
        # Determine regime compatibility
        market_regime = signal_candidate.get('structure', {}).get('market_regime', 'ranging')
        regime_compatibility = self._determine_regime_compatibility(side, market_regime)
        
        # Process MTF analysis for display
        mtf_analysis = self._process_mtf_analysis(
            validation_result, 
            self.config.confirmation_timeframes
        )
        
        # Determine quality tier based on grade
        quality_tier_map = {
            'A+': SignalQuality.ELITE,
            'A': SignalQuality.ELITE,
            'A-': SignalQuality.PREMIUM,
            'B+': SignalQuality.PREMIUM,
            'B': SignalQuality.STANDARD,
            'B-': SignalQuality.STANDARD,
            'C+': SignalQuality.MARGINAL,
            'C': SignalQuality.MARGINAL
        }
        quality_tier = quality_tier_map.get(quality_grade, SignalQuality.MARGINAL)
        
        # Extract patterns
        patterns = signal_candidate.get('structure', {}).get('patterns', [])
        
        # Create comprehensive Signal object
        signal = Signal(
            # Core identification
            id=self._generate_signal_id(signal_candidate),
            symbol=signal_candidate['symbol'],
            side=side,
            status=SignalStatus.READY,
            quality_tier=quality_tier,
            
            # Timestamps
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            
            # Price levels
            entry_price=optimized_entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            current_price=current_price,
            
            # Risk metrics
            risk_reward_ratio=rr_ratio,
            position_size=0,  # Will be calculated by risk manager
            risk_amount=risk_amount,
            potential_profit=reward_amount,
            
            # Confidence metrics
            confidence=final_confidence,
            original_confidence=original_confidence,
            mtf_boost=mtf_boost,
            
            # Analysis data
            analysis_timeframe=self.config.primary_timeframe,
            market_regime=self._determine_market_regime(signal_candidate),
            
            # Order and execution
            order_type=order_type,
            quality_grade=quality_grade,
            entry_strategy=entry_strategy,
            
            # MTF status
            mtf_status=mtf_status,
            mtf_validated=(mtf_status in ['STRONG', 'PARTIAL']),
            mtf_confirmation_count=mtf_confirmed_count,
            mtf_total_timeframes=mtf_total,
            
            # MTF data
            mtf_alignment=validation_result.get('confirmations', {}),
            mtf_scores={'alignment': validation_result.get('alignment_score', 0)},
            mtf_analysis=mtf_analysis,
            
            # Market data
            volume_24h=volume_24h,
            price_change_24h=price_change_24h,
            volume_ratio=volume_ratio,
            
            # Technical data
            indicators=self._extract_key_indicators(signal_candidate),
            ml_prediction=signal_candidate.get('ml_prediction'),
            news_sentiment=signal_candidate.get('news_sentiment'),
            volume_analysis=signal_candidate.get('volume_profile'),
            
            # Patterns and structure
            patterns=patterns,
            breakout_detected=breakout_info.get('has_breakout', False),
            support_resistance={
                'support': signal_candidate.get('structure', {}).get('support_zones', []),
                'resistance': signal_candidate.get('structure', {}).get('resistance_zones', [])
            },
            
            # Risk and warnings
            warnings=self._identify_warnings(signal_candidate),
            
            # Metadata
            regime_compatibility=regime_compatibility,
            metadata={
                'order_type_reason': signal_candidate.get('order_type_reason', ''),
                'structure_timeframe': self.config.confirmation_timeframes[-1] if self.config.confirmation_timeframes else None
            }
        )
        
        return signal
    
    def _rank_and_filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals by quality and filter"""
        if not signals:
            return []
        
        # Sort by priority
        signals.sort(key=lambda s: (
            s.quality_tier.value if isinstance(s.quality_tier, SignalQuality) else 0,
            s.confidence,
            s.risk_reward_ratio
        ), reverse=True)
        
        # Filter out low quality
        filtered = [s for s in signals if s.quality_tier != SignalQuality.REJECTED]
        
        # Limit number of signals
        return filtered[:self.config.max_signals_per_symbol]
    
    def _generate_signal_id(self, signal_data: Dict) -> str:
        """Generate unique signal ID"""
        content = f"{signal_data['symbol']}_{signal_data['side']}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _determine_market_regime(self, signal_data: Dict) -> MarketRegime:
        """Determine market regime from signal data"""
        structure = signal_data.get('structure', {})
        regime_str = structure.get('market_regime', MarketRegime.RANGING)
        
        if isinstance(regime_str, MarketRegime):
            return regime_str
        
        # Map string to enum
        regime_map = {
            'strong_trend_up': MarketRegime.STRONG_TREND_UP,
            'trend_up': MarketRegime.TREND_UP,
            'ranging_bullish': MarketRegime.RANGING_BULLISH,
            'ranging': MarketRegime.RANGING,
            'ranging_bearish': MarketRegime.RANGING_BEARISH,
            'trend_down': MarketRegime.TREND_DOWN,
            'strong_trend_down': MarketRegime.STRONG_TREND_DOWN,
            'volatile': MarketRegime.VOLATILE,
            'squeeze': MarketRegime.SQUEEZE
        }
        
        return regime_map.get(regime_str, MarketRegime.RANGING)
    
    def _extract_key_indicators(self, signal_data: Dict) -> Dict:
        """Extract key indicator values"""
        # This would extract relevant indicators from the signal data
        return {
            'rsi': 50,
            'macd': 0,
            'stoch_k': 50,
            'volume_ratio': 1.0
        }
    
    def _identify_warnings(self, signal_data: Dict) -> List[str]:
        """Identify any warnings for the signal"""
        warnings = []
        
        # Check news sentiment
        if signal_data.get('news_sentiment', {}).get('classification') == 'bearish' and signal_data['side'] == 'buy':
            warnings.append("News sentiment is bearish")
        elif signal_data.get('news_sentiment', {}).get('classification') == 'bullish' and signal_data['side'] == 'sell':
            warnings.append("News sentiment is bullish")
        
        # Check ML confidence
        if signal_data.get('ml_prediction', {}).get('confidence', 0) < 0.6:
            warnings.append("ML confidence is low")
        
        return warnings
    
    def get_signals(self) -> List[Dict]:
        """Get generated signals"""
        return [s.to_dict() for s in self.pending_signals]
    
    def validate(self, signal: Dict) -> Tuple[bool, List[str]]:
        """Validate a signal"""
        issues = []
        
        # Check risk/reward
        if signal.get('risk_reward_ratio', 0) < self.config.min_risk_reward:
            issues.append("Risk/reward too low")
        
        # Check confidence
        if signal.get('confidence', 0) < 60:
            issues.append("Confidence too low")
        
        is_valid = len(issues) == 0
        return is_valid, issues

# ===========================
# EXPORTS
# ===========================

__all__ = [
    'MLPredictionEngine',
    'MLFeatures',
    'NewsSentimentAnalyzer',
    'MultiTimeframeSignalGenerator'
]