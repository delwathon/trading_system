"""
Enhanced MTF Validator - Strict directional bias enforcement
Prevents counter-trend signals by properly analyzing higher timeframes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MTFValidationResult:
    """MTF validation result with detailed analysis"""
    is_valid: bool
    mtf_status: str  # 'STRONG', 'PARTIAL', 'NONE'
    dominant_bias: str  # 'bullish', 'bearish', 'neutral'
    alignment_score: float
    rejection_reason: Optional[str] = None
    tp1_reachability_score: float = 0.0
    structure_analysis: Dict = None

class StrictMTFValidator:
    """Strict MTF validator that prevents counter-trend trades"""
    
    def __init__(self, confirmation_timeframes: List[str]):
        self.confirmation_timeframes = confirmation_timeframes
        self.min_alignment_score = 0.6  # Minimum 60% alignment required
        self.min_tp1_reachability = 0.7  # 70% confidence for TP1
    
    def validate_signal_with_mtf(self, signal_side: str, 
                                primary_df: pd.DataFrame,
                                mtf_data: Dict[str, pd.DataFrame],
                                current_price: float,
                                tp1_price: float) -> MTFValidationResult:
        """
        Strictly validate signal against higher timeframes
        
        Returns:
            MTFValidationResult with validation details
        """
        
        # Analyze each timeframe
        timeframe_analysis = {}
        alignment_scores = []
        
        for tf in self.confirmation_timeframes:
            if tf not in mtf_data or mtf_data[tf].empty:
                continue
                
            tf_analysis = self._analyze_timeframe_structure(
                mtf_data[tf], signal_side, current_price
            )
            timeframe_analysis[tf] = tf_analysis
            alignment_scores.append(tf_analysis['alignment_score'])
        
        # Calculate overall alignment
        if not alignment_scores:
            return MTFValidationResult(
                is_valid=False,
                mtf_status='NONE',
                dominant_bias='neutral',
                alignment_score=0.0,
                rejection_reason='No MTF data available'
            )
        
        overall_alignment = np.mean(alignment_scores)
        dominant_bias = self._determine_dominant_bias(timeframe_analysis)
        
        # Check TP1 reachability
        tp1_reachability = self._assess_tp1_reachability(
            signal_side, current_price, tp1_price, 
            primary_df, mtf_data, dominant_bias
        )
        
        # Determine MTF status
        if overall_alignment >= 0.8 and tp1_reachability >= self.min_tp1_reachability:
            mtf_status = 'STRONG'
            is_valid = True
            rejection_reason = None
        elif overall_alignment >= 0.6 and tp1_reachability >= 0.6:
            mtf_status = 'PARTIAL'
            is_valid = True
            rejection_reason = None
        else:
            mtf_status = 'NONE'
            is_valid = False
            rejection_reason = self._get_rejection_reason(
                overall_alignment, tp1_reachability, dominant_bias, signal_side
            )
        
        return MTFValidationResult(
            is_valid=is_valid,
            mtf_status=mtf_status,
            dominant_bias=dominant_bias,
            alignment_score=overall_alignment,
            rejection_reason=rejection_reason,
            tp1_reachability_score=tp1_reachability,
            structure_analysis=timeframe_analysis
        )
    
    def _analyze_timeframe_structure(self, df: pd.DataFrame, 
                                    signal_side: str, 
                                    current_price: float) -> Dict:
        """Analyze structure on a specific timeframe"""
        
        if len(df) < 50:
            return {'alignment_score': 0.0, 'bias': 'neutral', 'strength': 0.0}
        
        latest = df.iloc[-1]
        
        # Multiple trend confirmation methods
        # 1. Moving Average Analysis
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
        
        ma_bullish_count = 0
        if current_price > sma_20: ma_bullish_count += 1
        if current_price > sma_50: ma_bullish_count += 1
        if sma_20 > sma_50: ma_bullish_count += 1
        if ema_12 > ema_26: ma_bullish_count += 1
        
        ma_bias = 'bullish' if ma_bullish_count >= 3 else 'bearish' if ma_bullish_count <= 1 else 'neutral'
        
        # 2. Higher Highs/Lower Lows Structure
        recent_highs = df['high'].tail(20)
        recent_lows = df['low'].tail(20)
        
        hh_ll_pattern = self._identify_hh_ll_pattern(recent_highs, recent_lows)
        
        # 3. Momentum Analysis
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        rsi = latest.get('rsi', 50)
        
        momentum_bullish = macd > macd_signal and rsi > 45 and rsi < 75
        momentum_bearish = macd < macd_signal and rsi < 55 and rsi > 25
        
        # 4. Volume Trend
        volume_trend = df['volume'].tail(10).mean() > df['volume'].tail(30).mean()
        
        # 5. Key Level Analysis
        resistance = df['high'].tail(50).max()
        support = df['low'].tail(50).min()
        price_position = (current_price - support) / (resistance - support)
        
        # Determine bias
        bullish_signals = 0
        bearish_signals = 0
        
        if ma_bias == 'bullish': bullish_signals += 2
        elif ma_bias == 'bearish': bearish_signals += 2
        
        if hh_ll_pattern == 'bullish': bullish_signals += 2
        elif hh_ll_pattern == 'bearish': bearish_signals += 2
        
        if momentum_bullish: bullish_signals += 1
        elif momentum_bearish: bearish_signals += 1
        
        if price_position > 0.6: bullish_signals += 1
        elif price_position < 0.4: bearish_signals += 1
        
        # Final bias determination
        if bullish_signals >= bearish_signals + 2:
            timeframe_bias = 'bullish'
        elif bearish_signals >= bullish_signals + 2:
            timeframe_bias = 'bearish'
        else:
            timeframe_bias = 'neutral'
        
        # Calculate alignment with signal
        alignment_score = 0.0
        if signal_side == 'buy' and timeframe_bias == 'bullish':
            alignment_score = min(1.0, bullish_signals / 6)
        elif signal_side == 'sell' and timeframe_bias == 'bearish':
            alignment_score = min(1.0, bearish_signals / 6)
        elif timeframe_bias == 'neutral':
            alignment_score = 0.3  # Weak alignment for neutral
        else:
            alignment_score = 0.0  # Counter-trend
        
        return {
            'alignment_score': alignment_score,
            'bias': timeframe_bias,
            'strength': max(bullish_signals, bearish_signals) / 6,
            'ma_bias': ma_bias,
            'structure_pattern': hh_ll_pattern,
            'momentum_aligned': momentum_bullish if signal_side == 'buy' else momentum_bearish,
            'price_position': price_position
        }
    
    def _identify_hh_ll_pattern(self, highs: pd.Series, lows: pd.Series) -> str:
        """Identify Higher High/Lower Low pattern"""
        
        if len(highs) < 10:
            return 'neutral'
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs) - 2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                swing_highs.append((i, highs.iloc[i]))
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                swing_lows.append((i, lows.iloc[i]))
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'neutral'
        
        # Check last two swing highs and lows
        last_two_highs = swing_highs[-2:]
        last_two_lows = swing_lows[-2:]
        
        higher_high = last_two_highs[1][1] > last_two_highs[0][1]
        higher_low = last_two_lows[1][1] > last_two_lows[0][1]
        lower_high = last_two_highs[1][1] < last_two_highs[0][1]
        lower_low = last_two_lows[1][1] < last_two_lows[0][1]
        
        if higher_high and higher_low:
            return 'bullish'
        elif lower_high and lower_low:
            return 'bearish'
        else:
            return 'neutral'
    
    def _determine_dominant_bias(self, timeframe_analysis: Dict) -> str:
        """Determine dominant market bias from all timeframes"""
        
        biases = []
        weights = []
        
        # Higher timeframes get more weight
        weight_map = {'2h': 1.0, '4h': 1.5, '6h': 2.0}
        
        for tf, analysis in timeframe_analysis.items():
            biases.append(analysis['bias'])
            weights.append(weight_map.get(tf, 1.0) * analysis['strength'])
        
        if not biases:
            return 'neutral'
        
        # Weighted voting
        bullish_weight = sum(w for b, w in zip(biases, weights) if b == 'bullish')
        bearish_weight = sum(w for b, w in zip(biases, weights) if b == 'bearish')
        
        if bullish_weight > bearish_weight * 1.3:
            return 'bullish'
        elif bearish_weight > bullish_weight * 1.3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_tp1_reachability(self, signal_side: str, current_price: float, 
                                tp1_price: float, primary_df: pd.DataFrame,
                                mtf_data: Dict, dominant_bias: str) -> float:
        """Assess probability of reaching TP1"""
        
        tp1_distance = abs(tp1_price - current_price) / current_price
        
        # Check if TP1 is realistic based on ATR
        if 'atr' in primary_df.columns:
            atr = primary_df['atr'].iloc[-1]
            atr_pct = atr / current_price
            
            # TP1 should be within 2-3 ATR moves
            if tp1_distance > atr_pct * 3:
                return 0.3  # Too ambitious
        
        # Check resistance/support levels between entry and TP1
        obstacles = 0
        for tf, df in mtf_data.items():
            if df.empty:
                continue
                
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            
            if signal_side == 'buy':
                # Check for resistance before TP1
                if current_price < recent_high < tp1_price:
                    obstacles += 1
            else:
                # Check for support before TP1
                if tp1_price < recent_low < current_price:
                    obstacles += 1
        
        # Calculate reachability score
        base_score = 0.8
        
        # Alignment bonus
        if (signal_side == 'buy' and dominant_bias == 'bullish') or \
           (signal_side == 'sell' and dominant_bias == 'bearish'):
            base_score += 0.2
        elif dominant_bias == 'neutral':
            base_score -= 0.1
        else:
            base_score -= 0.3
        
        # Obstacle penalty
        base_score -= obstacles * 0.15
        
        # Distance penalty for unrealistic targets
        if tp1_distance > 0.05:  # More than 5%
            base_score -= 0.2
        elif tp1_distance > 0.03:  # More than 3%
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _get_rejection_reason(self, alignment: float, tp1_reach: float, 
                             dominant_bias: str, signal_side: str) -> str:
        """Generate detailed rejection reason"""
        
        reasons = []
        
        if alignment < self.min_alignment_score:
            reasons.append(f"Poor MTF alignment ({alignment:.1%})")
        
        if tp1_reach < self.min_tp1_reachability:
            reasons.append(f"Low TP1 reachability ({tp1_reach:.1%})")
        
        if (signal_side == 'buy' and dominant_bias == 'bearish') or \
           (signal_side == 'sell' and dominant_bias == 'bullish'):
            reasons.append(f"Counter-trend signal (market is {dominant_bias})")
        
        return " | ".join(reasons) if reasons else "Failed validation criteria"