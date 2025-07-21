"""
Advanced Fibonacci and Multi-Method Confluence Analysis for the Enhanced Bybit Trading System.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List


class FibonacciConfluenceAnalyzer:
    """Advanced Fibonacci and Multi-Method Confluence Analysis"""
    
    def __init__(self):
        self.fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618]
        self.logger = logging.getLogger(__name__)
        
    def calculate_fibonacci_levels(self, df: pd.DataFrame, period: int = 50) -> Dict:
        """Calculate comprehensive Fibonacci levels"""
        try:
            recent_data = df.tail(period)
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Determine trend direction
            trend = 'uptrend' if df['close'].iloc[-1] > df['close'].iloc[-period] else 'downtrend'
            
            fib_range = swing_high - swing_low
            levels = {}
            
            if trend == 'uptrend':
                # Retracement levels (from high to low)
                for ratio in self.fib_ratios:
                    levels[f'{ratio*100:.1f}%'] = swing_high - (fib_range * ratio)
            else:
                # Extension levels (from low to high)
                for ratio in self.fib_ratios:
                    levels[f'{ratio*100:.1f}%'] = swing_low + (fib_range * ratio)
            
            # Calculate strength of each level based on historical touches
            level_strength = {}
            for level_name, price in levels.items():
                touches = self.count_level_touches(df, price, tolerance_pct=0.005)
                level_strength[level_name] = touches
            
            return {
                'trend': trend,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'range': fib_range,
                'levels': levels,
                'level_strength': level_strength,
                'key_levels': {
                    'golden_ratio': levels.get('61.8%', swing_low),
                    'half_retracement': levels.get('50.0%', swing_low),
                    'shallow_retracement': levels.get('38.2%', swing_low)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Fibonacci calculation error: {e}")
            return self.get_default_fibonacci()
    
    def count_level_touches(self, df: pd.DataFrame, level: float, tolerance_pct: float = 0.005) -> int:
        """Count how many times price touched a specific level"""
        try:
            tolerance = level * tolerance_pct
            touches = 0
            
            for _, row in df.iterrows():
                if (row['low'] <= level + tolerance <= row['high']) or \
                   (abs(row['close'] - level) <= tolerance):
                    touches += 1
            
            return touches
        except Exception:
            return 0
    
    def find_confluence_zones(self, df: pd.DataFrame, volume_profile: Dict, current_price: float) -> List[Dict]:
        """Find confluence zones where multiple analysis methods agree"""
        try:
            confluence_zones = []
            
            # Get all analysis levels
            fibonacci_data = self.calculate_fibonacci_levels(df)
            
            # Collect all significant levels
            all_levels = []
            
            # Fibonacci levels
            for level_name, price in fibonacci_data['levels'].items():
                strength = fibonacci_data['level_strength'].get(level_name, 0)
                all_levels.append({
                    'price': price,
                    'type': f'fib_{level_name}',
                    'strength': max(1, strength),
                    'importance': 3 if '61.8%' in level_name else 2 if level_name in ['38.2%', '50.0%'] else 1
                })
            
            # Volume Profile levels
            all_levels.append({
                'price': volume_profile.get('poc', current_price),
                'type': 'volume_poc',
                'strength': 3,
                'importance': 3
            })
            
            all_levels.append({
                'price': volume_profile.get('vah', current_price * 1.02),
                'type': 'volume_vah',
                'strength': 2,
                'importance': 2
            })
            
            all_levels.append({
                'price': volume_profile.get('val', current_price * 0.98),
                'type': 'volume_val',
                'strength': 2,
                'importance': 2
            })
            
            # High Volume Nodes
            for hvn in volume_profile.get('hvn_levels', []):
                all_levels.append({
                    'price': hvn,
                    'type': 'hvn',
                    'strength': 2,
                    'importance': 2
                })
            
            # Technical levels from indicators
            latest = df.iloc[-1]
            tech_levels = [
                ('bb_upper', latest.get('bb_upper', 0), 1),
                ('bb_lower', latest.get('bb_lower', 0), 1),
                ('support', latest.get('support', 0), 2),
                ('resistance', latest.get('resistance', 0), 2),
                ('sma_50', latest.get('sma_50', 0), 1),
                ('sma_200', latest.get('sma_200', 0), 2),
                ('vwap', latest.get('vwap', 0), 2)
            ]
            
            for name, price, importance in tech_levels:
                if price > 0:
                    all_levels.append({
                        'price': price,
                        'type': name,
                        'strength': importance,
                        'importance': importance
                    })
            
            # Find confluence (levels within 0.5% of each other)
            confluence_threshold = 0.005  # 0.5%
            processed_indices = set()
            
            for i, level1 in enumerate(all_levels):
                if i in processed_indices:
                    continue
                    
                confluence_group = [level1]
                total_strength = level1['strength']
                total_importance = level1['importance']
                
                for j, level2 in enumerate(all_levels[i+1:], i+1):
                    if j in processed_indices:
                        continue
                        
                    price_diff = abs(level1['price'] - level2['price']) / max(level1['price'], 0.01)
                    
                    if price_diff <= confluence_threshold:
                        confluence_group.append(level2)
                        total_strength += level2['strength']
                        total_importance += level2['importance']
                        processed_indices.add(j)
                
                if len(confluence_group) >= 2:  # At least 2 methods agree
                    avg_price = np.mean([level['price'] for level in confluence_group])
                    
                    confluence_zones.append({
                        'price': avg_price,
                        'strength': total_strength,
                        'importance': total_importance,
                        'confluence_count': len(confluence_group),
                        'methods': [level['type'] for level in confluence_group],
                        'distance_from_current': abs(avg_price - current_price) / current_price,
                        'zone_type': self.classify_zone_type(avg_price, current_price)
                    })
                
                processed_indices.add(i)
            
            # Sort by total score (strength + importance - distance)
            for zone in confluence_zones:
                distance_penalty = zone['distance_from_current'] * 5
                zone['total_score'] = zone['strength'] + zone['importance'] - distance_penalty
            
            confluence_zones.sort(key=lambda x: x['total_score'], reverse=True)
            
            return confluence_zones[:10]  # Top 10 confluence zones
            
        except Exception as e:
            self.logger.error(f"Confluence analysis error: {e}")
            return []
    
    def classify_zone_type(self, zone_price: float, current_price: float) -> str:
        """Classify zone as support or resistance"""
        if zone_price < current_price:
            return 'support'
        elif zone_price > current_price:
            return 'resistance'
        else:
            return 'current_level'
    
    def get_default_fibonacci(self) -> Dict:
        """Return default Fibonacci data"""
        return {
            'trend': 'sideways',
            'swing_high': 0,
            'swing_low': 0,
            'range': 0,
            'levels': {},
            'level_strength': {},
            'key_levels': {'golden_ratio': 0, 'half_retracement': 0, 'shallow_retracement': 0}
        }