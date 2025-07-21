"""
Professional Volume Profile Analysis for the Enhanced Bybit Trading System.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict


class VolumeProfileAnalyzer:
    """Professional Volume Profile Analysis"""
    
    def __init__(self, bins: int = 50):
        self.bins = bins
        self.logger = logging.getLogger(__name__)
        
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive volume profile"""
        try:
            if df.empty or len(df) < 20:
                return self.get_default_volume_profile()
            
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, self.bins)
            
            # Initialize volume arrays
            volume_at_price = np.zeros(len(price_bins) - 1)
            buy_volume = np.zeros(len(price_bins) - 1)
            sell_volume = np.zeros(len(price_bins) - 1)
            
            # Distribute volume across price levels
            for _, row in df.iterrows():
                if row['high'] <= row['low'] or row['volume'] <= 0:
                    continue
                    
                # Create sub-bins for this candle
                candle_prices = np.linspace(row['low'], row['high'], 10)
                volume_per_tick = row['volume'] / len(candle_prices)
                
                # Determine buy/sell volume based on close vs open
                if row['close'] > row['open']:
                    buy_vol_ratio = 0.6  # 60% buy volume for green candles
                else:
                    buy_vol_ratio = 0.4  # 40% buy volume for red candles
                
                buy_vol_per_tick = volume_per_tick * buy_vol_ratio
                sell_vol_per_tick = volume_per_tick * (1 - buy_vol_ratio)
                
                for price in candle_prices:
                    bin_idx = np.digitize(price, price_bins) - 1
                    if 0 <= bin_idx < len(volume_at_price):
                        volume_at_price[bin_idx] += volume_per_tick
                        buy_volume[bin_idx] += buy_vol_per_tick
                        sell_volume[bin_idx] += sell_vol_per_tick
            
            # Calculate key levels
            poc_idx = np.argmax(volume_at_price)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Value Area (68% of volume)
            total_volume = np.sum(volume_at_price)
            value_area_volume = total_volume * 0.68
            
            # Find Value Area High and Low
            sorted_indices = np.argsort(volume_at_price)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_at_price[idx]
                value_area_indices.append(idx)
                if cumulative_volume >= value_area_volume:
                    break
            
            if value_area_indices:
                vah_idx = max(value_area_indices)
                val_idx = min(value_area_indices)
                vah = (price_bins[vah_idx] + price_bins[vah_idx + 1]) / 2
                val = (price_bins[val_idx] + price_bins[val_idx + 1]) / 2
            else:
                vah = poc_price * 1.02
                val = poc_price * 0.98
            
            # High Volume Nodes (HVN) and Low Volume Nodes (LVN)
            volume_threshold_high = np.percentile(volume_at_price, 80)
            volume_threshold_low = np.percentile(volume_at_price, 20)
            
            hvn_levels = []
            lvn_levels = []
            
            for i, vol in enumerate(volume_at_price):
                price = (price_bins[i] + price_bins[i + 1]) / 2
                if vol >= volume_threshold_high:
                    hvn_levels.append(price)
                elif vol <= volume_threshold_low:
                    lvn_levels.append(price)
            
            # Delta analysis
            net_delta = buy_volume - sell_volume
            cumulative_delta = np.cumsum(net_delta)
            
            return {
                'poc': poc_price,
                'vah': vah,
                'val': val,
                'hvn_levels': hvn_levels,
                'lvn_levels': lvn_levels,
                'volume_distribution': list(zip(price_bins[:-1], volume_at_price)),
                'buy_volume_distribution': list(zip(price_bins[:-1], buy_volume)),
                'sell_volume_distribution': list(zip(price_bins[:-1], sell_volume)),
                'delta_profile': list(zip(price_bins[:-1], net_delta)),
                'cumulative_delta': cumulative_delta,
                'total_volume': total_volume,
                'value_area_volume_pct': (cumulative_volume / total_volume) * 100 if total_volume > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Volume profile calculation error: {e}")
            return self.get_default_volume_profile()
    
    def get_default_volume_profile(self) -> Dict:
        """Return default volume profile when calculation fails"""
        return {
            'poc': 0,
            'vah': 0,
            'val': 0,
            'hvn_levels': [],
            'lvn_levels': [],
            'volume_distribution': [],
            'buy_volume_distribution': [],
            'sell_volume_distribution': [],
            'delta_profile': [],
            'cumulative_delta': [],
            'total_volume': 0,
            'value_area_volume_pct': 0
        }
    
    def find_optimal_entry_from_volume(self, df: pd.DataFrame, current_price: float, side: str) -> Dict:
        """Find optimal entry based on volume profile"""
        try:
            volume_profile = self.calculate_volume_profile(df)
            
            if not volume_profile['hvn_levels']:
                return {'entry_price': current_price, 'confidence': 0.3, 'method': 'fallback'}
            
            poc = volume_profile['poc']
            hvn_levels = volume_profile['hvn_levels']
            val = volume_profile['val']
            vah = volume_profile['vah']
            
            if side == 'buy':
                # For buy orders, look for support levels below current price
                candidate_levels = [level for level in hvn_levels if level < current_price]
                candidate_levels.append(poc if poc < current_price else None)
                candidate_levels.append(val if val < current_price else None)
                candidate_levels = [level for level in candidate_levels if level is not None]
                
                if candidate_levels:
                    # Choose the closest high-volume level
                    optimal_entry = max(candidate_levels)
                    distance = abs(optimal_entry - current_price) / current_price
                    confidence = max(0.5, 1 - distance * 10)
                else:
                    optimal_entry = current_price * 0.995  # 0.5% below
                    confidence = 0.4
            
            else:  # sell
                # For sell orders, look for resistance levels above current price
                candidate_levels = [level for level in hvn_levels if level > current_price]
                candidate_levels.append(poc if poc > current_price else None)
                candidate_levels.append(vah if vah > current_price else None)
                candidate_levels = [level for level in candidate_levels if level is not None]
                
                if candidate_levels:
                    optimal_entry = min(candidate_levels)
                    distance = abs(optimal_entry - current_price) / current_price
                    confidence = max(0.5, 1 - distance * 10)
                else:
                    optimal_entry = current_price * 1.005  # 0.5% above
                    confidence = 0.4
            
            return {
                'entry_price': optimal_entry,
                'confidence': min(0.95, confidence),
                'method': 'volume_profile',
                'supporting_data': {
                    'poc': poc,
                    'vah': vah,
                    'val': val,
                    'hvn_count': len(hvn_levels)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Volume profile entry calculation error: {e}")
            return {'entry_price': current_price, 'confidence': 0.3, 'method': 'fallback'}