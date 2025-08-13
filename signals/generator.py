"""
signals/generator.py - Integration wrapper for V13
===================================================
This file integrates the new V13 architecture with your existing system
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import asyncio

# Try to import V13 components
try:
    from signals.signal_gen_v13_main import SignalGeneratorV13, create_signal_generator as create_v13_generator
    from signals.signal_gen_v13_core import (
        SystemConfiguration, TimeFrame, SignalQuality
    )
    V13_AVAILABLE = True
except ImportError as e:
    print(f"Warning: V13 modules not fully available: {e}")
    V13_AVAILABLE = False

# Import V12 as fallback (your existing generator)
try:
    from signals.generator_v12 import SignalGeneratorV12  # Your existing V12
    V12_AVAILABLE = True
except ImportError:
    V12_AVAILABLE = False

# ===========================
# COMPATIBILITY WRAPPER
# ===========================

class SignalGenerator:
    """
    Compatibility wrapper that can use either V13 (new) or V12 (existing)
    """
    
    def __init__(self, config: Any, exchange_manager=None):
        """Initialize with automatic version selection"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchange_manager = exchange_manager
        
        # Try to use V13 first
        if V13_AVAILABLE:
            try:
                self._init_v13(config, exchange_manager)
                self.version = "13.0"
                self.logger.info("Using Signal Generator V13 (New Architecture)")
            except Exception as e:
                self.logger.warning(f"V13 initialization failed: {e}")
                if V12_AVAILABLE:
                    self._init_v12(config, exchange_manager)
                    self.version = "12.0"
                    self.logger.info("Falling back to V12")
                else:
                    raise RuntimeError("No signal generator version available")
        elif V12_AVAILABLE:
            self._init_v12(config, exchange_manager)
            self.version = "12.0"
            self.logger.info("Using Signal Generator V12")
        else:
            raise RuntimeError("No signal generator implementation available")
    
    def _init_v13(self, config, exchange_manager):
        """Initialize V13 generator"""
        # Convert config to V13 format
        v13_config = SystemConfiguration(
            primary_timeframe=TimeFrame.from_string(getattr(config, 'timeframe', '6h')),
            confirmation_timeframes=[
                TimeFrame.from_string(tf) for tf in 
                getattr(config, 'confirmation_timeframes', ['4h', '1h'])
            ],
            entry_timeframe=TimeFrame.from_string('1h'),
            max_concurrent_positions=getattr(config, 'max_concurrent_positions', 10),
            max_risk_per_trade=getattr(config, 'max_risk_per_trade', 0.02),
            max_daily_risk=getattr(config, 'max_daily_risk', 0.06)
        )
        
        self.generator = SignalGeneratorV13(v13_config, exchange_manager)
        self._is_v13 = True
    
    def _init_v12(self, config, exchange_manager):
        """Initialize V12 generator"""
        self.generator = SignalGeneratorV12(config, exchange_manager)
        self._is_v13 = False
    
    # ===========================
    # PUBLIC API (Compatible)
    # ===========================
    
    def analyze_symbol_comprehensive(self, df: pd.DataFrame, symbol_data: Dict,
                                   volume_entry: Dict, fibonacci_data: Dict,
                                   confluence_zones: List[Dict], 
                                   timeframe: str) -> Optional[Dict]:
        """
        Analyze symbol - compatible with both V12 and V13
        
        This method maintains backward compatibility with the existing interface
        """
        if self._is_v13:
            # V13 uses async internally, but we'll run it synchronously for compatibility
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.generator.analyze_symbol_comprehensive(symbol_data['symbol'])
                )
                return result
            finally:
                loop.close()
        else:
            # V12 - use existing method
            return self.generator.analyze_symbol_comprehensive(
                df, symbol_data, volume_entry, fibonacci_data,
                confluence_zones, timeframe
            )
    
    def rank_opportunities_with_mtf(self, signals: List[Dict], 
                                  dfs: Optional[Dict[str, pd.DataFrame]] = None) -> List[Dict]:
        """Rank signals - compatible with both versions"""
        if self._is_v13:
            return self.generator.rank_opportunities_with_mtf(signals)
        else:
            return self.generator.rank_opportunities_with_mtf(signals, dfs)
    
    async def start(self):
        """Start the generator (V13 only)"""
        if self._is_v13:
            await self.generator.start()
        else:
            self.logger.info("V12 doesn't require explicit start")
    
    async def stop(self):
        """Stop the generator (V13 only)"""
        if self._is_v13:
            await self.generator.stop()
        else:
            self.logger.info("V12 doesn't require explicit stop")
    
    def get_statistics(self) -> Dict:
        """Get generator statistics"""
        if self._is_v13:
            return self.generator.get_performance_stats()
        elif hasattr(self.generator, 'get_statistics'):
            return self.generator.get_statistics()
        else:
            return {
                'signals_generated': 0,
                'signals_rejected': 0,
                'success_rate': 0.0,
                'rejection_reasons': {}
            }
    
    # ===========================
    # V12 COMPATIBILITY METHODS
    # ===========================
    
    def create_technical_summary(self, df: pd.DataFrame, latest: pd.Series = None) -> Dict:
        """Create technical summary - V12 compatibility"""
        if hasattr(self.generator, 'create_technical_summary'):
            return self.generator.create_technical_summary(df, latest)
        else:
            # Basic implementation for V13
            return {
                'trend': {'direction': 'neutral', 'strength': 0.5},
                'momentum': {'rsi': 50, 'macd_bullish': False},
                'volatility': {'atr_percentage': 2.0},
                'volume': {'ratio': 1.0}
            }
    
    def analyze_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns - V12 compatibility"""
        if hasattr(self.generator, 'analyze_volume_patterns'):
            return self.generator.analyze_volume_patterns(df)
        else:
            return {'pattern': 'stable', 'buying_pressure': 0.5}
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Calculate trend strength - V12 compatibility"""
        if hasattr(self.generator, 'calculate_trend_strength'):
            return self.generator.calculate_trend_strength(df)
        else:
            return {'strength': 0.5, 'direction': 'neutral', 'consistency': 'medium'}
    
    def analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze price action - V12 compatibility"""
        if hasattr(self.generator, 'analyze_price_action'):
            return self.generator.analyze_price_action(df)
        else:
            return {'patterns': [], 'momentum': 0, 'strength': 0.5}
    
    def assess_market_conditions(self, df: pd.DataFrame, symbol_data: Dict) -> Dict:
        """Assess market conditions - V12 compatibility"""
        if hasattr(self.generator, 'assess_market_conditions'):
            return self.generator.assess_market_conditions(df, symbol_data)
        else:
            return {
                'liquidity': 'medium',
                'sentiment': 'neutral',
                'price_change_24h': symbol_data.get('price_change_24h', 0),
                'volume_24h': symbol_data.get('volume_24h', 0)
            }
    
    def assess_risk(self, df: pd.DataFrame, symbol_data: Dict) -> Dict:
        """Risk assessment - V12 compatibility"""
        if hasattr(self.generator, 'assess_risk'):
            return self.generator.assess_risk(df, symbol_data)
        else:
            return {
                'total_risk_score': 0.5,
                'volatility_risk': 0.02,
                'risk_level': 'Medium'
            }

# ===========================
# FACTORY FUNCTIONS
# ===========================

def create_signal_generator(config: Any, exchange_manager=None) -> SignalGenerator:
    """
    Factory function to create signal generator
    Automatically selects best available version
    """
    return SignalGenerator(config, exchange_manager)

def create_mtf_signal_generator(config: Any, exchange_manager=None) -> SignalGenerator:
    """
    Alias for compatibility with existing code
    """
    return create_signal_generator(config, exchange_manager)

# ===========================
# EXPORTS
# ===========================

__all__ = [
    'SignalGenerator',
    'create_signal_generator',
    'create_mtf_signal_generator'
]

# Version info
__version__ = "13.0" if V13_AVAILABLE else "12.0"
__status__ = "PRODUCTION"