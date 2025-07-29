import pandas as pd
import numpy as np
import ta
import time
import logging
from typing import Dict, List, Optional
from enum import Enum

from config.config import EnhancedSystemConfig
from core.exchange import ExchangeManager
from analysis.technical import EnhancedTechnicalAnalysis
from analysis.volume_profile import VolumeProfileAnalyzer
from analysis.fibonacci import FibonacciConfluenceAnalyzer
from signals.generator import SignalGenerator

# Import the new configurable optimizer
from analysis.configurable_mtf_optimizer import EnhancedMultiTimeframeAnalyzer

# Your existing MultiTimeframeAnalyzer becomes a wrapper:
class NewMultiTimeframeAnalyzer:
    """Multi-timeframe signal confirmation analyzer with configurable optimization"""
    
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the enhanced analyzer with configurable optimization
        self.enhanced_analyzer = EnhancedMultiTimeframeAnalyzer(exchange, config)
        
        self.logger.info("✅ MTF Analyzer with Configurable Level Optimization initialized")
        self.logger.info(f"   Primary: {config.timeframe}")
        self.logger.info(f"   Confirmations: {getattr(config, 'confirmation_timeframes', [])}")
        self.logger.info("   🎯 Level optimization: ENABLED and ADAPTIVE")
        
    def analyze_symbol_multi_timeframe_with_level_optimization(self, symbol_data: Dict, primary_signal: Dict) -> Dict:
        """Main interface - now with enhanced level optimization"""
        return self.enhanced_analyzer.analyze_symbol_multi_timeframe(symbol_data, primary_signal)
    

class OldMultiTimeframeAnalyzer:
    """Multi-timeframe signal confirmation analyzer"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.exchange_manager = ExchangeManager(config)
        self.enhanced_ta = EnhancedTechnicalAnalysis()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.fibonacci_analyzer = FibonacciConfluenceAnalyzer()
        self.signal_generator = SignalGenerator(config)
        
    def analyze_symbol_multi_timeframe(self, symbol_data: Dict, primary_signal: Dict) -> Dict:
        """Analyze symbol across multiple timeframes for signal confirmation"""
        try:
            symbol = symbol_data['symbol']
            self.logger.debug(f"🔍 Multi-timeframe analysis for {symbol}")
            
            confirmation_results = {
                'confirmed_timeframes': [],
                'conflicting_timeframes': [],
                'neutral_timeframes': [],
                'confirmation_strength': 0.0,
                'mtf_confidence_boost': 0.0,
                'timeframe_signals': {}
            }
            
            primary_side = primary_signal['side']
            
            # Analyze each confirmation timeframe
            for timeframe in self.config.confirmation_timeframes:
                try:
                    self.logger.debug(f"   Analyzing {timeframe} timeframe...")
                    
                    # Fetch data for this timeframe
                    df = self.exchange_manager.fetch_ohlcv_data(symbol, timeframe)
                    if df.empty or len(df) < 50:
                        self.logger.debug(f"Insufficient {timeframe} data for {symbol}")
                        confirmation_results['neutral_timeframes'].append(timeframe)
                        continue

                    # APPROACH 1: Enhanced Technical Analysis
                    df = self.enhanced_ta.calculate_all_indicators(df, self.config)
                    
                    # APPROACH 2: Volume Profile Analysis
                    volume_profile = self.volume_analyzer.calculate_volume_profile(df)
                    volume_entry = self.volume_analyzer.find_optimal_entry_from_volume(
                        df, symbol_data['current_price'], 'buy'
                    )
                    
                    # APPROACH 3: Fibonacci & Confluence Analysis
                    fibonacci_data = self.fibonacci_analyzer.calculate_fibonacci_levels(df)
                    confluence_zones = self.fibonacci_analyzer.find_confluence_zones(
                        df, volume_profile, symbol_data['current_price']
                    )
                    
                    timeframe_signal = self.signal_generator.analyze_symbol_comprehensive(
                        df, symbol_data, volume_entry, fibonacci_data, confluence_zones, timeframe
                    )

                    if timeframe_signal:
                        confirmation_results['timeframe_signals'][timeframe] = timeframe_signal

                        if timeframe_signal['side'] == primary_side:
                            confirmation_results['confirmed_timeframes'].append(timeframe)
                            self.logger.debug(f"   ✅ {timeframe}: {symbol} {timeframe_signal['side'].upper()} signal confirmed")
                        else:
                            confirmation_results['conflicting_timeframes'].append(timeframe)
                            self.logger.debug(f"   ⚠️ {timeframe}: {symbol} {timeframe_signal['side'].upper()} signal conflicts")
                    else:
                        confirmation_results['neutral_timeframes'].append(timeframe)
                        self.logger.debug(f"   ➖ {timeframe}: {symbol} No clear signal")
                    
                    # Add small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"   Error analyzing {timeframe}: {e}")
                    confirmation_results['neutral_timeframes'].append(timeframe)
            
            # Calculate confirmation strength
            total_timeframes = len(self.config.confirmation_timeframes)
            confirmed_count = len(confirmation_results['confirmed_timeframes'])
            conflicting_count = len(confirmation_results['conflicting_timeframes'])
            neutral_count = len(confirmation_results['neutral_timeframes'])
            
            if total_timeframes > 0:
                confirmation_results['confirmation_strength'] = confirmed_count / total_timeframes
                
                # Calculate confidence boost
                if confirmed_count > 0:
                    # Boost based on confirmed timeframes minus conflicts
                    net_confirmation = confirmed_count - conflicting_count
                    confirmation_results['mtf_confidence_boost'] = max(0, 
                        (net_confirmation / total_timeframes) * (self.config.mtf_weight_multiplier - 1) * 100
                    )
            
            self.logger.debug(f"   📊 Confirmation: {confirmed_count}/{total_timeframes} timeframes")
            self.logger.debug(f"   💪 Strength: {confirmation_results['confirmation_strength']:.1%}")
            
            return confirmation_results
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe analysis error for {symbol}: {e}")
            return {
                'confirmed_timeframes': [],
                'conflicting_timeframes': [],
                'neutral_timeframes': self.config.confirmation_timeframes.copy(),
                'confirmation_strength': 0.0,
                'mtf_confidence_boost': 0.0,
                'timeframe_signals': {}
            }      
    