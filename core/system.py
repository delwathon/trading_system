"""
Complete Enhanced Bybit System Integration with Multi-Timeframe Analysis and MySQL Database.
Updated to use database storage and process ONLY TOP OPPORTUNITIES from generator.py
Primary timeframe: 30m, Confirmation timeframes: 1h, 4h, 6h
"""

import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from config.config import EnhancedSystemConfig, DatabaseConfig
from core.exchange import ExchangeManager
from analysis.technical import EnhancedTechnicalAnalysis
from analysis.volume_profile import VolumeProfileAnalyzer
from analysis.fibonacci import FibonacciConfluenceAnalyzer
from signals.generator import create_mtf_signal_generator
from visualization.charts import InteractiveChartGenerator
from utils.database_manager import EnhancedDatabaseManager
from utils.logging import get_logger 
from database.models import DatabaseManager


class CompleteEnhancedBybitSystem:
    """Complete enhanced Bybit system with multi-timeframe confirmation and TOP OPPORTUNITIES focus"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config.db_config.get_database_url())
        self.enhanced_db_manager = EnhancedDatabaseManager(self.db_manager)
        
        # Test database connection and create tables
        if not self.db_manager.test_connection():
            raise Exception("Failed to connect to MySQL database")
        
        try:
            self.db_manager.create_tables()
        except Exception as e:
            self.logger.warning(f"Tables may already exist: {e}")
        
        # Initialize components
        self.exchange_manager = ExchangeManager(config)
        self.enhanced_ta = EnhancedTechnicalAnalysis()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.fibonacci_analyzer = FibonacciConfluenceAnalyzer()
        self.signal_generator = create_mtf_signal_generator(config, self.exchange_manager)
        self.chart_generator = InteractiveChartGenerator(config)
        
        # System state
        self.signal_history = []
        self.analysis_lock = threading.Lock()
        self.results_queue = Queue()
        self.processed_count = 0
        
        self.logger.debug("🚀 TOP OPPORTUNITIES SYSTEM WITH MULTI-TIMEFRAME & MYSQL DATABASE INITIALIZED!")
        self.logger.debug("✅ Enhanced Technical Analysis (FIXED Ichimoku & Stochastic RSI)")
        self.logger.debug("✅ Volume Profile Analysis") 
        self.logger.debug("✅ Fibonacci & Confluence Analysis")
        self.logger.debug("✅ Interactive Chart Generation")
        self.logger.debug("✅ MySQL Database Storage")
        self.logger.debug("🎯 TOP OPPORTUNITIES FOCUS: Quality over Quantity")
        if self.config.mtf_confirmation_required:
            self.logger.debug(f"✅ Multi-Timeframe Confirmation: Primary 30m, Confirmation 1h/4h/6h")
        self.logger.debug(f"✅ Database: {self.config.db_config.database} @ {self.config.db_config.host}")
    
    def analyze_symbol_complete_with_mtf(self, symbol_data: Dict) -> Optional[Dict]:
        """
        Enhanced MTF analysis method that works with the new structure-aware signal generator
        
        Key Changes:
        - The new generator handles MTF analysis internally
        - No need for separate multi_timeframe.py calls
        - Better integration and fewer API calls
        - Maintains full compatibility with existing system flow
        """
        try:
            symbol = symbol_data['symbol']
            current_price = symbol_data['current_price']
            
            self.logger.debug(f"📊 Enhanced MTF analysis for {symbol} (Primary: {self.config.timeframe})")
            
            # Fetch primary timeframe data (your configured timeframe, e.g., 1h)
            df = self.exchange_manager.fetch_ohlcv_data(symbol, self.config.timeframe)
            if df.empty or len(df) < 50:
                self.logger.debug(f"   Insufficient {self.config.timeframe} data for {symbol}")
                return None
            
            # Enhanced Technical Analysis on primary timeframe
            df = self.enhanced_ta.calculate_all_indicators(df, self.config)
            
            # Volume Profile Analysis (existing)
            volume_profile = self.volume_analyzer.calculate_volume_profile(df)
            volume_entry = self.volume_analyzer.find_optimal_entry_from_volume(
                df, current_price, 'buy'
            )
            
            # Fibonacci & Confluence Analysis (existing)
            fibonacci_data = self.fibonacci_analyzer.calculate_fibonacci_levels(df)
            confluence_zones = self.fibonacci_analyzer.find_confluence_zones(
                df, volume_profile, current_price
            )
            
            # === NEW: Structure-Aware Signal Generation ===
            # The signal generator now handles MTF analysis internally
            # This replaces the old separate MTF confirmation step
            primary_signal = self.signal_generator.analyze_symbol_comprehensive(
                df, symbol_data, volume_entry, fibonacci_data, confluence_zones, self.config.timeframe
            )
            
            if primary_signal:
                # Check if signal was validated with MTF structure analysis
                mtf_validated = primary_signal.get('mtf_validated', False)
                analysis_details = primary_signal.get('analysis_details', {})
                
                if mtf_validated:
                    # Signal was generated with full MTF structure awareness
                    self.logger.debug(f"   ✅ {symbol} - MTF-validated {primary_signal['side'].upper()} signal")
                    self.logger.debug(f"   📊 Entry: ${primary_signal['entry_price']:.6f} via {primary_signal.get('entry_strategy', 'structure')}")
                    self.logger.debug(f"   🎯 Confidence: {primary_signal['confidence']:.1f}% (MTF: {analysis_details.get('mtf_trend', 'confirmed')})")
                    
                    # Add MTF status for compatibility with ranking system
                    primary_signal['mtf_status'] = 'MTF_VALIDATED'
                    primary_signal['mtf_analysis'] = {
                        'method': 'structure_aware_generation',
                        'structure_timeframe': analysis_details.get('structure_timeframe', self.config.confirmation_timeframes[-1]),
                        'confirmation_score': analysis_details.get('confirmation_score', 0.8),
                        'entry_method': primary_signal.get('entry_strategy', 'structure_based'),
                        'mtf_trend': analysis_details.get('mtf_trend', 'aligned'),
                        'validated_timeframes': self.config.confirmation_timeframes,
                        'primary_timeframe': self.config.timeframe
                    }
                    
                # Chart generation will happen later for top signals only
                primary_signal['chart_file'] = None
                
                # Store chart data for later generation (top signals only)
                primary_signal['_chart_data'] = {
                    'df': df,
                    'symbol_data': symbol_data
                }
                
                # Add execution metadata
                primary_signal['analysis_method'] = 'enhanced_mtf' if mtf_validated else 'traditional_fallback'
                primary_signal['generation_timestamp'] = pd.Timestamp.now()
                
                # Enhanced logging based on signal quality
                if mtf_validated:
                    entry_strategy = primary_signal.get('entry_strategy', 'structure')
                    self.logger.debug(f"   🎯 Structure-aware signal: {entry_strategy} entry method")
                else:
                    self.logger.debug(f"   ⚠️  Traditional signal: May lack higher timeframe validation")
                
                return primary_signal
            
            else:
                # No signal generated - this is often due to structure filtering
                self.logger.debug(f"   ❌ {symbol} - No signal (likely filtered by MTF structure analysis)")
                return None
                
        except Exception as e:
            self.logger.error(f"Enhanced MTF analysis failed for {symbol_data.get('symbol', 'unknown')}: {e}")
            return None

    def analyze_symbol_thread_safe_mtf(self, symbol_data: Dict, thread_id: int) -> Optional[Dict]:
        """
        Enhanced thread-safe version of MTF symbol analysis
        
        Updated to work with the new structure-aware signal generation
        """
        try:
            symbol = symbol_data['symbol']
            
            # Thread-safe logging
            with self.analysis_lock:
                self.processed_count += 1
                timeframe_display = f"{self.config.timeframe}→{'/'.join(self.config.confirmation_timeframes)}"
                self.logger.debug(f"[Thread-{thread_id}] [{self.processed_count}] {symbol} with {timeframe_display} TF...")
            
            # Call the enhanced MTF analysis method
            result = self.analyze_symbol_complete_with_mtf(symbol_data)
            
            if result:
                with self.analysis_lock:
                    mtf_status = result.get('mtf_status', 'UNKNOWN')
                    analysis_method = result.get('analysis_method', 'unknown')
                    confidence = result.get('confidence', 0)
                    entry_strategy = result.get('entry_strategy', 'immediate')
                    
                    status_emoji = {
                        'MTF_VALIDATED': '🚀',
                        'STRONG': '✅', 
                        'PARTIAL': '🔸',
                        'NONE': '⚪',
                        'DISABLED': '❌'
                    }.get(mtf_status, '❓')
                    
                    self.logger.debug(f"[Thread-{thread_id}] {status_emoji} {symbol} - {result['side'].upper()} "
                                    f"signal (MTF: {mtf_status}, Method: {analysis_method}, "
                                    f"Conf: {confidence:.0f}%, Entry: {entry_strategy})")
            
            return result
            
        except Exception as e:
            with self.analysis_lock:
                self.logger.error(f"[Thread-{thread_id}] Error analyzing {symbol_data.get('symbol', 'unknown')}: {e}")
            return None

    def run_complete_analysis_parallel_mtf(self) -> Dict:
        """
        Enhanced parallel analysis with integrated MTF structure awareness
        
        Key improvements:
        - Uses new structure-aware signal generator
        - Reduced API calls (MTF analysis integrated)
        - Better signal quality through structure filtering
        - Maintains compatibility with existing chart generation
        """
        
        if self.config.monitor_mode: 
            self.logger.info("🔍 Running in MONITOR MODE - No analysis performed")
            return {}

        start_time = time.time()
        
        # Display method being used
        timeframe_display = f"{self.config.timeframe}→{'/'.join(self.config.confirmation_timeframes)}"
        
        self.logger.info(f"🚀 STARTING ANALYSIS")
        self.logger.info(f"   Timeframes: {timeframe_display}")
        self.logger.info(f"   Structure Analysis: {self.config.confirmation_timeframes[-1] if self.config.confirmation_timeframes else '6h'} dominant")
        self.logger.info(f"   Chart Strategy: Top {self.config.charts_per_batch} signals only")
        
        # Get top symbols
        symbols = self.exchange_manager.get_top_symbols()
        if not symbols:
            self.logger.error("No symbols found")
            return {}
        
        # Initialize counters
        self.processed_count = 0
        all_signals = []
        analysis_results = []
        
        # ===== PHASE 1: ENHANCED SIGNAL GENERATION =====
        self.logger.info("📈 PHASE 1: Enhanced Signal Generation")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_symbol = {}
            
            for i, symbol_data in enumerate(symbols):
                future = executor.submit(self.analyze_symbol_thread_safe_mtf, symbol_data, i % self.config.max_workers)
                future_to_symbol[future] = symbol_data
            
            # Process completed tasks
            completed_tasks = 0
            mtf_validated_count = 0
            traditional_count = 0
            
            for future in as_completed(future_to_symbol):
                symbol_data = future_to_symbol[future]
                completed_tasks += 1
                
                try:
                    result = future.result()
                    
                    if result:
                        with self.analysis_lock:
                            all_signals.append(result)
                            
                            # Track signal generation method
                            if result.get('mtf_validated', False):
                                mtf_validated_count += 1
                            else:
                                traditional_count += 1
                        
                        # Enhanced analysis results tracking
                        analysis_results.append({
                            'symbol': symbol_data['symbol'],
                            'status': 'success',
                            'signal_generated': True,
                            'confidence': result['confidence'],
                            'original_confidence': result.get('original_confidence', result['confidence']),
                            'side': result['side'],
                            'mtf_status': result.get('mtf_status', 'UNKNOWN'),
                            'analysis_method': result.get('analysis_method', 'unknown'),
                            'entry_strategy': result.get('entry_strategy', 'immediate'),
                            'mtf_validated': result.get('mtf_validated', False)
                        })
                    else:
                        analysis_results.append({
                            'symbol': symbol_data['symbol'],
                            'status': 'no_signal',
                            'signal_generated': False,
                            'reason': 'filtered_by_structure_or_conditions'
                        })
                    
                    # Enhanced progress reporting
                    if completed_tasks % 10 == 0 or completed_tasks == len(symbols):
                        progress = (completed_tasks / len(symbols)) * 100
                        self.logger.debug(f"   Progress: {completed_tasks}/{len(symbols)} ({progress:.1f}%) - "
                                        f"Signals: {len(all_signals)} (MTF: {mtf_validated_count}, Traditional: {traditional_count})")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol_data['symbol']}: {e}")
                    analysis_results.append({
                        'symbol': symbol_data['symbol'],
                        'status': 'error',
                        'error': str(e)
                    })
        
        self.logger.info(f"✅ Signal generation completed")
        self.logger.info(f"   📊 Results: {len(all_signals)} signals ({mtf_validated_count} MTF-validated, {traditional_count} traditional)")
        
        # ===== PHASE 2: ENHANCED SIGNAL RANKING =====
        analysis_time = time.time() - start_time
        self.logger.info(f"📊 PHASE 2: Enhanced Signal Ranking")
        
        # Use enhanced ranking system that prioritizes MTF-validated signals
        ranked_signals = self.signal_generator.rank_opportunities_with_mtf(all_signals)

        if not ranked_signals:
            self.logger.info("No signals found after enhanced ranking")
            return self._create_empty_results(start_time, len(symbols), analysis_results)
        
        # Enhanced ranking summary
        mtf_validated_ranked = len([s for s in ranked_signals if s.get('mtf_validated', False)])
        avg_confidence = np.mean([s['confidence'] for s in ranked_signals])
        
        self.logger.info(f"✅ Enhanced ranking completed: {len(ranked_signals)} top signals")
        self.logger.info(f"   🎯 MTF-validated in top results: {mtf_validated_ranked}/{len(ranked_signals)}")
        self.logger.info(f"   📈 Average confidence: {avg_confidence:.1f}%")
        
        execution_time = time.time() - start_time
        
        # ===== PHASE 3: ENHANCED RESULTS COMPILATION =====
        return self._compile_enhanced_results(
            start_time, execution_time, symbols, ranked_signals, analysis_results, 
            mtf_validated_count, traditional_count
        )

    def _create_empty_results(self, start_time: float, symbols_count: int, analysis_results: List[Dict]) -> Dict:
        """Create empty results structure when no signals found"""
        execution_time = time.time() - start_time
        
        return {
            'scan_info': {
                'timestamp': datetime.now(),
                'execution_time_seconds': execution_time,
                'symbols_analyzed': symbols_count,
                'signals_generated': 0,
                'success_rate': 0,
                'method': 'enhanced_mtf',
                'mtf_enabled': True
            },
            'signals': [],
            'analysis_results': analysis_results,
            'top_opportunities': [],
            'market_summary': {},
            'system_performance': {
                'signals_per_minute': 0,
                'avg_confidence': 0,
                'mtf_validation_rate': 0
            }
        }

    def _compile_enhanced_results(self, start_time: float, execution_time: float, symbols: List[Dict], 
                                ranked_signals: List[Dict], analysis_results: List[Dict], 
                                mtf_validated_count: int, traditional_count: int) -> Dict:
        """Compile enhanced results with MTF statistics"""
        
        scan_info = {
            'timestamp': datetime.now(),
            'execution_time_seconds': execution_time,
            'symbols_analyzed': len(symbols),
            'signals_generated': len(ranked_signals),
            'success_rate': len(ranked_signals) / len(symbols) * 100 if symbols else 0,
            'parallel_processing': True,
            'threads_used': self.config.max_workers,
            'method': 'enhanced_mtf_structure_aware',
            'mtf_enabled': True,
            'confirmation_timeframes': self.config.confirmation_timeframes,
            'primary_timeframe': self.config.timeframe,
            'optimization': 'structure_aware_generation'
        }
        
        # Enhanced system performance metrics
        mtf_validation_rate = (mtf_validated_count / (mtf_validated_count + traditional_count) * 100) if (mtf_validated_count + traditional_count) > 0 else 0
        
        results = {
            'scan_info': scan_info,
            'signals': ranked_signals,
            'analysis_results': analysis_results,
            'top_opportunities': ranked_signals,
            'market_summary': self.create_market_summary_with_mtf(symbols, ranked_signals),
            'system_performance': {
                'signals_per_minute': len(ranked_signals) / (execution_time / 60) if execution_time > 0 else 0,
                'avg_confidence': np.mean([s['confidence'] for s in ranked_signals]) if ranked_signals else 0,
                'avg_original_confidence': np.mean([s.get('original_confidence', s['confidence']) for s in ranked_signals]) if ranked_signals else 0,
                'mtf_boost_avg': np.mean([s['confidence'] - s.get('original_confidence', s['confidence']) for s in ranked_signals]) if ranked_signals else 0,
                'mtf_validation_rate': mtf_validation_rate,
                'mtf_validated_signals': mtf_validated_count,
                'traditional_signals': traditional_count,
                'structure_filtered_rate': (len(symbols) - len(ranked_signals)) / len(symbols) * 100 if symbols else 0,
                'order_type_distribution': self.get_order_type_distribution(ranked_signals),
                'mtf_distribution': self.get_enhanced_mtf_distribution(ranked_signals),
                'speedup_factor': self.calculate_speedup_factor(execution_time, len(symbols)),
            }
        }
        
        # Enhanced completion logging
        method_name = "ENHANCED MTF STRUCTURE-AWARE ANALYSIS"
        self.logger.info(f"⚡ {method_name} COMPLETED in {execution_time:.1f}s")
        self.logger.info(f"   📊 Signal Quality: {mtf_validated_count} MTF-validated + {traditional_count} traditional")
        self.logger.info(f"   🎯 MTF Validation Rate: {mtf_validation_rate:.1f}%")
        self.logger.info(f"   📈 Structure Filtering: {((len(symbols) - len(ranked_signals)) / len(symbols) * 100):.1f}% symbols filtered")
        
        return results

    def get_enhanced_mtf_distribution(self, signals: List[Dict]) -> Dict:
        """Get enhanced MTF status distribution including new categories"""
        try:
            mtf_validated = len([s for s in signals if s.get('mtf_validated', False)])
            strong_count = len([s for s in signals if s.get('mtf_status') == 'STRONG'])
            partial_count = len([s for s in signals if s.get('mtf_status') == 'PARTIAL'])
            none_count = len([s for s in signals if s.get('mtf_status') == 'NONE'])
            disabled_count = len([s for s in signals if s.get('mtf_status') == 'DISABLED'])
            total = len(signals)
            
            return {
                'mtf_validated_signals': mtf_validated,
                'strong_confirmation': strong_count,
                'partial_confirmation': partial_count,
                'no_confirmation': none_count,
                'disabled': disabled_count,
                'mtf_validation_rate': (mtf_validated / total * 100) if total > 0 else 0,
                'traditional_confirmation_rate': ((strong_count + partial_count) / total * 100) if total > 0 else 0,
                'structure_aware_percentage': (mtf_validated / total * 100) if total > 0 else 0
            }
        except Exception:
            return {}
    
    def generate_charts_for_top_signals(self, top_signals: List[Dict]) -> int:
        """Generate charts (HTML + Screenshot) for top-ranked signals only"""
        charts_generated = 0
        
        try:
            self.logger.debug(f"📊 Generating charts for TOP {len(top_signals)} signals only...")
            
            for i, signal in enumerate(top_signals):
                try:
                    symbol = signal['symbol']
                    mtf_status = signal.get('mtf_status', 'UNKNOWN')
                    confidence = signal.get('confidence', 0)
                    
                    self.logger.debug(f"   Generating chart {i+1}/{len(top_signals)}: {symbol} ({confidence}% conf, {mtf_status} MTF)")
                    
                    # Retrieve stored chart data
                    chart_data = signal.get('_chart_data', {})
                    if not chart_data:
                        self.logger.warning(f"   Missing chart data for {symbol}, skipping chart generation")
                        signal['chart_file'] = "Chart data unavailable"
                        continue
                    
                    df = chart_data.get('df')
                    symbol_data = chart_data.get('symbol_data')
                    analysis = signal.get('analysis', {})
                    
                    if df is None or df.empty:
                        self.logger.warning(f"   Invalid dataframe for {symbol}, skipping chart")
                        signal['chart_file'] = "Invalid chart data"
                        continue
                    
                    # Generate comprehensive chart (HTML + Screenshot)
                    chart_file = self.chart_generator.create_comprehensive_chart(
                        symbol=symbol,
                        df=df,
                        signal_data=signal,
                        volume_profile=analysis.get('volume_profile', {}),
                        fibonacci_data=analysis.get('fibonacci_data', {}),
                        confluence_zones=analysis.get('confluence_zones', [])
                    )
                    
                    # Update signal with chart file path
                    signal['chart_file'] = chart_file
                    charts_generated += 1
                    
                    self.logger.debug(f"   ✅ Chart {i+1}: {symbol} - {chart_file}")
                    
                    # Clean up stored chart data to save memory
                    if '_chart_data' in signal:
                        del signal['_chart_data']
                    
                except Exception as e:
                    self.logger.error(f"   Error generating chart for {signal.get('symbol', 'unknown')}: {e}")
                    signal['chart_file'] = f"Chart generation failed: {str(e)}"
            
            self.logger.info(f"✅ Chart generation completed")
            return charts_generated
            
        except Exception as e:
            self.logger.error(f"Chart generation process failed: {e}")
            return charts_generated

    def generate_top_opportunities_summary(self, opportunities: List[Dict]) -> Dict:
        """Generate summary specifically for TOP OPPORTUNITIES"""
        try:
            if not opportunities:
                return {
                    'status': 'no_opportunities',
                    'message': '📭 No top opportunities found in current market conditions',
                    'recommendation': 'Wait for better setups or lower position sizes'
                }
            
            total_ops = len(opportunities)
            
            # Categorize by confidence levels
            premium_ops = len([op for op in opportunities if op['confidence'] >= 65])
            quality_ops = len([op for op in opportunities if 55 <= op['confidence'] < 65])
            decent_ops = len([op for op in opportunities if 45 <= op['confidence'] < 55])
            
            # Categorize by R/R
            excellent_rr = len([op for op in opportunities if op.get('risk_reward_ratio', 0) >= 2.5])
            good_rr = len([op for op in opportunities if 2.0 <= op.get('risk_reward_ratio', 0) < 2.5])
            
            # Side distribution
            buy_signals = len([op for op in opportunities if op['side'] == 'buy'])
            sell_signals = len([op for op in opportunities if op['side'] == 'sell'])
            
            # MTF confirmation
            mtf_confirmed = len([op for op in opportunities if op.get('mtf_status') in ['STRONG', 'PARTIAL']])
            
            # Calculate averages
            avg_confidence = sum(op['confidence'] for op in opportunities) / total_ops
            avg_rr = sum(op.get('risk_reward_ratio', 0) for op in opportunities) / total_ops
            
            # Market recommendation
            if premium_ops >= 2:
                market_status = 'excellent'
                recommendation = f"🚀 EXCELLENT: {premium_ops} premium opportunities available - consider larger position sizes"
            elif quality_ops >= 2:
                market_status = 'good'
                recommendation = f"✅ GOOD: {quality_ops} quality opportunities - standard position sizing recommended"
            elif total_ops >= 3:
                market_status = 'fair'
                recommendation = f"⚠️ FAIR: {total_ops} opportunities available - use smaller position sizes"
            else:
                market_status = 'limited'
                recommendation = f"📉 LIMITED: Only {total_ops} opportunity(ies) - wait for better setups"
            
            return {
                'status': market_status,
                'total_opportunities': total_ops,
                'distribution': {
                    'premium': premium_ops,
                    'quality': quality_ops, 
                    'decent': decent_ops,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals
                },
                'quality_metrics': {
                    'avg_confidence': round(avg_confidence, 1),
                    'avg_risk_reward': round(avg_rr, 2),
                    'excellent_rr_count': excellent_rr,
                    'good_rr_count': good_rr,
                    'mtf_confirmed': mtf_confirmed,
                    'mtf_confirmation_rate': round((mtf_confirmed / total_ops) * 100, 1) if total_ops > 0 else 0
                },
                'recommendation': recommendation,
                'message': f'🎯 {total_ops} TOP OPPORTUNITIES selected from market scan'
            }
            
        except Exception as e:
            self.logger.error(f"Top opportunities summary error: {e}")
            return {
                'status': 'error',
                'message': f'Error generating summary: {str(e)}',
                'total_opportunities': len(opportunities)
            }

    def create_market_summary_with_mtf(self, symbols: List[Dict], signals: List[Dict]) -> Dict:
        """Create market summary with MTF statistics"""
        try:
            base_summary = self.create_market_summary(symbols, signals)
            
            # Add MTF-specific statistics
            if signals:
                mtf_strong_signals = [s for s in signals if s.get('mtf_status') == 'STRONG']
                mtf_partial_signals = [s for s in signals if s.get('mtf_status') == 'PARTIAL']
                mtf_none_signals = [s for s in signals if s.get('mtf_status') == 'NONE']
                
                mtf_summary = {
                    'mtf_enabled': self.config.mtf_confirmation_required,
                    'primary_timeframe': self.config.timeframe,
                    'confirmation_timeframes': self.config.confirmation_timeframes,
                    'mtf_distribution': {
                        'strong_confirmation': len(mtf_strong_signals),
                        'partial_confirmation': len(mtf_partial_signals),
                        'no_confirmation': len(mtf_none_signals),
                        'disabled': len([s for s in signals if s.get('mtf_status') == 'DISABLED'])
                    },
                    'mtf_performance': {
                        'avg_mtf_boost': np.mean([s['confidence'] - s.get('original_confidence', s['confidence']) for s in signals]),
                        'max_mtf_boost': max([s['confidence'] - s.get('original_confidence', s['confidence']) for s in signals]) if signals else 0,
                        'signals_with_boost': len([s for s in signals if s['confidence'] > s.get('original_confidence', s['confidence'])])
                    }
                }
                
                base_summary['mtf_analysis'] = mtf_summary
            
            return base_summary
            
        except Exception as e:
            self.logger.error(f"MTF market summary error: {e}")
            return self.create_market_summary(symbols, signals)
    
    def create_market_summary(self, symbols: List[Dict], signals: List[Dict]) -> Dict:
        """Create market overview summary"""
        try:
            # Volume analysis
            total_volume = sum(s['volume_24h'] for s in symbols)
            avg_volume = total_volume / len(symbols) if symbols else 0
            
            # Price change analysis
            price_changes = [s['price_change_24h'] for s in symbols]
            bullish_count = sum(1 for change in price_changes if change > 0)
            bearish_count = len(price_changes) - bullish_count
            
            # Signal analysis
            buy_signals = [s for s in signals if s['side'] == 'buy']
            sell_signals = [s for s in signals if s['side'] == 'sell']
            
            market_orders = [s for s in signals if s['order_type'] == 'market']
            limit_orders = [s for s in signals if s['order_type'] == 'limit']
            
            return {
                'total_market_volume': total_volume,
                'average_volume': avg_volume,
                'market_sentiment': {
                    'bullish_symbols': bullish_count,
                    'bearish_symbols': bearish_count,
                    'bullish_percentage': bullish_count / len(symbols) * 100 if symbols else 0
                },
                'signal_distribution': {
                    'buy_signals': len(buy_signals),
                    'sell_signals': len(sell_signals),
                    'market_orders': len(market_orders),
                    'limit_orders': len(limit_orders)
                },
                'top_movers': {
                    'biggest_gainer': max(symbols, key=lambda x: x['price_change_24h']) if symbols else None,
                    'biggest_loser': min(symbols, key=lambda x: x['price_change_24h']) if symbols else None,
                    'highest_volume': max(symbols, key=lambda x: x['volume_24h']) if symbols else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market summary error: {e}")
            return {}
    
    def get_order_type_distribution(self, signals: List[Dict]) -> Dict:
        """Get distribution of order types"""
        try:
            market_count = sum(1 for s in signals if s['order_type'] == 'market')
            limit_count = sum(1 for s in signals if s['order_type'] == 'limit')
            total = len(signals)
            
            return {
                'market_orders': market_count,
                'limit_orders': limit_count,
                'market_percentage': market_count / total * 100 if total > 0 else 0,
                'limit_percentage': limit_count / total * 100 if total > 0 else 0
            }
        except Exception:
            return {}
    
    def get_mtf_status_distribution(self, signals: List[Dict]) -> Dict:
        """Get distribution of MTF confirmation statuses"""
        try:
            strong_count = len([s for s in signals if s.get('mtf_status') == 'STRONG'])
            partial_count = len([s for s in signals if s.get('mtf_status') == 'PARTIAL'])
            none_count = len([s for s in signals if s.get('mtf_status') == 'NONE'])
            disabled_count = len([s for s in signals if s.get('mtf_status') == 'DISABLED'])
            total = len(signals)
            
            return {
                'strong_confirmation': strong_count,
                'partial_confirmation': partial_count,
                'no_confirmation': none_count,
                'disabled': disabled_count,
                'strong_percentage': strong_count / total * 100 if total > 0 else 0,
                'partial_percentage': partial_count / total * 100 if total > 0 else 0,
                'confirmation_rate': (strong_count + partial_count) / total * 100 if total > 0 else 0
            }
        except Exception:
            return {}
    
    def count_mtf_confirmations(self, signals: List[Dict]) -> str:
        """Count MTF confirmations for logging"""
        try:
            strong = len([s for s in signals if s.get('mtf_status') == 'STRONG'])
            partial = len([s for s in signals if s.get('mtf_status') == 'PARTIAL'])
            none = len([s for s in signals if s.get('mtf_status') == 'NONE'])
            return f"Strong: {strong}, Partial: {partial}, None: {none}"
        except Exception:
            return "Unknown"
    
    def calculate_speedup_factor(self, parallel_time: float, symbol_count: int) -> float:
        """Calculate theoretical speedup factor"""
        try:
            estimated_sequential_time = symbol_count * 4.0  # MTF analysis takes longer
            speedup = estimated_sequential_time / parallel_time if parallel_time > 0 else 1
            return min(speedup, self.config.max_workers)
        except Exception:
            return 1.0
    
    def print_comprehensive_results_with_mtf(self, results: Dict):
        """
        ENHANCED: Comprehensive display with all necessary signal information
        Shows detailed MTF analysis, signal quality, risk metrics, and execution readiness
        """
        
        if not results:
            self.logger.debug("❌ No results to display")
            return
        
        scan_info = results['scan_info']
        opportunities = results['top_opportunities']
        
        # Ensure MTF display data exists
        opportunities = self._ensure_mtf_display_data(opportunities)
        
        # Enhanced scan header with comprehensive info
        timeframe_display = f"{self.config.timeframe}→{'/'.join(self.config.confirmation_timeframes)}"
        execution_time = scan_info.get('execution_time_seconds', 0)
        method = scan_info.get('method', 'unknown')
        
        self.logger.debug("=" * 200)
        self.logger.debug(f"\n🏆 ENHANCED TRADING OPPORTUNITIES ANALYSIS ({timeframe_display} MTF)")
        self.logger.debug(f"📊 Method: {method.upper()} | ⏱️  Execution: {execution_time:.1f}s | 🎯 Found: {len(opportunities)} signals")
        self.logger.debug("=" * 200)
        
        if not opportunities:
            print("   No trading opportunities found")
            return
        
        # Display detailed opportunities table
        tp1_marker = "TP1 ✅" if self.config.default_tp_level == 'take_profit_1' else "TP1"
        tp2_marker = "TP2 ✅" if self.config.default_tp_level != 'take_profit_1' else "TP2"
        
        # Enhanced header with all critical information
        header = (
            f"{'#':<1} | {'Symbol':<8} | {'Side':<9} | {'Entry':<10} | {'SL':<10} | "
            f"{tp1_marker:<10} | {tp2_marker:<10} | {'R/R':<4} | {'Conf':<8} | "
            f"{'Qua':<5} | {'MTF':<15} | {'Strategy':<18} | "
            f"{'✅ TF':<10} | {'❌ TF':<10} | "
            f"{'Vol':<8} | {'Regime':<12} | {'Premium':<7} | {'Execute':<10}"
        )
        print(header)
        print("-" * 200)
        
        # Track statistics for summary
        total_confirmed_tfs = 0
        total_conflicting_tfs = 0
        total_neutral_tfs = 0
        premium_signals = 0
        mtf_validated_signals = 0
        high_confidence_signals = 0
        
        for i, opp in enumerate(opportunities):
            try:
                # Basic signal info
                symbol = opp['symbol'].split('/')[0]
                side = opp['side']
                entry_price = opp['entry_price']
                stop_loss = opp['stop_loss']
                take_profit_1 = opp.get('take_profit_1', 0)
                take_profit_2 = opp.get('take_profit_2', 0)
                risk_reward = opp.get('risk_reward_ratio', 0)
                confidence = opp.get('confidence', 0)
                volume_24h = opp.get('volume_24h', 0)
                
                # Execution status
                will_execute = i < self.config.max_execution_per_trade
                execution_status = "🎯 EXEC" if will_execute else "⏭️ SKIP"
                
                # Side formatting with emoji
                if side == 'buy':
                    side_display = "🟢 LONG"
                else:
                    side_display = "🔴 SHORT"
                
                # Quality assessment
                quality_grade = opp.get('quality_grade', 'C')
                if quality_grade in ['A+', 'A']:
                    quality_display = f"⭐ {quality_grade}"
                    premium_signals += 1
                elif quality_grade in ['A-', 'B+']:
                    quality_display = f"✅ {quality_grade}"
                elif quality_grade in ['B', 'B-']:
                    quality_display = f"🔸 {quality_grade}"
                else:
                    quality_display = f"⚪ {quality_grade}"
                
                # MTF status with enhanced display
                mtf_status = opp.get('mtf_status', 'UNKNOWN')
                mtf_validated = opp.get('mtf_validated', False)
                
                if mtf_validated:
                    mtf_display = "🎯 VALIDATED"
                    mtf_validated_signals += 1
                elif mtf_status == 'STRONG':
                    mtf_display = "⭐ STRONG"
                elif mtf_status == 'PARTIAL':
                    mtf_display = "🔸 PARTIAL"
                elif mtf_status == 'NONE':
                    mtf_display = "⚪ NONE"
                else:
                    mtf_display = "❌ DISABLED"
                
                # Confidence with MTF boost indication
                if confidence >= 70:
                    high_confidence_signals += 1
                    
                mtf_boost = opp.get('mtf_boost', 0)
                if mtf_boost > 0:
                    conf_display = f"{confidence:.0f}% (+{mtf_boost:.0f})"
                else:
                    conf_display = f"{confidence:.0f}%"
                
                # Entry strategy formatting
                entry_strategy = opp.get('entry_strategy', 'immediate')
                strategy_display = entry_strategy.replace('_', ' ').title()
                if len(strategy_display) > 16:
                    strategy_display = strategy_display[:14] + ".."
                
                # MTF timeframe analysis
                mtf_analysis = opp.get('mtf_analysis', {})
                confirmed_tfs = mtf_analysis.get('confirmed_timeframes', [])
                conflicting_tfs = mtf_analysis.get('conflicting_timeframes', [])
                neutral_tfs = mtf_analysis.get('neutral_timeframes', [])
                
                # Update statistics
                total_confirmed_tfs += len(confirmed_tfs)
                total_conflicting_tfs += len(conflicting_tfs)
                total_neutral_tfs += len(neutral_tfs)
                
                # Format timeframe displays with enhanced indicators
                def format_timeframes(tfs_list, max_length=18):
                    if not tfs_list:
                        return "None"
                    
                    # Sort timeframes for consistent display
                    sorted_tfs = sorted(tfs_list, key=lambda x: x.replace('*', ''))
                    
                    # Add strength indicators
                    formatted_tfs = []
                    for tf in sorted_tfs:
                        if '*' in tf:  # Structure timeframe
                            formatted_tfs.append(f"{tf}")  # Keep the *
                        else:
                            formatted_tfs.append(tf)
                    
                    result = ', '.join(formatted_tfs)
                    return result[:max_length-2] + ".." if len(result) > max_length else result
                
                confirmed_display = format_timeframes(confirmed_tfs, 18)
                conflicting_display = format_timeframes(conflicting_tfs, 13)
                neutral_display = format_timeframes(neutral_tfs, 13)
                
                # Add confirmation strength indicator
                if len(confirmed_tfs) >= 2:
                    confirmed_display = f"💪 {confirmed_display}"
                elif len(confirmed_tfs) == 1:
                    confirmed_display = f"✅ {confirmed_display}"
                
                # Add conflict warning
                if len(conflicting_tfs) >= 2:
                    conflicting_display = f"⚠️  {conflicting_display}"
                elif len(conflicting_tfs) == 1:
                    conflicting_display = f"❌ {conflicting_display}"
                
                # Volume display
                if volume_24h >= 10_000_000:
                    volume_display = f"{volume_24h/1e6:.0f}M"
                elif volume_24h >= 1_000_000:
                    volume_display = f"{volume_24h/1e6:.1f}M"
                else:
                    volume_display = f"{volume_24h/1e3:.0f}K"
                
                # Market regime with compatibility
                market_regime = opp.get('market_regime', 'unknown')
                regime_compatibility = opp.get('regime_compatibility', 'medium')
                
                if regime_compatibility == 'high':
                    regime_display = f"✅ {market_regime[:8]}"
                elif regime_compatibility == 'medium':
                    regime_display = f"🔸 {market_regime[:8]}"
                else:
                    regime_display = f"⚠️  {market_regime[:8]}"

                is_premium_signal = '✅' if opp['is_premium_signal'] else '❌'
                
                # Construct the row
                row = (
                    f"{i+1:<1} | {symbol:<8} | {side_display:<8} | "
                    f"{entry_price:<10.6f} | {stop_loss:<10.6f} | "
                    f"{take_profit_1:<10.6f} | {take_profit_2:<10.6f} | "
                    f"{risk_reward:<4.2f} | {conf_display:<8} | "
                    f"{quality_display:<5} | {mtf_display:<15} | {strategy_display:<18} | "
                    f"{confirmed_display:<10} | {conflicting_display:<10} | "
                    f"{volume_display:<8} | {regime_display:<12} | {is_premium_signal:<7} | {execution_status:<10}"
                )
                print(row)
                
            except Exception as e:
                error_row = (
                    f"{i+1:<3} | {opp.get('symbol', 'ERROR'):<15} | "
                    f"ERROR: {str(e)[:150]}"
                )
                print(error_row)
        
        print("-" * 200)
        
        # ENHANCED COMPREHENSIVE SUMMARY
        self.logger.debug(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY:")
        self.logger.debug("=" * 60)
        
        # Signal Quality Distribution
        self.logger.debug(f"🎯 SIGNAL QUALITY:")
        self.logger.debug(f"   Premium Signals (A+/A): {premium_signals}")
        self.logger.debug(f"   MTF Validated: {mtf_validated_signals}")
        self.logger.debug(f"   High Confidence (≥70%): {high_confidence_signals}")
        self.logger.debug(f"   Total Opportunities: {len(opportunities)}")
        
        # MTF Analysis Summary
        self.logger.debug(f"\n🔍 MULTI-TIMEFRAME ANALYSIS:")
        avg_confirmed = total_confirmed_tfs / len(opportunities) if opportunities else 0
        avg_conflicting = total_conflicting_tfs / len(opportunities) if opportunities else 0
        confirmation_ratio = total_confirmed_tfs / max(1, total_confirmed_tfs + total_conflicting_tfs)
        
        self.logger.debug(f"   Total Confirming Timeframes: {total_confirmed_tfs}")
        self.logger.debug(f"   Total Conflicting Timeframes: {total_conflicting_tfs}")
        self.logger.debug(f"   Total Neutral Timeframes: {total_neutral_tfs}")
        self.logger.debug(f"   Average Confirmations per Signal: {avg_confirmed:.1f}")
        self.logger.debug(f"   Confirmation Ratio: {confirmation_ratio:.1%}")
        
        # Market Regime Analysis
        regime_distribution = {}
        for opp in opportunities:
            regime = opp.get('market_regime', 'unknown')
            regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
        
        if regime_distribution:
            self.logger.debug(f"\n🌍 MARKET REGIME DISTRIBUTION:")
            for regime, count in regime_distribution.items():
                percentage = (count / len(opportunities)) * 100
                self.logger.debug(f"   {regime.title()}: {count} signals ({percentage:.1f}%)")
        
        # Risk/Reward Analysis
        if opportunities:
            rr_ratios = [opp.get('risk_reward_ratio', 0) for opp in opportunities]
            avg_rr = sum(rr_ratios) / len(rr_ratios)
            excellent_rr = len([r for r in rr_ratios if r >= 3.0])
            good_rr = len([r for r in rr_ratios if 2.3 <= r < 3.0])
            
            self.logger.debug(f"\n💰 RISK/REWARD ANALYSIS:")
            self.logger.debug(f"   Average R/R Ratio: {avg_rr:.2f}")
            self.logger.debug(f"   Excellent R/R (≥3.0): {excellent_rr}")
            self.logger.debug(f"   Good R/R (≥2.3): {good_rr}")
        
        # Execution Plan
        execution_count = min(len(opportunities), self.config.max_execution_per_trade)
        skipped_count = len(opportunities) - execution_count
        
        self.logger.debug(f"\n🎯 EXECUTION PLAN:")
        self.logger.debug(f"   Will Execute: {execution_count} trades")
        if skipped_count > 0:
            self.logger.debug(f"   Will Skip: {skipped_count} opportunities (position limits)")
        self.logger.debug(f"   Max Per Scan: {self.config.max_execution_per_trade}")
        self.logger.debug(f"   Auto-Execute: {'✅ Enabled' if self.config.auto_execute_trades else '❌ Disabled'}")
        
        # Trading Recommendations
        self.logger.debug(f"\n💡 TRADING RECOMMENDATIONS:")
        
        if premium_signals >= 2:
            self.logger.debug("   🚀 EXCELLENT CONDITIONS: Multiple premium signals available")
            self.logger.debug("      → Consider standard to increased position sizing")
        elif mtf_validated_signals >= 2:
            self.logger.debug("   ✅ GOOD CONDITIONS: Multiple MTF-validated signals")
            self.logger.debug("      → Standard position sizing recommended")
        elif high_confidence_signals >= 1:
            self.logger.debug("   🔸 MODERATE CONDITIONS: High-confidence signals available")
            self.logger.debug("      → Use cautious position sizing")
        else:
            self.logger.debug("   ⚠️  LIMITED CONDITIONS: Lower quality signals")
            self.logger.debug("      → Consider waiting for better setups")
        
        # MTF Health Check
        if confirmation_ratio > 0.7:
            self.logger.debug("   📈 MTF HEALTH: Strong timeframe alignment")
        elif confirmation_ratio > 0.5:
            self.logger.debug("   🔸 MTF HEALTH: Moderate timeframe alignment")
        else:
            self.logger.debug("   ⚠️  MTF HEALTH: Weak timeframe alignment - exercise caution")
        
        # System Performance
        system_perf = results.get('system_performance', {})
        signals_per_min = system_perf.get('signals_per_minute', 0)
        mtf_validation_rate = system_perf.get('mtf_validation_rate', 0)
        structure_filtered_rate = system_perf.get('structure_filtered_rate', 0)
        
        self.logger.debug(f"\n⚡ SYSTEM PERFORMANCE:")
        self.logger.debug(f"   Analysis Speed: {signals_per_min:.1f} signals/minute")
        self.logger.debug(f"   MTF Validation Rate: {mtf_validation_rate:.1f}%")
        self.logger.debug(f"   Structure Filter Rate: {structure_filtered_rate:.1f}%")
        self.logger.debug(f"   Primary Timeframe: {self.config.timeframe}")
        self.logger.debug(f"   Structure Timeframe: {self.config.confirmation_timeframes[-1] if self.config.confirmation_timeframes else 'N/A'}")
        
        # Legend for symbols
        self.logger.debug(f"\n🔤 LEGEND:")
        self.logger.debug("   * = Structure timeframe")
        self.logger.debug("   💪 = Strong confirmation (≥2 TFs)")
        self.logger.debug("   ✅ = Single confirmation")
        self.logger.debug("   ⚠️  = Multiple conflicts")
        self.logger.debug("   ❌ = Single conflict")
        self.logger.debug("   🎯 = Will execute")
        self.logger.debug("   ⏭️  = Will skip (position limits)")
        
        self.logger.debug("=" * 200)

    def _ensure_mtf_display_data(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Enhanced fallback to ensure MTF display data exists for all opportunities
        """
        try:
            for opp in opportunities:
                if 'mtf_analysis' not in opp:
                    opp['mtf_analysis'] = {}
                
                mtf_analysis = opp['mtf_analysis']
                
                # If display data is missing, try to reconstruct it
                if not mtf_analysis.get('confirmed_timeframes'):
                    symbol = opp['symbol']
                    side = opp['side']
                    
                    # Try to get MTF data from analysis_details
                    analysis_details = opp.get('analysis_details', {})
                    mtf_context = analysis_details.get('mtf_context')
                    
                    if mtf_context:
                        # Reconstruct timeframe categorization based on MTF context
                        confirmed_tfs = []
                        conflicting_tfs = []
                        neutral_tfs = []
                        
                        # Use confirmation score and trend alignment
                        confirmation_score = mtf_context.get('confirmation_score', 0.5)
                        dominant_trend = mtf_context.get('dominant_trend', 'neutral')
                        entry_bias = mtf_context.get('entry_bias', 'neutral')
                        
                        # Distribute timeframes based on signal strength
                        for i, tf in enumerate(self.config.confirmation_timeframes):
                            # Use position in timeframe list to determine strength
                            tf_strength = (len(self.config.confirmation_timeframes) - i) / len(self.config.confirmation_timeframes)
                            
                            if side == 'buy':
                                if (('bullish' in dominant_trend and confirmation_score > 0.6) or 
                                    (entry_bias == 'long_favored' and tf_strength > 0.6)):
                                    confirmed_tfs.append(tf)
                                elif ('bearish' in dominant_trend and confirmation_score > 0.6):
                                    conflicting_tfs.append(tf)
                                else:
                                    neutral_tfs.append(tf)
                            else:  # sell
                                if (('bearish' in dominant_trend and confirmation_score > 0.6) or 
                                    (entry_bias == 'short_favored' and tf_strength > 0.6)):
                                    confirmed_tfs.append(tf)
                                elif ('bullish' in dominant_trend and confirmation_score > 0.6):
                                    conflicting_tfs.append(tf)
                                else:
                                    neutral_tfs.append(tf)
                        
                        # Add structure timeframe marker
                        structure_tf = mtf_context.get('structure_timeframe', self.config.confirmation_timeframes[-1] if self.config.confirmation_timeframes else '6h')
                        if structure_tf in confirmed_tfs:
                            confirmed_tfs[confirmed_tfs.index(structure_tf)] = f"{structure_tf}*"
                        elif structure_tf in conflicting_tfs:
                            conflicting_tfs[conflicting_tfs.index(structure_tf)] = f"{structure_tf}*"
                        elif structure_tf in neutral_tfs:
                            neutral_tfs[neutral_tfs.index(structure_tf)] = f"{structure_tf}*"
                        
                        mtf_analysis.update({
                            'confirmed_timeframes': confirmed_tfs,
                            'conflicting_timeframes': conflicting_tfs,
                            'neutral_timeframes': neutral_tfs,
                            'fallback_method': 'mtf_context_reconstruction'
                        })
                    else:
                        # Final fallback: distribute based on MTF status
                        mtf_status = opp.get('mtf_status', 'NONE')
                        
                        if mtf_status in ['STRONG', 'MTF_VALIDATED']:
                            # Most timeframes confirm
                            confirmed_count = int(len(self.config.confirmation_timeframes) * 0.8)
                            mtf_analysis['confirmed_timeframes'] = self.config.confirmation_timeframes[:confirmed_count]
                            mtf_analysis['conflicting_timeframes'] = []
                            mtf_analysis['neutral_timeframes'] = self.config.confirmation_timeframes[confirmed_count:]
                        elif mtf_status == 'PARTIAL':
                            # Split timeframes
                            half = len(self.config.confirmation_timeframes) // 2
                            mtf_analysis['confirmed_timeframes'] = self.config.confirmation_timeframes[:half]
                            mtf_analysis['conflicting_timeframes'] = []
                            mtf_analysis['neutral_timeframes'] = self.config.confirmation_timeframes[half:]
                        else:
                            # Most timeframes are neutral or conflicting
                            mtf_analysis['confirmed_timeframes'] = []
                            mtf_analysis['conflicting_timeframes'] = self.config.confirmation_timeframes[:1]  # One conflict
                            mtf_analysis['neutral_timeframes'] = self.config.confirmation_timeframes[1:]
                        
                        # Add structure marker to the last timeframe (highest)
                        if self.config.confirmation_timeframes:
                            structure_tf = self.config.confirmation_timeframes[-1]
                            for key in ['confirmed_timeframes', 'conflicting_timeframes', 'neutral_timeframes']:
                                if structure_tf in mtf_analysis[key]:
                                    idx = mtf_analysis[key].index(structure_tf)
                                    mtf_analysis[key][idx] = f"{structure_tf}*"
                                    break
                        
                        mtf_analysis['fallback_method'] = 'mtf_status_distribution'
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error ensuring MTF display data: {e}")
            return opportunities
        
    def save_results_to_database(self, results: Dict) -> Dict[str, Any]:
        """Save TOP OPPORTUNITIES to MySQL database (replaces CSV export)"""
        try:
            self.logger.info("💾 Saving TOP OPPORTUNITIES to MySQL database...")
            
            # Only save top opportunities to database - not all signals
            database_results = {
                'scan_info': results['scan_info'],
                'signals': results['top_opportunities'],  # ONLY top opportunities
                'analysis_results': results['analysis_results'],
                'market_summary': results['market_summary'],
                'system_performance': results['system_performance']
            }
            
            save_result = self.enhanced_db_manager.save_all_results(database_results)
            
            if save_result.get('error'):
                self.logger.error(f"Failed to save top opportunities: {save_result['error']}")
            else:
                saved_count = save_result.get('signals', 0)
                scan_id = save_result.get('scan_id', 'Unknown')
                self.logger.info(f"✅ Saved {saved_count} TOP OPPORTUNITIES to database (Scan ID: {scan_id})")
                self.logger.info(f"🎯 Database Focus: Quality opportunities only, not all signals")
            
            return save_result
            
        except Exception as e:
            self.logger.error(f"Error saving results to database: {e}")
            return {'error': str(e), 'signals': 0}

    def get_top_opportunities_for_trading(self, results: Dict) -> List[Dict]:
        """Get top opportunities ready for trading - this is what autotrader should use"""
        try:
            top_opportunities = results.get('top_opportunities', [])
            
            # Additional validation for trading
            trading_ready = []
            
            for opp in top_opportunities:
                # Ensure all required fields exist
                required_fields = ['symbol', 'side', 'entry_price', 'stop_loss', 'take_profit_1', 
                                 'confidence', 'risk_reward_ratio', 'volume_24h']
                
                if all(field in opp for field in required_fields):
                    # Add trading-specific metadata
                    opp['trading_ready'] = True
                    opp['selection_rank'] = len(trading_ready) + 1
                    opp['quality_tier'] = self.get_quality_tier(opp)
                    trading_ready.append(opp)
            
            self.logger.info(f"🎯 {len(trading_ready)} TOP OPPORTUNITIES ready for trading")
            return trading_ready
            
        except Exception as e:
            self.logger.error(f"Error preparing opportunities for trading: {e}")
            return []
    
    def get_quality_tier(self, opportunity: Dict) -> str:
        """Determine quality tier of opportunity"""
        try:
            confidence = opportunity.get('confidence', 0)
            rr_ratio = opportunity.get('risk_reward_ratio', 0)
            volume_24h = opportunity.get('volume_24h', 0)
            mtf_status = opportunity.get('mtf_status', 'NONE')
            
            if (confidence >= 65 and rr_ratio >= 2.5 and volume_24h >= 2_000_000 and 
                mtf_status in ['STRONG', 'PARTIAL']):
                return 'PREMIUM'
            elif confidence >= 55 and rr_ratio >= 2.0 and volume_24h >= 1_000_000:
                return 'QUALITY'
            elif confidence >= 45 and rr_ratio >= 1.8 and volume_24h >= 500_000:
                return 'DECENT'
            else:
                return 'BASIC'
                
        except Exception:
            return 'UNKNOWN'