"""
Complete Enhanced Bybit System Integration with Multi-Timeframe Analysis and MySQL Database.
Updated to use database storage instead of CSV export.
Primary timeframe: 15m, Confirmation timeframes: 1h, 4h
"""

import pandas as pd
import numpy as np
import time
import logging
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
from analysis.multi_timeframe import MultiTimeframeAnalyzer
from signals.generator import SignalGenerator
from visualization.charts import InteractiveChartGenerator
from utils.database_manager import EnhancedDatabaseManager
from utils.logging import setup_logging
from database.models import DatabaseManager


class CompleteEnhancedBybitSystem:
    """Complete enhanced Bybit system with multi-timeframe confirmation and MySQL database storage"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = setup_logging()
        
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
        self.signal_generator = SignalGenerator(config)
        self.chart_generator = InteractiveChartGenerator(config)
        
        # Initialize multi-timeframe analyzer
        if self.exchange_manager.exchange:
            self.mtf_analyzer = MultiTimeframeAnalyzer(self.exchange_manager.exchange, config)
        else:
            self.mtf_analyzer = None
        
        # System state
        self.signal_history = []
        self.analysis_lock = threading.Lock()
        self.results_queue = Queue()
        self.processed_count = 0
        
        self.logger.info("🚀 COMPLETE ENHANCED SYSTEM WITH MULTI-TIMEFRAME & MYSQL DATABASE INITIALIZED!")
        self.logger.debug("✅ Enhanced Technical Analysis (FIXED Ichimoku & Stochastic RSI)")
        self.logger.debug("✅ Volume Profile Analysis") 
        self.logger.debug("✅ Fibonacci & Confluence Analysis")
        self.logger.debug("✅ Interactive Chart Generation")
        self.logger.debug("✅ MySQL Database Storage (Replaces CSV)")
        if self.config.mtf_confirmation_required:
            self.logger.debug(f"✅ Multi-Timeframe Confirmation: Primary 15m, Confirmation 1h/4h")
        self.logger.debug(f"✅ Database: {self.config.db_config.database} @ {self.config.db_config.host}")
    
    def analyze_symbol_complete_with_mtf(self, symbol_data: Dict) -> Optional[Dict]:
        """Complete analysis with multi-timeframe confirmation"""
        try:
            symbol = symbol_data['symbol']
            self.logger.debug(f"📊 Analyzing {symbol} (15m → 1h/4h MTF)...")
            
            # Fetch primary timeframe data (configurable)
            df = self.exchange_manager.fetch_ohlcv_data(symbol, self.config.timeframe)
            if df.empty or len(df) < 50:
                self.logger.warning(f"Insufficient {self.config.timeframe} data for {symbol}")
                return None
            
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
            
            # Generate primary signal on 15m timeframe
            primary_signal = self.signal_generator.generate_enhanced_signal(
                df, symbol_data, volume_entry, confluence_zones
            )
            
            if primary_signal:
                # Store original confidence before MTF boost
                primary_signal['original_confidence'] = primary_signal['confidence']
                
                # MULTI-TIMEFRAME CONFIRMATION (configurable timeframes)
                if self.config.mtf_confirmation_required and self.mtf_analyzer:
                    confirmation_tfs = ', '.join(self.config.confirmation_timeframes)
                    self.logger.debug(f"🔍 Running {confirmation_tfs} confirmation for {symbol}...")
                    mtf_analysis = self.mtf_analyzer.analyze_symbol_multi_timeframe(symbol, primary_signal)
                    
                    # Add MTF data to signal
                    primary_signal['mtf_analysis'] = mtf_analysis
                    
                    # Boost confidence based on MTF confirmation
                    if mtf_analysis['mtf_confidence_boost'] > 0:
                        new_confidence = min(95, primary_signal['confidence'] + mtf_analysis['mtf_confidence_boost'])
                        self.logger.debug(f"   📈 Confidence boosted: {primary_signal['confidence']:.1f}% → {new_confidence:.1f}%")
                        primary_signal['confidence'] = new_confidence
                    
                    # Mark MTF status
                    confirmed_count = len(mtf_analysis['confirmed_timeframes'])
                    total_timeframes = len(self.config.confirmation_timeframes)
                    
                    if confirmed_count >= max(2, total_timeframes * 0.75):  # 75% confirmation threshold
                        primary_signal['mtf_status'] = 'STRONG'
                        primary_signal['priority_boost'] = 100
                    elif confirmed_count >= max(1, total_timeframes * 0.5):  # 50% confirmation threshold
                        primary_signal['mtf_status'] = 'PARTIAL'
                        primary_signal['priority_boost'] = 50
                    else:
                        primary_signal['mtf_status'] = 'NONE'
                        primary_signal['priority_boost'] = 0
                    
                    self.logger.debug(f"   📊 MTF Status: {primary_signal['mtf_status']} ({confirmed_count}/{total_timeframes})")
                else:
                    # No MTF analysis
                    primary_signal['mtf_analysis'] = {
                        'confirmed_timeframes': [],
                        'conflicting_timeframes': [],
                        'neutral_timeframes': [],
                        'confirmation_strength': 0.0,
                        'mtf_confidence_boost': 0.0,
                        'timeframe_signals': {}
                    }
                    primary_signal['mtf_status'] = 'DISABLED'
                    primary_signal['priority_boost'] = 0
                
                # APPROACH 4: Generate Interactive Chart
                chart_file = self.chart_generator.create_comprehensive_chart(
                    symbol, df, primary_signal, volume_profile, fibonacci_data, confluence_zones
                )
                
                primary_signal['chart_file'] = chart_file
                
                # Add comprehensive analysis data
                primary_signal['analysis'] = {
                    'volume_profile': volume_profile,
                    'fibonacci_data': fibonacci_data,
                    'confluence_zones': confluence_zones,
                    'technical_summary': self.signal_generator.create_technical_summary(df),
                    'risk_assessment': self.signal_generator.assess_signal_risk(primary_signal, df)
                }
                
                self.logger.debug(f"✅ {symbol} - {primary_signal['side'].upper()} signal generated ({self.config.timeframe} base)")
                return primary_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"MTF analysis failed for {symbol_data.get('symbol', 'unknown')}: {e}")
            return None
    
    
    def analyze_symbol_thread_safe_mtf(self, symbol_data: Dict, thread_id: int) -> Optional[Dict]:
        """Thread-safe version of MTF symbol analysis"""
        try:
            symbol = symbol_data['symbol']
            
            # Thread-safe logging
            with self.analysis_lock:
                self.processed_count += 1
                confirmation_tfs = '→'.join([self.config.timeframe] + self.config.confirmation_timeframes)
                self.logger.debug(f"[Thread-{thread_id}] [{self.processed_count}] Analyzing {symbol} with {confirmation_tfs} MTF...")
            
            # Call the MTF analysis method
            result = self.analyze_symbol_complete_with_mtf(symbol_data)
            
            if result:
                with self.analysis_lock:
                    mtf_status = result.get('mtf_status', 'UNKNOWN')
                    self.logger.debug(f"[Thread-{thread_id}] ✅ {symbol} - {result['side'].upper()} signal (MTF: {mtf_status})")
            
            return result
            
        except Exception as e:
            with self.analysis_lock:
                self.logger.error(f"[Thread-{thread_id}] Error analyzing {symbol_data.get('symbol', 'unknown')}: {e}")
            return None
    
    def run_complete_analysis_parallel_mtf(self) -> Dict:
        """Parallel analysis with multi-timeframe confirmation and MySQL storage"""
        start_time = time.time()
        
        timeframe_display = f"{self.config.timeframe}→{'/'.join(self.config.confirmation_timeframes)}"
        self.logger.info(f"🚀 STARTING ENHANCED ANALYSIS WITH {timeframe_display} MTF CONFIRMATION (PARALLEL + MYSQL)")
        self.logger.info("=" * 90)
        
        # Get top symbols
        symbols = self.exchange_manager.get_top_symbols()
        if not symbols:
            self.logger.error("No symbols found")
            return {}
        
        self.logger.info(f"📊 Analyzing {len(symbols)} symbols with MTF confirmation...")
        self.logger.info(f"   Primary Timeframe: {self.config.timeframe}")
        self.logger.info(f"   Confirmation Timeframes: {', '.join(self.config.confirmation_timeframes)}")
        self.logger.info(f"   MTF Weight Multiplier: {self.config.mtf_weight_multiplier}x")
        self.logger.info(f"   Database: MySQL @ {self.config.db_config.host}")
        
        # Initialize counters
        self.processed_count = 0
        all_signals = []
        analysis_results = []
        charts_generated = 0
        
        # Create thread pool and submit tasks
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_symbol = {}
            
            for i, symbol_data in enumerate(symbols):
                future = executor.submit(self.analyze_symbol_thread_safe_mtf, symbol_data, i % self.config.max_workers)
                future_to_symbol[future] = symbol_data
            
            # Process completed tasks - ANALYZE ALL SYMBOLS FIRST
            completed_tasks = 0
            for future in as_completed(future_to_symbol):
                symbol_data = future_to_symbol[future]
                completed_tasks += 1
                
                try:
                    result = future.result()
                    
                    if result:
                        with self.analysis_lock:
                            all_signals.append(result)
                        
                        analysis_results.append({
                            'symbol': symbol_data['symbol'],
                            'status': 'success',
                            'signal_generated': True,
                            'confidence': result['confidence'],
                            'original_confidence': result.get('original_confidence', result['confidence']),
                            'side': result['side'],
                            'mtf_status': result.get('mtf_status', 'UNKNOWN'),
                            'mtf_confirmed_count': len(result.get('mtf_analysis', {}).get('confirmed_timeframes', [])),
                            'mtf_total_timeframes': len(self.config.confirmation_timeframes)
                        })
                    else:
                        analysis_results.append({
                            'symbol': symbol_data['symbol'],
                            'status': 'no_signal',
                            'signal_generated': False
                        })
                    
                    # Progress update
                    progress = (completed_tasks / len(symbols)) * 100
                    self.logger.debug(f"📈 Progress: {completed_tasks}/{len(symbols)} ({progress:.1f}%) - Signals: {len(all_signals)}")
                    
                    # NO EARLY BREAK HERE - Let all symbols complete first
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol_data['symbol']}: {e}")
                    analysis_results.append({
                        'symbol': symbol_data['symbol'],
                        'status': 'error',
                        'error': str(e)
                    })

        execution_time = time.time() - start_time
        
        # NOW SORT ALL SIGNALS BY PROPER RANKING (MTF priority + confidence)
        self.logger.info(f"📊 Sorting {len(all_signals)} signals by MTF priority and confidence...")
        
        # Sort signals by MTF priority first, then confidence
        # Priority order: MTF status (STRONG > PARTIAL > NONE), then confidence
        def get_sort_key(signal):
            mtf_status = signal.get('mtf_status', 'NONE')
            confidence = signal.get('confidence', 0)
            priority_boost = signal.get('priority_boost', 0)
            
            # MTF status priority: STRONG=3, PARTIAL=2, NONE=1, DISABLED=0
            mtf_priority = {
                'STRONG': 3,
                'PARTIAL': 2, 
                'NONE': 1,
                'DISABLED': 0
            }.get(mtf_status, 0)
            
            # Return tuple for sorting: (MTF_priority, confidence, priority_boost)
            # Higher values = better ranking
            return (mtf_priority, confidence, priority_boost)
        
        # Sort in descending order (highest priority first)
        all_signals.sort(key=get_sort_key, reverse=True)
        
        # Log the top signals after sorting
        self.logger.info("🏆 Top 10 Signals After Ranking:")
        for i, signal in enumerate(all_signals[:10]):
            mtf_status = signal.get('mtf_status', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            symbol = signal.get('symbol', 'UNKNOWN')
            side = signal.get('side', 'UNKNOWN')
            self.logger.info(f"   {i+1}. {symbol} {side.upper()} - Confidence: {confidence}% - MTF: {mtf_status}")
        
        # AFTER sorting, generate charts for top signals only
        self.logger.info(f"📊 Generating charts for top {self.config.charts_per_batch} signals...")
        for i, signal in enumerate(all_signals[:self.config.charts_per_batch]):
            try:
                # Re-generate chart for this top signal if needed
                symbol = signal['symbol']
                if signal.get('chart_file', '').endswith('Chart generated'):
                    # Chart was not properly generated, create it now
                    symbol_data = {'symbol': symbol, 'current_price': signal['current_price'], 'volume_24h': signal['volume_24h']}
                    
                    # Fetch data again for charting
                    df = self.exchange_manager.fetch_ohlcv_data(symbol, self.config.timeframe)
                    if not df.empty:
                        df = self.enhanced_ta.calculate_all_indicators(df, self.config)
                        volume_profile = self.volume_analyzer.calculate_volume_profile(df)
                        fibonacci_data = self.fibonacci_analyzer.calculate_fibonacci_levels(df)
                        confluence_zones = self.fibonacci_analyzer.find_confluence_zones(df, volume_profile, signal['current_price'])
                        
                        chart_file = self.chart_generator.create_comprehensive_chart(
                            symbol, df, signal, volume_profile, fibonacci_data, confluence_zones
                        )
                        signal['chart_file'] = chart_file
                        charts_generated += 1
                        
                self.logger.debug(f"   Chart {i+1}: {signal['symbol']} ({signal.get('mtf_status', 'UNKNOWN')} MTF)")
            except Exception as e:
                self.logger.error(f"Error generating chart for {signal.get('symbol', 'unknown')}: {e}")
        
        # Update charts_generated count
        scan_info = {
            'timestamp': datetime.now(),
            'execution_time_seconds': execution_time,
            'symbols_analyzed': len(symbols),
            'signals_generated': len(all_signals),
            'success_rate': len(all_signals) / len(symbols) * 100 if symbols else 0,
            'charts_generated': charts_generated,
            'parallel_processing': True,
            'threads_used': self.config.max_workers,
            'mtf_enabled': self.config.mtf_confirmation_required,
            'confirmation_timeframes': self.config.confirmation_timeframes,
            'primary_timeframe': self.config.timeframe
        }

        # Create comprehensive results with properly sorted signals
        results = {
            'scan_info': scan_info,
            'signals': all_signals,  # Now properly sorted
            'analysis_results': analysis_results,
            'top_opportunities': self.signal_generator.rank_opportunities_with_mtf(all_signals),
            'market_summary': self.create_market_summary_with_mtf(symbols, all_signals),
            'system_performance': {
                'signals_per_minute': len(all_signals) / (execution_time / 60) if execution_time > 0 else 0,
                'avg_confidence': np.mean([s['confidence'] for s in all_signals]) if all_signals else 0,
                'avg_original_confidence': np.mean([s.get('original_confidence', s['confidence']) for s in all_signals]) if all_signals else 0,
                'mtf_boost_avg': np.mean([s['confidence'] - s.get('original_confidence', s['confidence']) for s in all_signals]) if all_signals else 0,
                'order_type_distribution': self.get_order_type_distribution(all_signals),
                'mtf_distribution': self.get_mtf_status_distribution(all_signals),
                'speedup_factor': self.calculate_speedup_factor(execution_time, len(symbols))
            }
        }
        
        self.logger.info(f"⚡ {timeframe_display} MTF PARALLEL ANALYSIS COMPLETED!")
        self.logger.info(f"   Execution Time: {execution_time:.1f}s")
        self.logger.info(f"   Total Signals: {len(all_signals)}")
        self.logger.info(f"   Charts Generated: {charts_generated}")
        self.logger.info(f"   MTF Confirmations: {self.count_mtf_confirmations(all_signals)}")

        return results

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
        """Print comprehensive analysis results with MTF information"""
        if not results:
            print("❌ No results to display")
            return
        
        scan_info = results['scan_info']
        market_summary = results['market_summary']
        opportunities = results['top_opportunities']
        
        # Enhanced Top Opportunities Table with MTF
        timeframe_display = f"{self.config.timeframe}→{'/'.join(self.config.confirmation_timeframes)}"
        print(f"\n🏆 TOP TRADING OPPORTUNITIES ({timeframe_display} MTF CONFIRMATION):")
        if opportunities:
            print("=" * 200)
            header = (
                f"{'Rank':<4} | {'Symbol':<25} | {'Side':<7} | {'Type':<6} | "
                f"{'Conf':<8} | {'MTF':<4} | {'Entry':<12} | {'Stop':<12} | {'TP1':<12} | {'TP2':<12} | "
                f"{'R/R':<5} | {'Volume':<8} | {'MTF Status':<12} | {'Confirmed':<10} | {'Chart':<15}"
            )
            print(header)
            print("-" * 200)
            
            for opp in opportunities[:15]:  # Top 15
                try:
                    side_emoji = "🟢" if opp['side'] == 'BUY' else "🔴"
                    volume_str = f"${opp['volume_24h']/1e6:.0f}M"
                    chart_available = "✅ Available" if opp.get('chart_file', '').endswith('.jpg') else "❌ None"
                    
                    # MTF status with emojis
                    mtf_status = opp.get('mtf_status', 'UNKNOWN')
                    if mtf_status == 'STRONG':
                        mtf_display = "⭐ STRONG"
                    elif mtf_status == 'PARTIAL':
                        mtf_display = "🔸 PARTIAL"
                    elif mtf_status == 'NONE':
                        mtf_display = "⚪ NONE"
                    else:
                        mtf_display = "❌ DISABLED"
                    
                    # MTF confirmation count (configurable total)
                    total_confirmation_timeframes = len(self.config.confirmation_timeframes)
                    mtf_count = f"{opp.get('mtf_confirmation_count', 0)}/{total_confirmation_timeframes}"
                    
                    # MTF confirmed timeframes
                    confirmed_tfs = ', '.join(opp.get('mtf_confirmed', []))[:8] + ("..." if len(', '.join(opp.get('mtf_confirmed', []))) > 8 else "")
                    
                    # Show MTF boost in confidence
                    conf_display = f"{opp['confidence']:.0f}"
                    if opp.get('mtf_boost', 0) > 0:
                        conf_display += f" (+{opp['mtf_boost']:.0f})"
                    
                    # Get TP1 and TP2 values
                    tp1_value = opp.get('take_profit_1', opp.get('take_profit', 0))
                    tp2_value = opp.get('take_profit_2', 0)
                    
                    row = (
                        f"{opp['rank']:<4} | {opp['symbol']:<25} | {side_emoji} {opp['side']} | "
                        f"{opp['order_type']:<6} | {conf_display:<8} | {mtf_count:<4} | "
                        f"${opp['entry_price']:<11.4f} | ${opp['stop_loss']:<11.4f} | "
                        f"${tp1_value:<11.4f} | ${tp2_value:<11.4f} | "
                        f"{opp['risk_reward_ratio']:<5.1f} | "
                        f"{volume_str:<8} | {mtf_display:<12} | {confirmed_tfs:<10} | {chart_available:<15}"
                    )
                    print(f"{row}")
                    
                except Exception as e:
                    print(f"Error displaying opportunity: {e}")
            
            print("-" * 200)
        else:
            print("   No trading opportunities found")
    
    def save_results_to_database(self, results: Dict) -> Dict[str, Any]:
        """Save all results to MySQL database (replaces CSV export)"""
        return self.enhanced_db_manager.save_all_results(results)