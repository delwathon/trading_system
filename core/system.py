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
from analysis.multi_timeframe import MultiTimeframeAnalyzer
from signals.generator import SignalGenerator
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
        
        self.logger.info("🚀 TOP OPPORTUNITIES SYSTEM WITH MULTI-TIMEFRAME & MYSQL DATABASE INITIALIZED!")
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
        """Complete analysis with multi-timeframe confirmation - Generate comprehensive signal"""
        try:
            symbol = symbol_data['symbol']
            self.logger.debug(f"📊 Analyzing {symbol} (30m → 1h/4h/6h MTF)...")
            
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
            
            # Generate primary signal on 30m timeframe
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
                
                # Chart will be generated later only for top signals
                primary_signal['chart_file'] = None  # Will be populated later for top signals
                
                # Add comprehensive analysis data (needed for chart generation later)
                primary_signal['analysis'] = {
                    'volume_profile': volume_profile,
                    'fibonacci_data': fibonacci_data,
                    'confluence_zones': confluence_zones,
                    'technical_summary': self.signal_generator.create_technical_summary(df),
                    'risk_assessment': self.signal_generator.assess_signal_risk(primary_signal, df)
                }
                
                # Store raw data for later chart generation (for top signals only)
                primary_signal['_chart_data'] = {
                    'df': df,  # Store processed dataframe
                    'symbol_data': symbol_data
                }
                
                self.logger.debug(f"✅ {symbol} - {primary_signal['side'].upper()} signal generated ({self.config.timeframe} base) - Chart: Pending")
                return primary_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"MTF analysis failed for {symbol_data.get('symbol', 'unknown')}: {e}")
            return None

    def generate_charts_for_top_signals(self, top_signals: List[Dict]) -> int:
        """Generate charts (HTML + Screenshot) for top-ranked signals only"""
        charts_generated = 0
        
        try:
            self.logger.info(f"📊 Generating charts for TOP {len(top_signals)} signals only...")
            
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
            
            self.logger.info(f"✅ Chart generation complete: {charts_generated}/{len(top_signals)} charts created")
            return charts_generated
            
        except Exception as e:
            self.logger.error(f"Chart generation process failed: {e}")
            return charts_generated

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
        """Parallel analysis with multi-timeframe confirmation and optimized chart generation - FIXED"""
        start_time = time.time()
        
        timeframe_display = f"{self.config.timeframe}→{'/'.join(self.config.confirmation_timeframes)}"
        self.logger.info(f"🚀 STARTING OPTIMIZED ANALYSIS WITH {timeframe_display} MTF CONFIRMATION")
        self.logger.info("   📊 Charts generated ONLY for top-ranked signals (after analysis complete)")
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
        self.logger.info(f"   Chart Strategy: Top {self.config.charts_per_batch} signals only")
        
        # Initialize counters
        self.processed_count = 0
        all_signals = []
        analysis_results = []
        
        # ===== PHASE 1: SIGNAL GENERATION (NO CHARTS) =====
        self.logger.info("📈 PHASE 1: Signal Generation (No Charts)")
        
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
                    if completed_tasks % 10 == 0 or completed_tasks == len(symbols):
                        self.logger.debug(f"   Progress: {completed_tasks}/{len(symbols)} ({progress:.1f}%) - Signals: {len(all_signals)}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol_data['symbol']}: {e}")
                    analysis_results.append({
                        'symbol': symbol_data['symbol'],
                        'status': 'error',
                        'error': str(e)
                    })

        # ===== PHASE 2: SIGNAL RANKING =====
        analysis_time = time.time() - start_time
        self.logger.info(f"📊 PHASE 2: Signal Ranking ({len(all_signals)} signals in {analysis_time:.1f}s)")
        
        # Sort signals by MTF priority first, then confidence
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
            
            return (mtf_priority, confidence, priority_boost)
        
        # Sort in descending order (highest priority first)
        all_signals.sort(key=get_sort_key, reverse=True)
        
        # Log top signals after ranking
        self.logger.info(f"🏆 Top {self.config.charts_per_batch} Signals After Ranking:")
        for i, signal in enumerate(all_signals[:self.config.charts_per_batch]):
            mtf_status = signal.get('mtf_status', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            symbol = signal.get('symbol', 'UNKNOWN')
            side = signal.get('side', 'UNKNOWN')
            self.logger.info(f"   {i+1}. {symbol} {side.upper()} - {confidence}% conf - MTF: {mtf_status}")
        
        # ===== PHASE 3: OPTIMIZED CHART GENERATION =====
        # Generate charts ONLY for top N signals
        charts_generated = 0
        top_signals_for_charts = all_signals[:self.config.charts_per_batch]
        
        if top_signals_for_charts:
            self.logger.info(f"📊 PHASE 3: Chart Generation (Top {len(top_signals_for_charts)} signals only)")
            charts_generated = self.generate_charts_for_top_signals(top_signals_for_charts)
        else:
            self.logger.info("📊 PHASE 3: No signals found for chart generation")
        
        execution_time = time.time() - start_time
        
        # ===== PHASE 4: RESULTS COMPILATION =====
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
            'primary_timeframe': self.config.timeframe,
            'optimization': 'charts_for_top_signals_only'
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
                'speedup_factor': self.calculate_speedup_factor(execution_time, len(symbols)),
                'chart_efficiency': f"{charts_generated}/{len(all_signals)} signals charted"
            }
        }
        
        self.logger.info(f"⚡ {timeframe_display} OPTIMIZED MTF ANALYSIS COMPLETED!")
        self.logger.info(f"   Total Execution Time: {execution_time:.1f}s")
        self.logger.info(f"   Signal Generation: {analysis_time:.1f}s ({len(all_signals)} signals)")
        self.logger.info(f"   Chart Generation: {execution_time - analysis_time:.1f}s ({charts_generated} charts)")
        self.logger.info(f"   Chart Efficiency: {charts_generated}/{len(all_signals)} = {charts_generated/max(1,len(all_signals))*100:.1f}% charted")
        self.logger.info(f"   MTF Confirmations: {self.count_mtf_confirmations(all_signals)}")

        # FIXED: ADD THE MISSING RETURN STATEMENT!
        return results

    def filter_signals_for_top_opportunities(self, signals: List[Dict], max_opportunities: int = 8) -> List[Dict]:
        """Filter signals to return only TOP OPPORTUNITIES - IMPLEMENTED IN SYSTEM.PY"""
        try:
            if not signals:
                return []
            
            # TIER 1: Premium opportunities (highest priority)
            tier1_opportunities = []
            # TIER 2: Quality opportunities  
            tier2_opportunities = []
            # TIER 3: Decent opportunities
            tier3_opportunities = []
            
            for signal in signals:
                confidence = signal.get('confidence', 0)
                rr_ratio = signal.get('risk_reward_ratio', 0)
                volume_24h = signal.get('volume_24h', 0)
                mtf_status = signal.get('mtf_status', 'NONE')
                
                # Calculate quality score
                quality_score = (
                    confidence * 0.4 +  # 40% weight on confidence
                    min(100, rr_ratio * 18) * 0.3 +  # 30% weight on R/R
                    min(100, volume_24h / 800_000) * 0.2 +  # 20% weight on volume
                    (100 if signal.get('order_type') == 'market' else 92) * 0.1  # 10% weight on execution
                )
                
                signal['quality_score'] = quality_score
                
                # TIER 1: PREMIUM OPPORTUNITIES (Top 10%)
                if (confidence >= 65 and rr_ratio >= 2.5 and volume_24h >= 2_000_000 and 
                    mtf_status in ['STRONG', 'PARTIAL']):
                    tier1_opportunities.append(signal)
                
                # TIER 2: QUALITY OPPORTUNITIES (Next 20%)
                elif (confidence >= 55 and rr_ratio >= 2.0 and volume_24h >= 1_000_000):
                    tier2_opportunities.append(signal)
                
                # TIER 3: DECENT OPPORTUNITIES (Backup options)
                elif (confidence >= 45 and rr_ratio >= 1.8 and volume_24h >= 500_000):
                    tier3_opportunities.append(signal)
            
            # Sort each tier by quality score
            tier1_opportunities.sort(key=lambda x: x['quality_score'], reverse=True)
            tier2_opportunities.sort(key=lambda x: x['quality_score'], reverse=True)
            tier3_opportunities.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # Build final list prioritizing higher tiers
            top_opportunities = []
            
            # Take best from Tier 1 (max 5)
            top_opportunities.extend(tier1_opportunities[:5])
            remaining_slots = max_opportunities - len(top_opportunities)
            
            # Fill remaining with Tier 2 (max 3 more)
            if remaining_slots > 0:
                top_opportunities.extend(tier2_opportunities[:min(3, remaining_slots)])
                remaining_slots = max_opportunities - len(top_opportunities)
            
            # Fill any remaining with Tier 3 (max 2 more)
            if remaining_slots > 0:
                top_opportunities.extend(tier3_opportunities[:min(2, remaining_slots)])
            
            # Log the filtering results
            self.logger.info(f"🎯 TOP OPPORTUNITIES FILTER:")
            self.logger.info(f"   Tier 1 (Premium): {len(tier1_opportunities)} → Taking {len([s for s in top_opportunities if s in tier1_opportunities])}")
            self.logger.info(f"   Tier 2 (Quality): {len(tier2_opportunities)} → Taking {len([s for s in top_opportunities if s in tier2_opportunities])}")
            self.logger.info(f"   Tier 3 (Decent): {len(tier3_opportunities)} → Taking {len([s for s in top_opportunities if s in tier3_opportunities])}")
            self.logger.info(f"   📊 FINAL: {len(top_opportunities)} TOP OPPORTUNITIES selected from {len(signals)} total signals")
            
            return top_opportunities
            
        except Exception as e:
            self.logger.error(f"Top opportunities filtering error: {e}")
            # Fallback: return top signals by confidence
            signals_sorted = sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True)
            return signals_sorted[:max_opportunities]
        
        # ===== PHASE 3: OPTIMIZED CHART GENERATION =====
        # Generate charts ONLY for top opportunities
        charts_generated = 0
        
        if top_opportunities:
            self.logger.info(f"📊 PHASE 3: Chart Generation (Top {len(top_opportunities)} opportunities only)")
            charts_generated = self.generate_charts_for_top_signals(top_opportunities)
        else:
            self.logger.info("📊 PHASE 3: No top opportunities found for chart generation")
        
        execution_time = time.time() - start_time
        
        # ===== PHASE 4: RESULTS COMPILATION =====
        scan_info = {
            'timestamp': datetime.now(),
            'execution_time_seconds': execution_time,
            'symbols_analyzed': len(symbols),
            'signals_generated': len(all_signals),  # Total signals generated
            'top_opportunities_count': len(top_opportunities),  # What gets saved/displayed
            'success_rate': len(all_signals) / len(symbols) * 100 if symbols else 0,
            'charts_generated': charts_generated,
            'parallel_processing': True,
            'threads_used': self.config.max_workers,
            'mtf_enabled': self.config.mtf_confirmation_required,
            'confirmation_timeframes': self.config.confirmation_timeframes,
            'primary_timeframe': self.config.timeframe,
            'optimization': 'top_opportunities_focus'
        }

        # Create comprehensive results - ONLY TOP OPPORTUNITIES are returned as main signals
        results = {
            'scan_info': scan_info,
            'signals': all_signals,  # All signals for reference (not saved to DB)
            'analysis_results': analysis_results,
            'top_opportunities': top_opportunities,  # FIXED: This should contain the 8 selected opportunities
            'market_summary': self.create_market_summary_with_mtf(symbols, all_signals),
            'system_performance': {
                'signals_per_minute': len(all_signals) / (execution_time / 60) if execution_time > 0 else 0,
                'avg_confidence': np.mean([s['confidence'] for s in all_signals]) if all_signals else 0,
                'avg_original_confidence': np.mean([s.get('original_confidence', s['confidence']) for s in all_signals]) if all_signals else 0,
                'mtf_boost_avg': np.mean([s['confidence'] - s.get('original_confidence', s['confidence']) for s in all_signals]) if all_signals else 0,
                'order_type_distribution': self.get_order_type_distribution(all_signals),
                'mtf_distribution': self.get_mtf_status_distribution(all_signals),
                'speedup_factor': self.calculate_speedup_factor(execution_time, len(symbols)),
                'chart_efficiency': f"{charts_generated}/{len(top_opportunities)} opportunities charted",
                'top_opportunities_summary': self.generate_top_opportunities_summary(top_opportunities)
            }
        }
        
        # FIXED: Add debug logging to verify top_opportunities is set correctly
        self.logger.debug(f"🔍 DEBUG: Results structure check:")
        self.logger.debug(f"   results['top_opportunities'] length: {len(results.get('top_opportunities', []))}")
        self.logger.debug(f"   results['signals'] length: {len(results.get('signals', []))}")
        
        self.logger.info(f"⚡ {timeframe_display} TOP OPPORTUNITIES ANALYSIS COMPLETED!")
        self.logger.info(f"📊 TOTAL: {len(all_signals)} signals → TOP: {len(top_opportunities)} opportunities")
        
        return results
    
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
        """Print ONLY TOP OPPORTUNITIES (not all signals) - FIXED TP display"""
        if not results:
            print("❌ No results to display")
            return
        
        scan_info = results['scan_info']
        opportunities = results['top_opportunities']
        
        # Show summary first
        total_signals = scan_info.get('signals_generated', 0)
        charts_generated = scan_info.get('charts_generated', 0)
        
        print(f"\n📊 SCAN SUMMARY:")
        print(f"   Total Signals Generated: {total_signals}")
        print(f"   Top Opportunities Selected: {len(opportunities)}")
        print(f"   Charts Generated: {charts_generated}")
        print(f"   Analysis Time: {scan_info.get('execution_time_seconds', 0):.1f}s")
        
        # Enhanced Top Opportunities Table
        timeframe_display = f"{self.config.timeframe}→{'/'.join(self.config.confirmation_timeframes)}"
        print(f"\n🏆 TOP OPPORTUNITIES ONLY ({timeframe_display} MTF CONFIRMATION):")
        
        if opportunities:
            # Show only top opportunities that would be executed
            max_display = max(self.config.max_execution_per_trade, 5)  # Show at least 5, or max_execution_per_trade
            display_opportunities = opportunities[:max_display]
            
            print("=" * 150)
            header = (
                f"{'#':<3} | {'Symbol':<20} | {'Side':<7} | "
                f"{'Conf':<8} | {'MTF':<12} | {'Entry':<12} | {'Stop':<12} | {'TP1':<12} | "
                f"{'R/R':<5} | {'Volume':<8} | {'Status':<12}"
            )
            print(header)
            print("-" * 150)
            
            for i, opp in enumerate(display_opportunities):
                try:
                    # Will this be executed?
                    will_execute = i < self.config.max_execution_per_trade
                    status = "🎯 EXECUTE" if will_execute else "⏭️ SKIP"
                    
                    side_emoji = "🟢" if opp['side'] == 'buy' else "🔴"
                    volume_str = f"${opp['volume_24h']/1e6:.0f}M"
                    
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
                    
                    # Show MTF boost in confidence
                    conf_display = f"{opp['confidence']:.0f}%"
                    if opp.get('mtf_boost', 0) > 0:
                        conf_display += f" (+{opp['mtf_boost']:.0f})"
                    
                    # FIXED: Get TP1 value with proper fallbacks
                    tp1_value = opp.get('take_profit_1', opp.get('take_profit', 0))
                    if tp1_value == 0:  # Still no value, calculate based on entry and stop
                        entry = opp.get('entry_price', 0)
                        stop = opp.get('stop_loss', 0)
                        if entry > 0 and stop > 0:
                            risk = abs(entry - stop)
                            if opp.get('side', '').upper() == 'BUY':
                                tp1_value = entry + (risk * 3)  # 3:1 R/R
                            else:
                                tp1_value = entry - (risk * 3)  # 3:1 R/R
                    
                    row = (
                        f"{i+1:<3} | {opp['symbol']:<20} | {side_emoji} {opp['side']:<5} | "
                        f"{conf_display:<8} | {mtf_display:<12} | "
                        f"${opp['entry_price']:<11.4f} | ${opp['stop_loss']:<11.4f} | ${tp1_value:<11.4f} | "
                        f"{opp['risk_reward_ratio']:<5.1f} | "
                        f"{volume_str:<8} | {status:<12}"
                    )
                    print(f"{row}")
                    
                except Exception as e:
                    print(f"Error displaying opportunity: {e}")
            
            print("-" * 150)
            
            # Execution Summary
            execution_count = min(len(opportunities), self.config.max_execution_per_trade)
            print(f"\n🎯 EXECUTION PLAN:")
            print(f"   Will Execute: {execution_count} trades")
            print(f"   Will Skip: {max(0, len(opportunities) - execution_count)} opportunities")
            print(f"   Max Per Scan: {self.config.max_execution_per_trade}")
            
        else:
            print("   No trading opportunities found")
            
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