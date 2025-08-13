"""
Signal Generator V13.0 - Main Orchestrator
==========================================
PART 5: Main System Integration and Orchestration

This is the main entry point that integrates all components:
- Signal generation pipeline
- Entry monitoring
- Risk management
- Performance tracking
- API endpoints
"""

import os
import asyncio
import logging
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import pandas as pd
import json
from dataclasses import asdict
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import all components
from signals.signal_gen_v13_core import (
    SystemConfiguration, SignalCriteria, StateManager,
    Signal, SignalStatus, SignalQuality, TimeFrame,
    SignalLifecycleManager, SignalQueueManager,
    PerformanceTracker, SystemErrorHandler
)

from signals.signal_gen_v13_analysis import (
    TechnicalIndicators, VolumeProfileAnalyzer, MarketStructureAnalyzer
)

from signals.signal_gen_v13_ml_generation import (
    MLPredictionEngine, NewsSentimentAnalyzer, MultiTimeframeSignalGenerator
)

from signals.signal_gen_v13_monitoring import (
    EntryMonitoringService, AdvancedRiskManager, RiskProfile,
    TrailingStopManager
)

# ===========================
# MAIN ORCHESTRATOR
# ===========================
class ResourceManager:
    """Enhanced resource manager with event loop safety"""
    
    def __init__(self):
        self.http_sessions = {}
        self.active_tasks = set()
        self.temp_files = set()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._shutdown = False
    
    def _is_event_loop_running(self) -> bool:
        """Check if event loop is running and not closed"""
        try:
            loop = asyncio.get_running_loop()
            return loop is not None and not loop.is_closed()
        except RuntimeError:
            return False
    
    async def get_http_session(self, session_name: str = "default") -> Optional[aiohttp.ClientSession]:
        """Get or create HTTP session with safety checks"""
        if self._shutdown or not self._is_event_loop_running():
            return None
        
        try:
            if session_name not in self.http_sessions or self.http_sessions[session_name].closed:
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)
                
                self.http_sessions[session_name] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={'User-Agent': 'SignalGenerator/1.0'}
                )
                
                self.logger.debug(f"Created HTTP session: {session_name}")
            
            return self.http_sessions[session_name]
            
        except Exception as e:
            self.logger.debug(f"Session creation error: {e}")
            return None
    
    def track_task(self, task: asyncio.Task):
        """Track async task for cleanup"""
        if not self._shutdown:
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Starting ResourceManager shutdown...")
        self._shutdown = True
        
        # Give ongoing operations a chance to complete
        await asyncio.sleep(0.5)
        
        await self.cleanup()
    
    async def cleanup(self):
        """Cleanup all managed resources"""
        self._shutdown = True
        
        if not self._is_event_loop_running():
            self.logger.warning("Event loop not running during cleanup")
            return
        
        self.logger.info("Starting resource cleanup...")
        
        # Cancel active tasks
        cancelled_count = 0
        for task in list(self.active_tasks):
            if not task.done():
                task.cancel()
                cancelled_count += 1
        
        # Wait for cancellations to complete
        if cancelled_count > 0:
            self.logger.info(f"Cancelling {cancelled_count} active tasks...")
            await asyncio.sleep(0.2)
            
            # Clean up cancelled tasks
            for task in list(self.active_tasks):
                if task.cancelled() or task.done():
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
        
        # Close HTTP sessions
        session_count = 0
        for session_name, session in list(self.http_sessions.items()):
            try:
                if not session.closed:
                    await session.close()
                    session_count += 1
                    self.logger.debug(f"Closed HTTP session: {session_name}")
            except Exception as e:
                self.logger.debug(f"Error closing session {session_name}: {e}")
        
        if session_count > 0:
            self.logger.info(f"Closed {session_count} HTTP sessions")
        
        # Remove temporary files
        file_count = 0
        for filepath in list(self.temp_files):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    file_count += 1
                self.temp_files.discard(filepath)
            except Exception as e:
                self.logger.debug(f"Failed to remove temp file {filepath}: {e}")
        
        if file_count > 0:
            self.logger.info(f"Removed {file_count} temporary files")
        
        self.logger.info("Resource cleanup completed")

class SignalGeneratorV13:
    """
    Main orchestrator for the Signal Generation System V13
    Professional architecture with complete lifecycle management
    """
    
    def __init__(self, config: Optional[SystemConfiguration] = None,
                 exchange_manager=None):
        """Initialize the signal generation system"""
        
        # Configuration
        self.config = config or SystemConfiguration()
        self.exchange_manager = exchange_manager
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Add resource manager
        self.resource_manager = ResourceManager()
        
        # Pass resource manager to components that need it
        self.news_analyzer = NewsSentimentAnalyzer(config, self.resource_manager)
        
        # Initialize state manager
        self.state_manager = StateManager(self.config)
        
        # Initialize risk profile
        self.risk_profile = RiskProfile(
            max_risk_per_trade=self.config.max_risk_per_trade,
            max_daily_risk=self.config.max_daily_risk,
            max_open_positions=self.config.max_concurrent_positions
        )
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self._is_running = False
        self._main_loop_task = None
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'signals_executed': 0,
            'signals_cancelled': 0,
            'signals_expired': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        
        self.logger.info("=" * 60)
        self.logger.info("Signal Generator V13.0 - Professional Architecture")
        self.logger.info("=" * 60)
        self.logger.info(f"Primary TF: {self.config.primary_timeframe.label}")
        self.logger.info(f"Confirmation TFs: {[tf.label for tf in self.config.confirmation_timeframes]}")
        self.logger.info(f"Entry TF: {self.config.entry_timeframe.label}")
        self.logger.info(f"Max concurrent positions: {self.config.max_concurrent_positions}")
        self.logger.info(f"Risk per trade: {self.config.max_risk_per_trade:.1%}")
        self.logger.info("=" * 60)
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        # Core components
        self.lifecycle_manager = SignalLifecycleManager(
            self.state_manager, self.config
        )
        
        self.queue_manager = SignalQueueManager(
            self.state_manager, self.config
        )
        
        self.performance_tracker = PerformanceTracker(
            self.state_manager
        )
        
        self.error_handler = SystemErrorHandler()
        
        # Analysis components
        self.signal_generator = MultiTimeframeSignalGenerator(
            self.config, self.state_manager
        )
        
        self.ml_engine = MLPredictionEngine(self.config)
        self.news_analyzer = NewsSentimentAnalyzer(self.config)
        
        # Risk and monitoring
        self.risk_manager = AdvancedRiskManager(
            self.config, self.risk_profile
        )
        
        self.entry_monitor = EntryMonitoringService(
            self.state_manager, self.config
        )
        
        self.trailing_stop_manager = TrailingStopManager()
        
        self.logger.info("All components initialized successfully")
    
    # ===========================
    # SYSTEM LIFECYCLE
    # ===========================
    
    async def start(self):
        """Start the signal generation system"""
        if self._is_running:
            self.logger.warning("System already running")
            return
        
        self._is_running = True
        self.logger.info("Starting Signal Generator V13...")
        
        try:
            # Start monitoring service
            await self.entry_monitor.start_monitoring()
            
            # Start main loop
            self._main_loop_task = asyncio.create_task(self._main_loop())
            
            # Start queue processor
            asyncio.create_task(self._queue_processor())
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_task())
            
            self.logger.info("Signal Generator V13 started successfully")
            
        except Exception as e:
            self.logger.error(f"Startup error: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the signal generation system with proper cleanup order"""
        self.logger.info("Stopping Signal Generator V13...")
        
        # Set shutdown flag first
        self._is_running = False
        
        # Stop monitoring service first
        try:
            await asyncio.wait_for(self.entry_monitor.stop_monitoring(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Entry monitor stop timeout")
        except Exception as e:
            self.logger.warning(f"Entry monitor stop error: {e}")
        
        # Cancel main loop
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await asyncio.wait_for(self._main_loop_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Shutdown components in order
        shutdown_tasks = []
        
        # Shutdown news analyzer
        if hasattr(self, 'news_analyzer'):
            shutdown_tasks.append(self.news_analyzer.shutdown())
        
        # Shutdown resource manager
        if hasattr(self, 'resource_manager'):
            shutdown_tasks.append(self.resource_manager.shutdown())
        
        # Wait for all shutdowns with timeout
        if shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Shutdown timeout - some components may not have stopped cleanly")
        
        # Save state
        try:
            self._save_state()
        except Exception as e:
            self.logger.warning(f"State save error: {e}")
        
        self.logger.info("Signal Generator V13 stopped")

    async def _main_loop(self):
        """Main processing loop"""
        while self._is_running:
            try:
                # Process each symbol
                symbols = await self._get_symbols_to_analyze()
                
                for symbol in symbols:
                    if not self._is_running:
                        break
                    
                    try:
                        await self._process_symbol(symbol)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        if not self.error_handler.handle_error(e, {'module': 'main_loop', 'symbol': symbol}):
                            await self.stop()
                            break
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol with proper resource management"""
        active_tasks = []
        try:
            # Track tasks for proper cleanup
            self.logger.debug(f"Processing {symbol}")
            
            # Create task with proper cancellation handling
            analysis_task = asyncio.create_task(
                self.signal_generator.analyze(symbol, self.exchange_manager)
            )
            active_tasks.append(analysis_task)
            
            # Wait with timeout to prevent hanging
            signals = await asyncio.wait_for(analysis_task, timeout=30.0)
            
            if signals:
                for signal in signals:
                    # Process each signal with timeout
                    process_task = asyncio.create_task(self._process_signal(signal))
                    active_tasks.append(process_task)
                    await asyncio.wait_for(process_task, timeout=10.0)
        
        except asyncio.TimeoutError:
            self.logger.warning(f"Symbol processing timeout for {symbol}")
        except Exception as e:
            self.logger.error(f"Symbol processing error for {symbol}: {e}")
        finally:
            # CRITICAL: Cancel and cleanup all tasks
            for task in active_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass  # Expected when cancelling
    
    async def _process_signal(self, signal: Signal):
        """Process a generated signal"""
        try:
            # Get account balance
            account_balance = await self._get_account_balance()
            
            # Calculate position size
            position_result = self.risk_manager.calculate_position_size(
                signal, account_balance
            )
            
            # Update signal with position info
            signal.position_size = position_result.position_size
            signal.risk_amount = position_result.risk_amount
            signal.potential_profit = position_result.max_profit
            
            # Validate risk limits
            current_positions = self.state_manager.get_active_positions()
            is_valid, reason = self.risk_manager.validate_risk_limits(
                signal, current_positions
            )
            
            if not is_valid:
                self.logger.warning(f"Signal rejected: {reason}")
                signal.status = SignalStatus.CANCELLED
                return
            
            # Add risk warnings
            signal.warnings.extend(position_result.risk_warnings)
            
            # Process through lifecycle manager
            processed_signal = await self.lifecycle_manager.process_new_signal(
                signal.to_dict()
            )
            
            if processed_signal:
                self.stats['signals_generated'] += 1
                self.logger.info(f"Signal generated: {processed_signal.id}")
                self.logger.info(f"  Symbol: {processed_signal.symbol}")
                self.logger.info(f"  Side: {processed_signal.side}")
                self.logger.info(f"  Entry: {processed_signal.entry_price:.6f}")
                self.logger.info(f"  Stop: {processed_signal.stop_loss:.6f}")
                self.logger.info(f"  TP1: {processed_signal.take_profit_1:.6f}")
                self.logger.info(f"  R/R: {processed_signal.risk_reward_ratio:.2f}")
                self.logger.info(f"  Position Size: {processed_signal.position_size:.4f}")
                self.logger.info(f"  Quality: {processed_signal.quality_tier.value}")
            
        except Exception as e:
            self.logger.error(f"Signal processing error: {e}")
    
    async def _queue_processor(self):
        """Process pending signal queue"""
        while self._is_running:
            try:
                await self.queue_manager.process_queue()
                await asyncio.sleep(30)  # Process every 30 seconds
            except Exception as e:
                self.logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """Periodic cleanup of expired signals"""
        while self._is_running:
            try:
                expired_count = self.state_manager.cleanup_expired_signals()
                if expired_count > 0:
                    self.logger.info(f"Cleaned up {expired_count} expired signals")
                    self.stats['signals_expired'] += expired_count
                
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)
    
    # ===========================
    # PUBLIC API
    # ===========================
    
    async def analyze_symbol_comprehensive(self, symbol: str) -> Optional[Dict]:
        """
        Public API: Analyze a symbol comprehensively
        Compatible with existing interfaces
        """
        try:
            # Generate signals
            signals = await self.signal_generator.analyze(symbol, self.exchange_manager)
            
            if not signals:
                return None
            
            # Return the best signal
            best_signal = signals[0]
            
            # Convert to expected format
            return {
                'symbol': best_signal.symbol,
                'side': best_signal.side,
                'entry_price': best_signal.entry_price,
                'stop_loss': best_signal.stop_loss,
                'take_profit_1': best_signal.take_profit_1,
                'take_profit_2': best_signal.take_profit_2,
                'risk_reward_ratio': best_signal.risk_reward_ratio,
                'confidence': best_signal.confidence,
                'signal_type': f"{best_signal.side}_signal",
                'volume_24h': 0,  # Would come from exchange
                'price_change_24h': 0,  # Would come from exchange
                'current_price': best_signal.current_price,
                'analysis': {
                    'quality_tier': best_signal.quality_tier.value,
                    'ml_confidence': best_signal.ml_prediction.get('confidence', 0) if best_signal.ml_prediction else 0,
                    'news_sentiment': best_signal.news_sentiment.get('classification', 'neutral') if best_signal.news_sentiment else 'neutral',
                    'warnings': best_signal.warnings
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis error for {symbol}: {e}")
            return None
    
    def rank_opportunities_with_mtf(self, signals: List[Dict]) -> List[Dict]:
        """
        Public API: Rank signals by quality
        Compatible with existing interfaces
        """
        try:
            # Convert to Signal objects if needed
            signal_objects = []
            for s in signals:
                if isinstance(s, Signal):
                    signal_objects.append(s)
                else:
                    # Convert dict to Signal
                    signal_objects.append(Signal.from_dict(s))
            
            # Sort by quality and confidence
            signal_objects.sort(key=lambda x: (
                x.quality_tier.value if isinstance(x.quality_tier, SignalQuality) else 0,
                x.confidence,
                x.risk_reward_ratio
            ), reverse=True)
            
            # Convert back to dicts with ranking
            ranked = []
            for i, signal in enumerate(signal_objects, 1):
                signal_dict = signal.to_dict()
                signal_dict['rank'] = i
                signal_dict['priority'] = 1000 - (i * 10)  # Simple priority score
                ranked.append(signal_dict)
            
            return ranked
            
        except Exception as e:
            self.logger.error(f"Ranking error: {e}")
            return signals
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals"""
        active = self.state_manager.get_active_positions()
        return [s.to_dict() for s in active]
    
    def get_pending_signals(self) -> List[Dict]:
        """Get pending signals"""
        pending = self.state_manager.get_pending_signals()
        return [s.to_dict() for s in pending]
    
    def get_performance_stats(self) -> Dict:
        """Get system performance statistics"""
        return {
            'system_stats': self.stats,
            'performance_report': self.performance_tracker.get_performance_report(),
            'risk_metrics': self.risk_manager.get_risk_metrics(
                self.state_manager.get_active_positions(),
                100000  # Default balance
            )
        }
    
    def update_signal_outcome(self, signal_id: str, outcome: str, 
                            pnl: float = 0.0):
        """Update signal with execution outcome"""
        try:
            # Update signal status
            status = SignalStatus.EXECUTED if outcome == 'success' else SignalStatus.FAILED
            self.state_manager.update_signal(signal_id, {'status': status})
            
            # Update performance
            self.performance_tracker.update_signal_outcome(
                signal_id,
                {'result': 'win' if pnl > 0 else 'loss', 'pnl': pnl}
            )
            
            # Update stats
            if outcome == 'success':
                self.stats['signals_executed'] += 1
            else:
                self.stats['signals_cancelled'] += 1
            
            self.stats['total_pnl'] += pnl
            
        except Exception as e:
            self.logger.error(f"Outcome update error: {e}")
    
    # ===========================
    # HELPER METHODS
    # ===========================
    
    async def _get_symbols_to_analyze(self) -> List[str]:
        """Get list of symbols to analyze"""
        # This would typically come from a database or configuration
        # For now, return a default list
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
            'XRPUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT'
        ]
    
    def _should_analyze_symbol(self, symbol: str) -> bool:
        """Check if symbol should be analyzed"""
        symbol_state = self.state_manager._symbol_states.get(symbol)
        
        # Check if blacklisted
        if symbol_state and symbol_state.is_blacklisted:
            return False
        
        # Check if max signals reached
        if symbol_state and symbol_state.active_signals >= self.config.max_signals_per_symbol:
            return False
        
        # Check time since last analysis
        if symbol_state and symbol_state.last_analysis:
            time_since = (datetime.now(timezone.utc) - symbol_state.last_analysis).total_seconds()
            if time_since < 300:  # Less than 5 minutes
                return False
        
        return True
    
    async def _get_account_balance(self) -> float:
        """Get account balance from exchange"""
        if self.exchange_manager:
            try:
                # This would call exchange API
                return 100000.0  # Placeholder
            except Exception:
                return 100000.0
        return 100000.0  # Default
    
    def _save_state(self):
        """Save system state to persistent storage"""
        try:
            state_data = {
                'stats': self.stats,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Save to file or database
            with open('signal_generator_state.json', 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info("State saved successfully")
            
        except Exception as e:
            self.logger.error(f"State save error: {e}")
    
    def _load_state(self):
        """Load system state from persistent storage"""
        try:
            with open('signal_generator_state.json', 'r') as f:
                state_data = json.load(f)
                self.stats.update(state_data.get('stats', {}))
            
            self.logger.info("State loaded successfully")
            
        except Exception:
            self.logger.info("No previous state found, starting fresh")

# ===========================
# FACTORY FUNCTIONS
# ===========================

def create_signal_generator(config_dict: Optional[Dict] = None,
                          exchange_manager=None) -> SignalGeneratorV13:
    """
    Factory function to create Signal Generator V13
    
    Args:
        config_dict: Configuration dictionary
        exchange_manager: Exchange manager instance
    
    Returns:
        SignalGeneratorV13 instance
    """
    # Parse configuration
    if config_dict:
        config = SystemConfiguration(
            primary_timeframe=TimeFrame.from_string(config_dict.get('timeframe', '6h')),
            confirmation_timeframes=[
                TimeFrame.from_string(tf) for tf in 
                config_dict.get('confirmation_timeframes', ['4h', '1h'])
            ],
            entry_timeframe=TimeFrame.from_string(config_dict.get('entry_timeframe', '1h')),
            max_concurrent_positions=config_dict.get('max_concurrent_positions', 10),
            max_risk_per_trade=config_dict.get('max_risk_per_trade', 0.02),
            max_daily_risk=config_dict.get('max_daily_risk', 0.06)
        )
    else:
        config = SystemConfiguration()
    
    return SignalGeneratorV13(config, exchange_manager)

# ===========================
# MAIN ENTRY POINT
# ===========================

async def main():
    """Main entry point for standalone execution"""
    # Create configuration
    config = SystemConfiguration(
        primary_timeframe=TimeFrame.H6,
        confirmation_timeframes=[TimeFrame.H4, TimeFrame.H1],
        entry_timeframe=TimeFrame.H1
    )
    
    # Create signal generator
    generator = SignalGeneratorV13(config)
    
    try:
        # Start the system
        await generator.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
            # Print stats periodically
            stats = generator.get_performance_stats()
            print("\n" + "="*50)
            print("System Statistics:")
            print(f"Signals Generated: {stats['system_stats']['signals_generated']}")
            print(f"Signals Executed: {stats['system_stats']['signals_executed']}")
            print(f"Total P&L: {stats['system_stats']['total_pnl']:.2f}")
            print("="*50 + "\n")
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        await generator.stop()

if __name__ == "__main__":
    asyncio.run(main())

# ===========================
# EXPORTS
# ===========================

__all__ = [
    'SignalGeneratorV13',
    'create_signal_generator'
]

# Maintain compatibility with old name
SignalGenerator = SignalGeneratorV13