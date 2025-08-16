import asyncio
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import uuid
import ccxt

from config.config import EnhancedSystemConfig
from trading_module.bybit import PositionSizer, LeverageManager, PositionManager, LeveragedProfitMonitor, OrderExecutor
from core.system import CompleteEnhancedBybitSystem
from database.models import DatabaseManager, AutoTradingSession
from utils.database_manager import EnhancedDatabaseManager
from telegram_bot_and_notification.bootstrap_manager import send_trading_notification


# ========================================
# PHASE 4: SCHEDULE MANAGEMENT
# ========================================

class ScheduleManager:
    """Handle scan timing and scheduling logic - ENHANCED with market hours awareness"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def parse_start_hour(self, start_hour_str: str) -> Tuple[int, int]:
        """Parse start hour string like '01:00' to (hour, minute)"""
        try:
            hour, minute = start_hour_str.split(':')
            return int(hour), int(minute)
        except Exception as e:
            self.logger.error(f"Error parsing start hour {start_hour_str}: {e}")
            return 1, 0  # Default to 01:00
    
    def is_optimal_trading_time(self) -> Tuple[bool, str]:
        """
        ENHANCED: Check if current time is optimal for trading
        
        Returns:
            Tuple of (is_optimal, session_name)
        """
        try:
            now_utc = datetime.now(timezone.utc)
            hour_utc = now_utc.hour
            
            # Market session analysis
            if 0 <= hour_utc <= 6:  # Asian session
                return True, "asian_session"
            elif 6 <= hour_utc <= 14:  # European session
                return True, "european_session" 
            elif 14 <= hour_utc <= 22:  # US session
                return True, "us_session"
            else:  # Low liquidity hours (22-24 UTC)
                # Allow trading but note low liquidity
                return True, "low_liquidity"
            
        except Exception:
            return True, "unknown"
    
    def calculate_next_scan_time(self) -> datetime:
        """Calculate next scheduled scan time"""
        try:
            now = datetime.now()
            start_hour, start_minute = self.parse_start_hour(self.config.day_trade_start_hour)
            scan_interval_hours = self.config.scan_interval / 3600  # Convert seconds to hours
            
            # Create today's start time
            today_start = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            
            # Find next scan time
            if now < today_start:
                # If current time is before today's start, next scan is today's start
                next_scan = today_start
            else:
                # Calculate how many intervals have passed since start
                elapsed = now - today_start
                elapsed_hours = elapsed.total_seconds() / 3600
                intervals_passed = int(elapsed_hours / scan_interval_hours)
                
                # Next scan is the next interval
                next_scan = today_start + timedelta(hours=(intervals_passed + 1) * scan_interval_hours)
                
                # If next scan is tomorrow, move to next day's start time
                if next_scan.date() > now.date():
                    tomorrow_start = (now + timedelta(days=1)).replace(
                        hour=start_hour, minute=start_minute, second=0, microsecond=0
                    )
                    next_scan = tomorrow_start
            
            self.logger.debug(f"Next scan calculated: {next_scan}")
            return next_scan
        except Exception as e:
            self.logger.error(f"Error calculating next scan time: {e}")
            # Fallback: scan in 1 hour
            return datetime.now() + timedelta(hours=1)
    
    def is_scan_time(self) -> bool:
        """Check if it's time to scan (within 1 minute of scheduled time)"""
        try:
            next_scan = self.calculate_next_scan_time()
            now = datetime.now()
            
            # Check if we're within 5 minute of scan time
            time_diff = abs((next_scan - now).total_seconds())
            return time_diff <= 300
        except Exception as e:
            self.logger.error(f"Error checking scan time: {e}")
            return False

    def wait_for_next_scan(self) -> datetime:
        """Wait until next scheduled scan time"""
        try:
            next_scan = self.calculate_next_scan_time()
            now = datetime.now()

            if next_scan > now:
                wait_seconds = (next_scan - now).total_seconds()
                
                # Check if it's optimal trading time
                is_optimal, session = self.is_optimal_trading_time()
                session_info = f" ({session})" if session != "unknown" else ""
                
                self.logger.info(f"⏰ Next scan scheduled for: {next_scan}{session_info}")
                self.logger.info(f"⏳ Waiting {wait_seconds / 60:.1f} minutes...")

                time.sleep(wait_seconds)

            return next_scan
        except Exception as e:
            self.logger.error(f"Error waiting for next scan: {e}")
            time.sleep(3600)  # Wait 1 hour on error
            return datetime.now()

# ========================================
# PHASE 8: NOTIFICATION HELPERS
# ========================================

def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2
    FIXED: Prevents Telegram parsing errors
    """
    if not text:
        return ""
    
    # Convert to string if it's a number
    text = str(text)
    
    # Characters that need escaping in MarkdownV2
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    return text

def format_price(price: float) -> str:
    """Format price with proper escaping"""
    if price == 0:
        return "0"
    
    formatted = f"{price:.6f}"
    # Remove trailing zeros after decimal
    formatted = formatted.rstrip('0').rstrip('.')
    return escape_markdown(formatted)

def format_percentage(value: float) -> str:
    """Format percentage with proper escaping"""
    formatted = f"{value:.1f}%"
    return escape_markdown(formatted)


# ========================================
# PHASE 9: MAIN AUTO-TRADER CLASS
# ========================================

class AutoTrader:
    """
    Main auto-trading orchestration class
    ENHANCED: Trailing stops integrated with partial profit milestones
    """
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize database components
        self.db_manager = DatabaseManager(config.db_config.get_database_url())
        self.enhanced_db_manager = EnhancedDatabaseManager(self.db_manager)
        
        # Initialize trading system
        self.trading_system = CompleteEnhancedBybitSystem(config)
        self.exchange = self.trading_system.exchange_manager.exchange
        
        if not self.exchange:
            raise Exception("Failed to initialize exchange connection")
        
        # Initialize auto-trading components
        self.leverage_manager = LeverageManager(self.exchange)
        self.position_sizer = PositionSizer(self.exchange)
        self.schedule_manager = ScheduleManager(config)
        self.position_manager = PositionManager(self.exchange, config, self.db_manager)
        self.profit_monitor = LeveragedProfitMonitor(self.exchange, config)
        self.order_executor = OrderExecutor(
            self.exchange, config, self.leverage_manager, self.position_sizer
        )
        
        # Session tracking
        self.trading_session_id = None
        self.is_running = False
    
    def start_trading_session(self) -> str:
        """Start new auto-trading session"""
        try:
            session_id = f"autotrader_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trading_session_id = session_id
            
            # Save session to database
            session = self.db_manager.get_session()
            
            auto_session = AutoTradingSession(
                session_id=session_id,
                config_snapshot=self.config.to_dict()
            )
            
            session.add(auto_session)
            session.commit()
            session.close()
            
            self.logger.debug(f"🚀 Started enhanced auto-trading session: {session_id}")
            self.logger.debug(f"⚙️ Enhanced Configuration:")
            self.logger.debug(f"   Max concurrent positions: {self.config.max_concurrent_positions}")
            self.logger.debug(f"   Max executions per scan: {self.config.max_execution_per_trade}")
            self.logger.debug(f"   Base risk per trade: {self.config.risk_amount}% (adaptive)")
            self.logger.debug(f"   Leverage: {self.config.leverage}")
            self.logger.debug(f"   Auto-close profit target: {self.config.auto_close_profit_at}%")
            self.logger.debug(f"   Auto-close loss target: {self.config.auto_close_loss_at}%")
            self.logger.debug(f"   Scan interval: {self.config.scan_interval / 3600:.1f} hours")
            self.logger.debug(f"   🔥 Portfolio heat monitoring: Active")
            self.logger.debug(f"   🎯 PARTIAL PROFIT MILESTONES: 50% at 100%, 200%, 300% leveraged profit")
            self.logger.debug(f"   💰 TRAILING STOPS: Auto-activate at partial profit milestones")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start trading session: {e}")
            raise
    
    async def run_scan_and_execute(self) -> Tuple[int, int]:
        """
        Enhanced scan and execute with DEFERRED chart generation and notifications
        Charts and notifications happen AFTER trade execution
        """
        try:
            # Check if it's optimal trading time
            is_optimal, session = self.schedule_manager.is_optimal_trading_time()
            if session == "low_liquidity":
                self.logger.info("⏰ Low liquidity hours - proceeding with caution")
            
            self.logger.info("📊 Running enhanced MTF signal analysis...")
            
            # ===== PHASE 1: ENHANCED SIGNAL ANALYSIS =====
            results = self.trading_system.run_complete_analysis_parallel_mtf()
            
            if not results or not results.get('signals'):
                self.logger.warning("No signals generated in this scan")
                return 0, 0
            
            signals_count = len(results['signals'])
            self.logger.debug(f"📈 Generated {signals_count} signals")
            
            # Log MTF analysis results
            self._log_mtf_analysis_results(results)
            
            # ===== PHASE 2: FILTER EXISTING POSITIONS & ORDERS =====
            print()
            self.logger.info("=" * 80)
            self.logger.info("🔍 FILTERING OUT EXISTING POSITIONS & ORDERS...")
            
            already_ranked_signals = results.get('signals', [])
            signals_left_after_filter_out_existing_orders = self.position_manager.filter_signals_by_existing_symbols(already_ranked_signals)
            
            if not signals_left_after_filter_out_existing_orders:
                # Save original results to database first
                save_result = self.enhanced_db_manager.save_all_results(results)
                
                self.logger.warning("⚠️ No signals available for execution after filtering existing symbols")
                self.logger.info("⚠️  ALL SIGNALS FILTERED OUT:")
                self.logger.info("   Either symbols already have positions/orders")
                self.logger.info("   or no valid signals generated")
                print()
                return signals_count, 0
            
            print()
            self.logger.info("=" * 80 + "\n")

            # ===== PHASE 3: UPDATE RESULTS WITH FILTERED DATA =====
            filtered_results = self._update_results_with_filtering(
                results, already_ranked_signals, signals_left_after_filter_out_existing_orders
            )
            
            # ===== PHASE 4: SAVE FILTERED RESULTS TO DATABASE =====
            self.logger.debug("💾 Saving filtered results to MySQL database...")
            save_result = self.enhanced_db_manager.save_all_results(filtered_results)
            if save_result.get('error'):
                self.logger.error(f"Failed to save results: {save_result['error']}")
            else:
                self.logger.debug(f"✅ Filtered results saved - Scan ID: {save_result.get('scan_id', 'Unknown')}")
            
            # ===== PHASE 5: DISPLAY RESULTS TABLE =====
            self.trading_system.print_comprehensive_results_with_mtf(filtered_results)
            
            # ===== PHASE 6: CHECK POSITION AVAILABILITY =====
            can_trade, available_slots = self.position_manager.can_open_new_positions(
                self.config.max_execution_per_trade
            )
            
            if not can_trade:
                portfolio_heat = self.position_manager.calculate_portfolio_heat()
                self.logger.warning(
                    f"⚠️ Cannot open new positions - at max capacity or portfolio heat too high "
                    f"({self.config.max_concurrent_positions} positions, {portfolio_heat:.1f}% heat)"
                )
                self.logger.info(f"⚠️  POSITION LIMIT REACHED:")
                self.logger.info(f"   Max positions: {self.config.max_concurrent_positions}")
                self.logger.info(f"   Portfolio heat: {portfolio_heat:.1f}%")
                self.logger.info(f"   No new trades will be executed until positions are closed")
                
                # NOTE: Even if we can't execute, we DON'T send signal notifications here
                # No charts or notifications without execution
                return signals_count, 0
            
            # ===== PHASE 7: EXECUTE TRADES =====
            execution_count = min(available_slots, self.config.max_execution_per_trade, len(signals_left_after_filter_out_existing_orders))
            selected_opportunities = signals_left_after_filter_out_existing_orders[:execution_count]
            
            # Enhanced execution logging
            self._log_enhanced_execution_plan(
                signals_count, 
                len(signals_left_after_filter_out_existing_orders), 
                available_slots, 
                execution_count
            )
            
            executed_trades = []  # Track successfully executed trades
            executed_count = 0
            
            # Execute selected trades and track successful ones
            if self.config.auto_execute_trades:
                executed_trades, executed_count = await self._execute_enhanced_trades_with_tracking(
                    selected_opportunities, save_result.get('scan_session_id')
                )
            else:
                self.logger.info("📄 Auto-execution disabled - no charts or notifications will be sent")
            
            # ===== PHASE 8: GENERATE CHARTS FOR EXECUTED TRADES ONLY =====
            if executed_trades:
                self.logger.info(f"📊 PHASE 8: Generating charts for {len(executed_trades)} executed trades...")
                charts_generated = self.trading_system.generate_charts_for_top_signals(executed_trades)
                self.logger.info(f"✅ Generated {charts_generated} charts for executed trades")
            else:
                self.logger.info("📊 PHASE 8: No executed trades - skipping chart generation")
            
            # ===== PHASE 9: SEND NOTIFICATIONS FOR EXECUTED TRADES ONLY =====
            if executed_trades:
                self.logger.info(f"📢 PHASE 9: Sending notifications for {len(executed_trades)} executed trades...")
                await self._send_executed_trade_notifications(executed_trades)
            else:
                self.logger.info("📢 PHASE 9: No executed trades - skipping notifications")
            
            # ===== PHASE 10: FINAL SUMMARY =====
            self._log_enhanced_execution_summary(signals_count, len(signals_left_after_filter_out_existing_orders), executed_count)
            
            return signals_count, executed_count
            
        except Exception as e:
            self.logger.error(f"Error in enhanced scan and execute: {e}")
            raise

    async def _execute_enhanced_trades_with_tracking(self, selected_opportunities: List[Dict], scan_session_id: str) -> Tuple[List[Dict], int]:
        """
        Execute trades and return list of successfully executed trades
        Returns: (list_of_executed_trades, count_of_executed_trades)
        """
        try:
            self.logger.info("🚀 Starting enhanced trade execution with tracking...")
            executed_trades = []
            executed_count = 0
            
            for i, opportunity in enumerate(selected_opportunities):
                try:
                    symbol = opportunity['symbol']
                    mtf_status = opportunity.get('mtf_status', 'UNKNOWN')
                    entry_strategy = opportunity.get('entry_strategy', 'immediate')
                    mtf_validated = opportunity.get('mtf_validated', False)
                    analysis_method = opportunity.get('analysis_method', 'unknown')
                    
                    # Enhanced execution logging
                    validation_emoji = "🎯" if mtf_validated else "⚠️"
                    self.logger.info(f"🔍 {validation_emoji} Executing trade {i+1}/{len(selected_opportunities)}: {symbol}")
                    self.logger.info(f"     MTF Status: {mtf_status}")
                    self.logger.info(f"     Entry Strategy: {entry_strategy}")
                    self.logger.info(f"     Analysis Method: {analysis_method}")
                    self.logger.info(f"     🎯 Partial Profit Milestones: 100%, 200%, 300% leveraged profit")
                    self.logger.info(f"     💰 Trailing Stops: Auto-activate at milestones")
                    
                    # Execute trade using enhanced order executor
                    success, message, position_data = self.order_executor.place_leveraged_order(
                        opportunity, self.config.risk_amount, self.config.leverage
                    )
                    
                    if success:
                        # Save position to database
                        position_id = self.position_manager.save_position_to_database(position_data, scan_session_id)
                        
                        if not position_id:
                            self.logger.error(f"Failed to save position data for {symbol}")
                            continue

                        executed_count += 1
                        
                        # Add execution data to opportunity for later use
                        opportunity['execution_status'] = 'success'
                        opportunity['position_id'] = position_id
                        opportunity['position_data'] = position_data
                        opportunity['execution_message'] = message
                        
                        # Add to executed trades list
                        executed_trades.append(opportunity)
                        
                        # Enhanced success logging
                        self.logger.info(f"✅ Trade {i+1} executed successfully")
                        self.logger.info(f"   Position ID: {position_id}")
                        self.logger.info(f"   Entry: ${opportunity['entry_price']:.6f}")
                        self.logger.info(f"   Risk: {position_data.get('risk_percentage', self.config.risk_amount):.1f}% (adaptive)")
                        self.logger.info(f"   MTF Validated: {mtf_validated}")
                        self.logger.info(f"   Partial Milestones: Active at {position_data['leverage']}x leverage")
                        
                    else:
                        # self.logger.info(f"   ❌ TRADE FAILED!")
                        # self.logger.info(f"   Error: {message}")
                        self.logger.error(f"❌ Trade execution failed: {message}")
                        
                        # Track failed execution
                        opportunity['execution_status'] = 'failed'
                        opportunity['execution_message'] = message
                        
                except Exception as e:
                    error_msg = f"Error executing trade for {opportunity['symbol']}: {e}"
                    self.logger.error(f"   ❌ {error_msg}")
                    # self.logger.error(error_msg)
                    opportunity['execution_status'] = 'error'
                    opportunity['execution_message'] = str(e)
            
            self.logger.info(f"🏁 Enhanced execution completed: {executed_count}/{len(selected_opportunities)} successful")
            return executed_trades, executed_count
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trade execution with tracking: {e}")
            return [], 0

    async def _send_executed_trade_notifications(self, executed_trades: List[Dict]):
        """
        Send notifications ONLY for successfully executed trades
        Includes both signal notification and trade confirmation
        """
        try:
            notification_count = 0
            
            for trade in executed_trades:
                try:
                    symbol = trade['symbol']
                    position_data = trade.get('position_data', {})
                    
                    # Send combined notification (signal + execution confirmation)
                    await self.send_executed_trade_notification(trade, position_data)
                    notification_count += 1
                    
                    self.logger.debug(f"✅ Notification sent for executed trade: {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Error sending notification for {trade['symbol']}: {e}")
            
            self.logger.info(f"📱 Sent {notification_count}/{len(executed_trades)} trade notifications")
            
        except Exception as e:
            self.logger.error(f"Error sending executed trade notifications: {e}")

    async def send_executed_trade_notification(self, trade: Dict, position_data: Dict):
        """
        Send comprehensive notification for executed trade
        Combines signal info with execution confirmation and partial milestone info
        """
        try:
            symbol = escape_markdown(trade['symbol'])
            chart_file = trade.get('chart_file', '')
            side = trade['side'].upper()
            entry_price = format_price(trade['entry_price'])
            take_profit_1 = format_price(trade.get('take_profit_1', 0))
            take_profit_2 = format_price(trade.get('take_profit_2', 0))
            stop_loss = format_price(trade.get('stop_loss', 0))
            risk_reward = trade.get('risk_reward_ratio', 0)
            confidence = trade.get('confidence', 0)
            mtf_status = escape_markdown(trade.get('mtf_status', 'N/A'))
            mtf_validated = trade.get('mtf_validated', False)
            
            # Execution details
            position_size = position_data.get('position_size', 0)
            leverage = position_data.get('leverage', 0)
            risk_usdt = position_data.get('risk_amount', 0)
            adaptive_risk = position_data.get('risk_percentage', self.config.risk_amount)
            position_id = trade.get('position_id', 'unknown')
            
            validation_emoji = "🎯" if mtf_validated else "⚠️"
            
            # Build comprehensive message
            message_parts = []
            message_parts.append(f"🚀 *{validation_emoji} TRADE EXECUTED SUCCESSFULLY*")
            message_parts.append("")
            
            if side == 'BUY':
                message_parts.append(f"🟢 *LONG {symbol}*")
            else:
                message_parts.append(f"🔴 *SHORT {symbol}*")
            
            message_parts.append("")
            message_parts.append(f"📊 *POSITION DETAILS:*")
            message_parts.append(f"    💰 Entry Price: ${entry_price}")
            message_parts.append(f"    📈 Position Size: {escape_markdown(f'{position_size:.4f}')} units")
            message_parts.append(f"    ⚡ Leverage: {escape_markdown(str(leverage))}x")
            message_parts.append(f"    💵 Risk Amount: {escape_markdown(f'{risk_usdt:.2f}')} USDT")
            message_parts.append(f"    📊 Risk Percentage: {format_percentage(adaptive_risk)}")
            message_parts.append("")
            
            message_parts.append(f"🎯 *TARGETS & STOPS:*")
            message_parts.append(f"    🎯 Take Profit 1: ${take_profit_1}")
            message_parts.append(f"    🎯 Take Profit 2: ${take_profit_2}")
            message_parts.append(f"    🚫 Stop Loss: ${stop_loss}")
            message_parts.append(f"    📊 Risk/Reward: {escape_markdown(f'{risk_reward:.2f}')}:1")
            message_parts.append("")
            
            # Add partial profit milestone information
            message_parts.append(f"💰 *PARTIAL PROFIT MILESTONES:*")
            message_parts.append(f"    🎯 Leverage: {escape_markdown(str(leverage))}x")
            message_parts.append(f"    1️⃣ 50% at 100% leveraged profit → SL to Break Even")
            message_parts.append(f"    2️⃣ 50% at 200% leveraged profit → SL to 100% profit")
            message_parts.append(f"    3️⃣ 50% at 300% leveraged profit → SL to 200% profit")
            message_parts.append(f"    📈 Trailing stops auto\\-activate at milestones")
            message_parts.append("")
            
            message_parts.append(f"📋 *SIGNAL QUALITY:*")
            validation_status = "✅" if mtf_validated else "❌"
            message_parts.append(f"    {validation_status} MTF Validated: {mtf_status}")
            message_parts.append(f"    🎯 Confidence: {format_percentage(confidence)}")
            message_parts.append("")
            
            message_parts.append(f"🆔 Position ID: {escape_markdown(position_id)}")
            message_parts.append(f"🕐 {escape_markdown(datetime.now().strftime('%H:%M:%S'))}")
            
            # Add chart info if available
            if chart_file and chart_file != "Chart data unavailable":
                message_parts.append("")
                message_parts.append("📊 Chart generated and attached")
            
            # Join message parts
            message = "\n".join(message_parts)
            
            # Position management keyboard
            keyboard = [
                [{"text": "📊 Check Position", "callback_data": f"check_pos_{symbol}"}],
                [{"text": "🔴 Close Position", "callback_data": f"close_pos_{symbol}"}]
            ]
            
            # Send notification with chart if available
            if chart_file and chart_file.endswith('.png') and chart_file.startswith('charts/'):
                await send_trading_notification(self.config, message, keyboard, image_path=chart_file)
            else:
                await send_trading_notification(self.config, message, keyboard)
            
        except Exception as e:
            self.logger.error(f"Failed to send executed trade notification: {e}")

    def _log_mtf_analysis_results(self, results: Dict):
        """Log enhanced MTF analysis results"""
        try:
            signals = results.get('signals', [])
            system_performance = results.get('system_performance', {})
            scan_info = results.get('scan_info', {})
            
            # MTF validation metrics
            mtf_validated_signals = system_performance.get('mtf_validated_signals', 0)
            traditional_signals = system_performance.get('traditional_signals', 0)
            mtf_validation_rate = system_performance.get('mtf_validation_rate', 0)
            structure_filtered_rate = system_performance.get('structure_filtered_rate', 0)
            
            # Analysis method
            method = scan_info.get('method', 'unknown')
            execution_time = scan_info.get('execution_time_seconds', 0)
            
            self.logger.info(f"🎯 Enhanced MTF Analysis Complete:")
            self.logger.info(f"   Method: {method.upper()}")
            self.logger.info(f"   Execution Time: {execution_time:.1f}s")
            self.logger.info(f"   Total Signals: {len(signals)}")
            self.logger.info(f"   MTF Validated: {mtf_validated_signals}")
            self.logger.info(f"   Traditional: {traditional_signals}")
            self.logger.info(f"   MTF Validation Rate: {mtf_validation_rate:.1f}%")
            self.logger.info(f"   Structure Filtering: {structure_filtered_rate:.1f}% symbols filtered")
            
            # Sample signal analysis for debugging
            if signals and len(signals) >= 3:
                self._debug_enhanced_signals(signals[:3])
                
        except Exception as e:
            self.logger.error(f"Error logging MTF analysis results: {e}")

    def _debug_enhanced_signals(self, sample_signals: List[Dict]):
        """Debug logging for enhanced signals with MTF details"""
        try:
            self.logger.debug("\n🔍 ENHANCED SIGNAL ANALYSIS:")
            for i, signal in enumerate(sample_signals, 1):
                symbol = signal['symbol']
                mtf_validated = signal.get('mtf_validated', False)
                mtf_status = signal.get('mtf_status', 'UNKNOWN')
                entry_strategy = signal.get('entry_strategy', 'immediate')
                analysis_method = signal.get('analysis_method', 'unknown')
                
                # Risk/reward analysis
                original_rr = signal.get('original_risk_reward_ratio', signal.get('risk_reward_ratio', 0))
                final_rr = signal.get('risk_reward_ratio', 0)
                
                # MTF details
                analysis_details = signal.get('analysis_details', {})
                mtf_trend = analysis_details.get('mtf_trend', 'unknown')
                structure_timeframe = analysis_details.get('structure_timeframe', 'unknown')
                
                self.logger.debug(f"  {i}. {symbol}:")
                self.logger.debug(f"     🎯 MTF Status: {'✅ VALIDATED' if mtf_validated else '❌ TRADITIONAL'} ({mtf_status})")
                self.logger.debug(f"     📊 Analysis: {analysis_method}")
                self.logger.debug(f"     📈 Trend Context: {mtf_trend} ({structure_timeframe})")
                self.logger.debug(f"     🎪 Entry Strategy: {entry_strategy}")
                self.logger.debug(f"     💰 R/R Ratio: {original_rr:.2f} → {final_rr:.2f}")
                self.logger.debug(f"     🎯 Confidence: {signal['confidence']:.1f}%")
                self.logger.debug(f"     📋 TP Level: {self.config.default_tp_level}")
                self.logger.debug(f"     📈 Partial Milestones: 100%, 200%, 300% leveraged profit")
                self.logger.debug()
                
        except Exception as e:
            self.logger.error(f"Error in enhanced signal debugging: {e}")

    def _update_results_with_filtering(self, results: Dict, original_signals: List[Dict], 
                                    filtered_signals: List[Dict]) -> Dict:
        """Update results structure with filtering information"""
        try:
            # Get filtered symbols for chart generation
            filtered_symbol_set = set(opp['symbol'] for opp in filtered_signals)
            top_filtered_signals = [signal for signal in original_signals if signal['symbol'] in filtered_symbol_set]
            
            # Update results structure
            updated_results = results.copy()
            updated_results['top_opportunities'] = filtered_signals  # For display
            updated_results['signals'] = top_filtered_signals  # For chart generation
            
            # Update scan info to reflect filtering
            if 'scan_info' not in updated_results:
                updated_results['scan_info'] = {}
                
            scan_info = updated_results['scan_info']
            scan_info['original_signals_count'] = len(original_signals)
            scan_info['total_signals_left_after_filter_out_existing_orders'] = len(filtered_signals)
            scan_info['displayed_opportunities'] = len(filtered_signals)
            scan_info['filtered_signals_count'] = len(top_filtered_signals)
            scan_info['already_ranked_signals_count'] = len(original_signals)
            scan_info['symbols_filtered_out'] = len(original_signals) - len(filtered_signals)
            
            return updated_results
            
        except Exception as e:
            self.logger.error(f"Error updating results with filtering: {e}")
            return results

    def _log_enhanced_execution_plan(self, original_signals: int, available_signals: int, 
                                available_slots: int, execution_count: int):
        """Enhanced execution plan logging"""
        try:
            portfolio_heat = self.position_manager.calculate_portfolio_heat()
            
            print()
            self.logger.info("=" * 80 + "\n")
            self.logger.info(f"🎯 ENHANCED EXECUTION PLAN:")
            self.logger.info(f"   Original Signals: {original_signals}")
            self.logger.info(f"   Total After Position/Order Filter: {available_signals}")
            self.logger.info(f"   Available Position Slots: {available_slots}")
            self.logger.info(f"   Portfolio Heat: {portfolio_heat:.1f}%")
            self.logger.info(f"   Selected for Execution: {execution_count}")
            self.logger.info(f"   Auto-Execute: {'✅ Enabled' if self.config.auto_execute_trades else '❌ Disabled'}")
            self.logger.info(f"   🎯 Partial Profit Milestones: 100%, 200%, 300% leveraged profit")
            self.logger.info(f"   💰 Trailing Stops: Auto-activate at milestones")
            
            self.logger.debug(
                f"🎯 Enhanced execution: {execution_count} trades from {available_signals} available "
                f"(slots: {available_slots}, heat: {portfolio_heat:.1f}%, original: {original_signals})"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging enhanced execution plan: {e}")

    def _log_enhanced_execution_summary(self, original_signals: int, available_signals: int, executed_count: int):
        """Enhanced execution summary with MTF metrics"""
        try:
            filter_rate = ((original_signals - available_signals) / original_signals * 100) if original_signals > 0 else 0
            execution_rate = (executed_count / available_signals * 100) if available_signals > 0 else 0
            portfolio_heat = self.position_manager.calculate_portfolio_heat()

            print()
            self.logger.info("=" * 80 + "\n")
            self.logger.info(f"📊 ENHANCED EXECUTION SUMMARY:")
            self.logger.info(f"   Original Signals: {original_signals}")
            self.logger.info(f"   Available After Filtering: {available_signals} ({100-filter_rate:.1f}%)")
            self.logger.info(f"   Successfully Executed: {executed_count} ({execution_rate:.1f}%)")
            self.logger.info(f"   Portfolio Heat: {portfolio_heat:.1f}%")
            self.logger.info(f"   Filter Efficiency: {filter_rate:.1f}% (prevents overexposure)")
            self.logger.info(f"   Risk Management: Enhanced (adaptive sizing + correlation filters)")
            self.logger.info(f"   🎯 Partial Milestones: 50% at 100%, 200%, 300% leveraged profit")
            self.logger.info(f"   💰 Trailing Stops: Auto-update at partial milestones")
            print()
            self.logger.info("=" * 80)
            
            self.logger.debug(f"🏁 Enhanced scan summary: {original_signals} → {available_signals} → {executed_count} trades")
            self.logger.debug(f"   📈 Filter rate: {filter_rate:.1f}% (risk management)")
            self.logger.debug(f"   🎯 Execution rate: {execution_rate:.1f}% (of available)")
            self.logger.debug(f"   🔥 Portfolio heat: {portfolio_heat:.1f}%")
            self.logger.debug(f"   📊 Partial milestones active for all positions")
            self.logger.debug(f"   💰 Trailing stops will activate at partial milestones")
            
        except Exception as e:
            self.logger.error(f"Error logging enhanced execution summary: {e}")

    def main_trading_loop(self):
        """
        Main auto-trading loop
        ENHANCED: With partial profit milestones and integrated trailing stops
        """
        try:
            self.is_running = True
            session_id = self.start_trading_session()
            
            # Start enhanced profit monitoring with partial milestones and trailing stops
            if self.config.auto_close_enabled:
                self.logger.debug("📈 Starting enhanced profit monitoring:")
                self.logger.debug("   - Partial profits at 100%, 200%, 300% leveraged profit")
                self.logger.debug("   - Trailing stops auto-activate at partial milestones")
                self.profit_monitor.start_monitoring()
            
            self.logger.info("🤖 Enhanced auto-trading loop started")
            self.logger.info("🎯 Partial profit milestones active")
            self.logger.info("💰 Trailing stops will activate at milestones")
            
            while self.is_running:
                try:
                    # Wait for next scheduled scan
                    next_scan_time = self.schedule_manager.wait_for_next_scan()
                    
                    if not self.is_running:
                        break
                    
                    # Check trading time optimality
                    is_optimal, session = self.schedule_manager.is_optimal_trading_time()
                    session_info = f" ({session})" if session != "unknown" else ""
                    
                    self.logger.info(f"⏰ Scan time reached: {next_scan_time}{session_info}")
                    
                    # Run enhanced scan and execute trades with proper async handling
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        total_signals, executed_count = loop.run_until_complete(
                            self.run_scan_and_execute()
                        )
                    finally:
                        loop.close()
                    
                    self.logger.info(
                        f"📊 Enhanced scan complete - Signals: {total_signals}, "
                        f"Executed: {executed_count}"
                    )
                    
                    # Update session statistics
                    self.update_session_stats(total_signals, executed_count)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Received interrupt signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in enhanced trading loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
            
        except Exception as e:
            self.logger.error(f"Critical error in enhanced trading loop: {e}")
        finally:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop auto-trading"""
        try:
            self.is_running = False
            self.profit_monitor.stop_monitoring()
            
            if self.trading_session_id:
                # Update session end time in database
                session = self.db_manager.get_session()
                auto_session = session.query(AutoTradingSession).filter(
                    AutoTradingSession.session_id == self.trading_session_id
                ).first()
                
                if auto_session:
                    auto_session.ended_at = datetime.utcnow()
                    auto_session.status = 'stopped'
                    session.commit()
                
                session.close()
            
            self.logger.info("🛑 Enhanced auto-trading stopped")
            self.logger.info("🎯 Partial profit milestones deactivated")
            self.logger.info("💰 Trailing stop automation deactivated")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
        
    def update_session_stats(self, total_signals: int, executed_count: int):
        """Update session statistics in database"""
        try:
            if not self.trading_session_id:
                return
            
            session = self.db_manager.get_session()
            auto_session = session.query(AutoTradingSession).filter(
                AutoTradingSession.session_id == self.trading_session_id
            ).first()
            
            if auto_session:
                auto_session.total_scans += 1
                auto_session.total_trades_placed += executed_count
                auto_session.last_scan_at = datetime.utcnow()
                auto_session.next_scan_at = self.schedule_manager.calculate_next_scan_time()
                
                session.commit()
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error updating session stats: {e}")


# ========================================
# PHASE 10: MAIN EXECUTION
# ========================================

def main():
    """
    Main function for running the enhanced auto-trader
    ENHANCED: With partial profit milestones and integrated trailing stops
    """
    try:
        from config.config import DatabaseConfig, EnhancedSystemConfig
        from utils.logging import setup_logging
        
        # Setup logging
        logger = setup_logging("INFO")
        
        # Load configuration
        db_config = DatabaseConfig()  # Load from environment or defaults
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        # Validate auto-trading configuration
        if not config.leverage or config.leverage not in LeverageManager.ACCEPTABLE_LEVERAGE:
            logger.error(f"Invalid leverage configuration: {config.leverage}")
            return
        
        if config.risk_amount <= 0:
            logger.error(f"Invalid risk amount: {config.risk_amount}")
            return
        
        # Start enhanced auto-trader
        logger.info("🚀 Starting Enhanced Auto-Trader v2.1 with:")
        logger.info("   🎯 PARTIAL PROFIT MILESTONES (50% at 100%, 200%, 300% leveraged profit)")
        logger.info("   💰 INTEGRATED TRAILING STOPS (auto-activate at partial milestones)")
        logger.info("   📊 Adaptive Position Sizing")
        logger.info("   🛑 Dynamic Stop Loss Management")
        logger.info("   🔗 Correlation Risk Management")
        logger.info("   📈 Market Regime Awareness")
        logger.info("   🔥 Portfolio Heat Monitoring")
        logger.info("   🔧 FIXED: Bybit API integration issues")
        logger.info("")
        logger.info("💰 Partial profit strategy:")
        logger.info("   - 50% of position closed at 100% leveraged profit → SL moves to Break Even")
        logger.info("   - 50% of remaining closed at 200% leveraged profit → SL moves to 100% profit")
        logger.info("   - 50% of remaining closed at 300% leveraged profit → SL moves to 200% profit")
        logger.info("   - Each partial is taken ONCE and tracked to prevent repetition")
        logger.info("")
        logger.info("🎯 Trailing stop integration:")
        logger.info("   - Automatically activates when partial profits are taken")
        logger.info("   - Uses correct Bybit API parameters (triggerPrice, stop_market)")
        logger.info("   - Protects profits while allowing for continued upside")
        logger.info("   - Never moves stops backward (one-way progression)")
        logger.info("")
        logger.info("🔧 Technical fixes:")
        logger.info("   - Fixed Bybit stop order API parameters")
        logger.info("   - Improved error handling for API failures")
        logger.info("   - Enhanced position tracking and synchronization")
        logger.info("   - Better debugging and monitoring logs")
        logger.info("")
        
        auto_trader = AutoTrader(config)
        auto_trader.main_trading_loop()
        
    except KeyboardInterrupt:
        logger.info("Enhanced auto-trader stopped by user")
    except Exception as e:
        logger.error(f"Enhanced auto-trader failed: {e}")
        raise

if __name__ == "__main__":
    main()