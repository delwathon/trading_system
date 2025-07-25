"""
Auto-Trading System for Enhanced Bybit Trading System.
Handles scheduled scanning, position management, and automated trading with leverage support.
UPDATED: Works with TOP OPPORTUNITIES from system.py and updates top_opportunities table
"""

import asyncio
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import uuid
import ccxt

from config.config import EnhancedSystemConfig
from core.system import CompleteEnhancedBybitSystem
from database.models import DatabaseManager, TradingPosition, AutoTradingSession
from utils.database_manager import EnhancedDatabaseManager
from notifier.telegram import send_trading_notification


@dataclass
class PositionData:
    """Data class for position tracking"""
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: float
    risk_amount: float
    stop_loss: float
    take_profit: float
    position_id: str
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class LeverageManager:
    """Handle leverage validation and conversion"""
    
    ACCEPTABLE_LEVERAGE = ['10', '12.5', '25', '50', 'max']
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
    
    def validate_leverage(self, leverage_str: str) -> bool:
        """Validate leverage is acceptable"""
        return leverage_str in self.ACCEPTABLE_LEVERAGE
    
    def convert_leverage_to_float(self, leverage_str: str, symbol: str) -> float:
        """Convert leverage string to float, handling 'max' case"""
        try:
            if leverage_str == 'max':
                # Get maximum leverage for symbol from exchange
                market_info = self.exchange.market(symbol)
                max_leverage = market_info.get('limits', {}).get('leverage', {}).get('max', 100)
                self.logger.debug(f"Max leverage for {symbol}: {max_leverage}")
                return float(max_leverage)
            else:
                return float(leverage_str)
        except Exception as e:
            self.logger.error(f"Error converting leverage {leverage_str} for {symbol}: {e}")
            return 10.0  # Fallback to safe leverage
    
    def set_symbol_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for symbol on exchange"""
        try:
            # Get current leverage first
            try:
                positions = self.exchange.fetch_positions([symbol])
                current_leverage = None
                for pos in positions:
                    if pos['symbol'] == symbol:
                        current_leverage = pos.get('leverage', 1)
                        break
                
                # If leverage is already set correctly, skip
                if current_leverage and abs(float(current_leverage) - leverage) < 0.1:
                    self.logger.debug(f"✅ Leverage already set to {current_leverage}x for {symbol}")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"Could not check current leverage for {symbol}: {e}")
            
            # Set leverage on Bybit
            result = self.exchange.set_leverage(leverage, symbol)
            self.logger.debug(f"✅ Set leverage {leverage}x for {symbol}")
            return True
        except Exception as e:
            error_msg = str(e)
            # If leverage not modified error, it might already be set correctly
            if "110043" in error_msg or "leverage not modified" in error_msg:
                self.logger.warning(f"⚠️ Leverage not modified for {symbol} - may already be set correctly")
                return True  # Continue with trade execution
            else:
                self.logger.error(f"❌ Failed to set leverage {leverage}x for {symbol}: {e}")
                return False
    
    def get_max_leverage_for_symbol(self, symbol: str) -> float:
        """Get maximum available leverage for symbol"""
        try:
            market_info = self.exchange.market(symbol)
            return float(market_info.get('limits', {}).get('leverage', {}).get('max', 100))
        except Exception as e:
            self.logger.error(f"Error getting max leverage for {symbol}: {e}")
            return 50.0  # Conservative fallback


class PositionSizer:
    """Calculate position sizes with leverage"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, risk_amount_pct: float, leverage: float, entry_price: float) -> float:
        """
        Calculate position size using: (account_balance × risk_percentage × leverage) / entry_price
        
        Args:
            risk_amount_pct: Risk percentage of account balance (e.g., 5.0 for 5%)
            leverage: Trading leverage (e.g., 25)
            entry_price: Entry price for the position
            
        Returns:
            Position size in base currency units
        """
        try:
            if entry_price <= 0 or leverage <= 0:
                raise ValueError(f"Invalid entry_price ({entry_price}) or leverage ({leverage})")
            
            # Get current account balance
            account_balance = self.get_available_balance()
            if account_balance <= 0:
                raise ValueError(f"Invalid account balance: {account_balance}")
            
            # Calculate risk amount in USDT
            risk_amount_usdt = account_balance * (risk_amount_pct / 100)
            
            # Calculate position size: (risk_amount_usdt × leverage) / entry_price
            position_size = (risk_amount_usdt * leverage) / entry_price
            
            self.logger.debug(f"Position size calculation:")
            self.logger.debug(f"  Account Balance: {account_balance} USDT")
            self.logger.debug(f"  Risk Percentage: {risk_amount_pct}%")
            self.logger.debug(f"  Risk Amount: {risk_amount_usdt} USDT")
            self.logger.debug(f"  Leverage: {leverage}x")
            self.logger.debug(f"  Entry Price: {entry_price}")
            self.logger.debug(f"  Position Size: ({risk_amount_usdt} × {leverage}) / {entry_price} = {position_size}")
            
            return position_size
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def validate_position_size(self, symbol: str, size: float) -> Tuple[bool, float]:
        """Validate position size against exchange limits"""
        try:
            market_info = self.exchange.market(symbol)
            limits = market_info.get('limits', {})
            amount_limits = limits.get('amount', {})
            
            min_size = amount_limits.get('min', 0.001)
            max_size = amount_limits.get('max', 1000000)
            
            if size < min_size:
                self.logger.warning(f"Position size {size} below minimum {min_size} for {symbol}")
                return False, min_size
            
            if size > max_size:
                self.logger.warning(f"Position size {size} above maximum {max_size} for {symbol}")
                return False, max_size
            
            return True, size
        except Exception as e:
            self.logger.error(f"Error validating position size for {symbol}: {e}")
            return False, 0.0
    
    def get_available_balance(self) -> float:
        """Get available USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0


class ScheduleManager:
    """Handle scan timing and scheduling logic"""
    
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
            
            # Check if we're within 1 minute of scan time
            time_diff = abs((next_scan - now).total_seconds())
            return time_diff <= 60
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
                self.logger.info(f"⏰ Next scan scheduled for: {next_scan}")
                self.logger.info(f"⏳ Waiting {wait_seconds / 60:.1f} minutes...")
                
                time.sleep(wait_seconds)
            
            return next_scan
        except Exception as e:
            self.logger.error(f"Error waiting for next scan: {e}")
            time.sleep(3600)  # Wait 1 hour on error
            return datetime.now()


class LeveragedProfitMonitor:
    """Monitor leveraged profits and auto-close positions"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
    
    def calculate_leveraged_profit_pct(self, position: PositionData, current_price: float) -> float:
        """Calculate profit percentage considering leverage"""
        try:
            if position.entry_price <= 0:
                return 0.0
            
            # Calculate price change percentage
            price_change_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # Apply leverage multiplier
            leveraged_profit_pct = price_change_pct * position.leverage
            
            # Consider position side
            if position.side.lower() == 'sell':
                leveraged_profit_pct = -leveraged_profit_pct
            
            return leveraged_profit_pct
        except Exception as e:
            self.logger.error(f"Error calculating leveraged profit for {position.symbol}: {e}")
            return 0.0
    
    def should_auto_close(self, position: PositionData, current_price: float) -> bool:
        """Check if position should be auto-closed based on profit target"""
        try:
            leveraged_profit_pct = self.calculate_leveraged_profit_pct(position, current_price)
            return leveraged_profit_pct >= self.config.auto_close_profit_at
        except Exception as e:
            self.logger.error(f"Error checking auto-close for {position.symbol}: {e}")
            return False
    
    def close_position(self, position: PositionData) -> bool:
        """Close a position on the exchange"""
        try:
            # Get current position details from exchange
            positions = self.exchange.fetch_positions([position.symbol])
            current_position = None
            
            for pos in positions:
                if pos['symbol'] == position.symbol and pos['contracts'] > 0:
                    current_position = pos
                    break
            
            if not current_position:
                self.logger.warning(f"No active position found for {position.symbol}")
                return False
            
            # Determine correct close side based on current position
            current_side = current_position['side']
            position_size = current_position['contracts']
            
            # Use proper capitalized sides for Bybit API
            if current_side.lower() == 'long':
                close_side = 'Sell'  # Capitalized
            elif current_side.lower() == 'short':
                close_side = 'Buy'   # Capitalized
            else:
                self.logger.error(f"Unknown position side: {current_side}")
                return False
            
            self.logger.debug(f"Closing {current_side} position of {position_size} {position.symbol} with {close_side} order")
            
            # Place market order to close position
            order = self.exchange.create_order(
                symbol=position.symbol,
                type='market',
                side=close_side,  # Use capitalized side
                amount=position_size,
                params={'reduceOnly': True}  # Ensure this closes the position
            )
            
            self.logger.info(f"✅ Closed position {position.symbol} - Profit target reached")
            self.logger.debug(f"Close order: {order}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "110017" in error_msg or "reduce-only order has same side" in error_msg:
                self.logger.error(f"❌ Close order side error for {position.symbol}. Trying alternative method...")
                # Try closing with position close API instead
                try:
                    close_result = self.exchange.close_position(position.symbol)
                    self.logger.info(f"✅ Closed position {position.symbol} using close_position API")
                    return True
                except Exception as e2:
                    self.logger.error(f"❌ Alternative close method also failed for {position.symbol}: {e2}")
            else:
                self.logger.error(f"❌ Failed to close position {position.symbol}: {e}")
            return False
    
    def get_current_positions(self) -> List[PositionData]:
        """Get current positions from exchange"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = []
            
            for pos in positions:
                if pos['contracts'] > 0:  # Active position
                    # Map exchange position side to our format
                    side = 'buy' if pos['side'].lower() == 'long' else 'sell'
                    
                    position_data = PositionData(
                        symbol=pos['symbol'],
                        side=side,
                        size=pos['contracts'],
                        entry_price=pos['entryPrice'],
                        leverage=pos.get('leverage', 1.0),
                        risk_amount=0.0,  # Will be updated from database
                        stop_loss=0.0,
                        take_profit=0.0,
                        position_id=pos.get('id', ''),
                        unrealized_pnl=pos.get('unrealizedPnl', 0.0),
                        unrealized_pnl_pct=pos.get('percentage', 0.0)
                    )
                    active_positions.append(position_data)
            
            return active_positions
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return []
    
    def monitor_positions(self):
        """Main monitoring loop"""
        try:
            while self.monitoring:
                positions = self.get_current_positions()
                
                for position in positions:
                    try:
                        # Get current price
                        ticker = self.exchange.fetch_ticker(position.symbol)
                        current_price = ticker['last']
                        
                        # Check if should auto-close
                        if self.should_auto_close(position, current_price):
                            leveraged_profit = self.calculate_leveraged_profit_pct(position, current_price)
                            self.logger.info(
                                f"🎯 Auto-close triggered for {position.symbol}: "
                                f"{leveraged_profit:.2f}% profit (target: {self.config.auto_close_profit_at}%)"
                            )
                            
                            if self.close_position(position):
                                # Update database record
                                self.update_position_in_database(position, 'closed', 'profit_target')
                        
                    except Exception as e:
                        self.logger.error(f"Error monitoring position {position.symbol}: {e}")
                
                # Sleep between monitoring cycles
                time.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
    
    def start_monitoring(self):
        """Start position monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)
            self.monitor_thread.start()
            self.logger.info("📊 Started position monitoring")
    
    def stop_monitoring(self):
        """Stop position monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛑 Stopped position monitoring")
    
    def update_position_in_database(self, position: PositionData, status: str, close_reason: str = None):
        """Update position status in database"""
        # This would be implemented to update the TradingPosition table
        # Left as placeholder for database integration
        pass


class OrderExecutor:
    """Execute leveraged orders with proper sizing"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig, leverage_manager: LeverageManager, position_sizer: PositionSizer):
        self.exchange = exchange
        self.config = config
        self.leverage_manager = leverage_manager
        self.position_sizer = position_sizer
        self.logger = logging.getLogger(__name__)
    
    def place_leveraged_order(self, signal: Dict, risk_amount_pct: float, leverage_str: str) -> Tuple[bool, str, Dict]:
        """Place a leveraged order based on signal with percentage-based risk"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal.get('take_profit_1', signal.get('take_profit', 0))
            
            self.logger.info(f"🚀 Placing {side.upper()} order for {symbol}")
            
            # Convert leverage to float
            leverage = self.leverage_manager.convert_leverage_to_float(leverage_str, symbol)
            
            # Set leverage on exchange
            if not self.leverage_manager.set_symbol_leverage(symbol, leverage):
                return False, "Failed to set leverage", {}
            
            # Calculate position size using percentage-based risk
            position_size = self.position_sizer.calculate_position_size(risk_amount_pct, leverage, entry_price)
            
            # Validate position size
            is_valid, adjusted_size = self.position_sizer.validate_position_size(symbol, position_size)
            if not is_valid:
                return False, f"Invalid position size: {position_size}", {}
            
            position_size = adjusted_size
            
            # Check available balance and calculate required margin
            available_balance = self.position_sizer.get_available_balance()
            required_margin = (position_size * entry_price) / leverage
            risk_amount_usdt = available_balance * (risk_amount_pct / 100)
            
            if required_margin > available_balance:
                return False, f"Insufficient balance: need {required_margin}, have {available_balance}", {}
            
            self.logger.info(f"💰 Risk calculation:")
            self.logger.info(f"   Account Balance: {available_balance} USDT")
            self.logger.info(f"   Risk Percentage: {risk_amount_pct}%")
            self.logger.info(f"   Risk Amount: {risk_amount_usdt} USDT")
            self.logger.info(f"   Position Size: {position_size} units")
            self.logger.info(f"   Required Margin: {required_margin} USDT")
            
            # Capitalize the side parameter for Bybit API
            bybit_side = side.capitalize()  # 'buy' -> 'Buy', 'sell' -> 'Sell'
            
            # Place entry order with integrated SL/TP
            order_type = 'market' if signal.get('order_type') == 'market' else 'limit'
            
            # Prepare order parameters with integrated SL/TP
            order_params = {}
            if stop_loss > 0:
                order_params['stopLoss'] = stop_loss
            if take_profit > 0:
                order_params['takeProfit'] = take_profit
            
            if order_type == 'market':
                entry_order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=bybit_side,  # Use capitalized side
                    amount=position_size,
                    params=order_params  # Include SL/TP in main order
                )
            else:
                entry_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=bybit_side,  # Use capitalized side
                    amount=position_size,
                    price=entry_price,
                    params=order_params  # Include SL/TP in main order
                )
            
            self.logger.info(f"✅ Entry order placed: {entry_order['id']}")
            if stop_loss > 0:
                self.logger.debug(f"✅ Stop loss integrated: ${stop_loss:.6f}")
            if take_profit > 0:
                self.logger.debug(f"✅ Take profit integrated: ${take_profit:.6f}")
            
            # SL/TP are now integrated - no separate orders needed
            sl_order = None
            tp_order = None
            
            # Create position tracking data
            position_data = {
                'symbol': symbol,
                'side': side,  # Keep original lowercase for internal tracking
                'position_size': position_size,
                'entry_price': entry_price,
                'leverage': leverage,
                'risk_amount': risk_amount_usdt,  # Store actual USDT amount
                'risk_percentage': risk_amount_pct,  # Store percentage for reference
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_order_id': entry_order['id'],
                'stop_loss_order_id': sl_order['id'] if sl_order else None,
                'take_profit_order_id': tp_order['id'] if tp_order else None,
                'signal_confidence': signal.get('confidence', 0),
                'mtf_status': signal.get('mtf_status', ''),
                'auto_close_profit_target': self.config.auto_close_profit_at,
                'quality_tier': signal.get('quality_tier', 'UNKNOWN'),
                'selection_rank': signal.get('selection_rank', 0)
            }
            
            return True, "Order placed successfully", position_data
            
        except Exception as e:
            error_msg = f"Failed to place order for {signal.get('symbol', 'unknown')}: {e}"
            self.logger.error(error_msg)
            return False, error_msg, {}


class PositionManager:
    """Track and manage concurrent positions"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig, db_manager: DatabaseManager):
        self.exchange = exchange
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def get_current_positions_count(self) -> int:
        """Get count of current active positions"""
        try:
            positions = self.exchange.fetch_positions()
            active_count = sum(1 for pos in positions if pos.get('contracts', 0) > 0 or pos.get('size', 0) > 0)
            return active_count
        except Exception as e:
            self.logger.error(f"Error getting positions count: {e}")
            return 0
    
    def get_symbols_with_positions(self) -> Set[str]:
        """Get set of symbols that have active positions"""
        try:
            positions = self.exchange.fetch_positions()
            symbols_with_positions = set()
            
            for pos in positions:
                if pos.get('contracts', 0) > 0 or pos.get('size', 0) > 0:
                    symbols_with_positions.add(pos['symbol'])
            
            return symbols_with_positions
            
        except Exception as e:
            self.logger.error(f"Error getting symbols with positions: {e}")
            return set()
    
    def get_symbols_with_orders(self) -> Set[str]:
        """Get set of symbols that have pending orders"""
        try:
            open_orders = self.exchange.fetch_open_orders()
            symbols_with_orders = set()
            
            for order in open_orders:
                symbols_with_orders.add(order['symbol'])
            
            return symbols_with_orders
            
        except Exception as e:
            self.logger.error(f"Error getting symbols with orders: {e}")
            return set()
    
    def filter_top_opportunities_by_existing_symbols(self, top_opportunities: List[Dict]) -> List[Dict]:
        """Filter out top opportunities for symbols that already have positions or orders"""
        try:
            symbols_with_positions = self.get_symbols_with_positions()
            symbols_with_orders = self.get_symbols_with_orders()
            excluded_symbols = symbols_with_positions.union(symbols_with_orders)
            
            filtered_opportunities = []
            skipped_count = 0
            
            for opportunity in top_opportunities:
                symbol = opportunity['symbol']
                if symbol in excluded_symbols:
                    reason = ""
                    if symbol in symbols_with_positions:
                        reason = "existing position"
                    elif symbol in symbols_with_orders:
                        reason = "pending orders"
                    
                    self.logger.info(f"⏭️ Skipping TOP opportunity {symbol} - {reason}")
                    skipped_count += 1
                else:
                    filtered_opportunities.append(opportunity)
            
            if skipped_count > 0:
                self.logger.info(f"🔍 Filtered out {skipped_count} top opportunities with existing exposure")
                
            return filtered_opportunities
            
        except Exception as e:
            self.logger.error(f"Error filtering top opportunities by existing symbols: {e}")
            return top_opportunities
    
    def can_open_new_positions(self, requested_count: int) -> Tuple[bool, int]:
        """Check if we can open new positions"""
        try:
            current_count = self.get_current_positions_count()
            max_positions = self.config.max_concurrent_positions
            available_slots = max_positions - current_count
            
            if available_slots <= 0:
                return False, 0
            
            can_open = min(requested_count, available_slots)
            return True, can_open
        except Exception as e:
            self.logger.error(f"Error checking position availability: {e}")
            return False, 0
    
    def save_position_to_database(self, position_data: Dict, scan_session_id: int = None) -> str:
        """Save position to database and return position ID"""
        try:
            session = self.db_manager.get_session()
            
            position_id = str(uuid.uuid4())
            
            position = TradingPosition(
                position_id=position_id,
                scan_session_id=scan_session_id,
                symbol=position_data['symbol'],
                side=position_data['side'],
                entry_price=position_data['entry_price'],
                position_size=position_data['position_size'],
                leverage=str(position_data['leverage']),
                risk_amount=position_data['risk_amount'],
                entry_order_id=position_data.get('entry_order_id'),
                stop_loss_order_id=position_data.get('stop_loss_order_id'),
                take_profit_order_id=position_data.get('take_profit_order_id'),
                stop_loss_price=position_data.get('stop_loss', 0),
                take_profit_price=position_data.get('take_profit', 0),
                auto_close_profit_target=position_data.get('auto_close_profit_target', 10.0),
                signal_confidence=position_data.get('signal_confidence'),
                mtf_status=position_data.get('mtf_status'),
                quality_tier=position_data.get('quality_tier', 'UNKNOWN'),
                selection_rank=position_data.get('selection_rank', 0),
                status='open'
            )
            
            session.add(position)
            session.commit()
            session.close()
            
            self.logger.debug(f"Position saved to database: {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error saving position to database: {e}")
            return ""
    
    def update_position_status(self, position_id: str, status: str, close_reason: str = None):
        """Update position status in database"""
        try:
            session = self.db_manager.get_session()
            
            position = session.query(TradingPosition).filter(
                TradingPosition.position_id == position_id
            ).first()
            
            if position:
                position.status = status
                if close_reason:
                    position.close_reason = close_reason
                if status == 'closed':
                    position.closed_at = datetime.utcnow()
                
                session.commit()
                self.logger.debug(f"Updated position {position_id} status to {status}")
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error updating position status: {e}")


class TopOpportunitiesProcessor:
    """Process and validate top opportunities for trading"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_top_opportunity_for_trading(self, opportunity: Dict) -> bool:
        """Validate that a top opportunity is ready for trading - FIXED: More lenient validation"""
        try:
            # Required fields for trading
            required_fields = [
                'symbol', 'side', 'entry_price', 'stop_loss', 'take_profit_1',
                'confidence', 'risk_reward_ratio', 'volume_24h'
            ]
            
            # Check all required fields exist
            for field in required_fields:
                if field not in opportunity:
                    self.logger.warning(f"Missing required field {field} in opportunity {opportunity.get('symbol', 'unknown')}")
                    return False
            
            # FIXED: Much more lenient validation since signals are already filtered
            # These should match or be more lenient than generator.py validation
            
            # Very basic validation - signals are already quality-filtered
            if opportunity['confidence'] < 30:  # FIXED: Much lower (was 40)
                self.logger.warning(f"Confidence too low for {opportunity['symbol']}: {opportunity['confidence']}%")
                return False
            
            if opportunity['risk_reward_ratio'] < 1.2:  # FIXED: Much lower (was 1.5)
                self.logger.warning(f"R/R ratio too low for {opportunity['symbol']}: {opportunity['risk_reward_ratio']}")
                return False
            
            if opportunity['volume_24h'] < 50_000:  # FIXED: Much lower (was 100_000)
                self.logger.warning(f"Volume too low for {opportunity['symbol']}: {opportunity['volume_24h']}")
                return False
            
            # Additional safety checks for trading
            if opportunity['entry_price'] <= 0:
                self.logger.warning(f"Invalid entry price for {opportunity['symbol']}: {opportunity['entry_price']}")
                return False
            
            if opportunity['stop_loss'] <= 0:
                self.logger.warning(f"Invalid stop loss for {opportunity['symbol']}: {opportunity['stop_loss']}")
                return False
            
            self.logger.debug(f"✅ Top opportunity validated for trading: {opportunity['symbol']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating opportunity: {e}")
            return False
    
    def prioritize_top_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Further prioritize top opportunities for execution"""
        try:
            # They should already be ranked, but we can add additional prioritization
            prioritized = []
            
            for opp in opportunities:
                # Add execution priority score
                priority_score = 0
                
                # Quality tier bonus
                quality_tier = opp.get('quality_tier', 'BASIC')
                if quality_tier == 'PREMIUM':
                    priority_score += 100
                elif quality_tier == 'QUALITY':
                    priority_score += 50
                elif quality_tier == 'DECENT':
                    priority_score += 25
                
                # MTF confirmation bonus
                mtf_status = opp.get('mtf_status', 'NONE')
                if mtf_status == 'STRONG':
                    priority_score += 75
                elif mtf_status == 'PARTIAL':
                    priority_score += 35
                
                # Confidence bonus
                confidence = opp.get('confidence', 0)
                priority_score += confidence * 0.5
                
                # Risk-reward bonus
                rr_ratio = opp.get('risk_reward_ratio', 0)
                priority_score += min(50, rr_ratio * 10)
                
                opp['execution_priority_score'] = priority_score
                prioritized.append(opp)
            
            # Sort by execution priority
            prioritized.sort(key=lambda x: x['execution_priority_score'], reverse=True)
            
            return prioritized
            
        except Exception as e:
            self.logger.error(f"Error prioritizing opportunities: {e}")
            return opportunities


class AutoTrader:
    """Main auto-trading orchestration class - FOCUSES ON TOP OPPORTUNITIES ONLY"""
    
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
        self.top_opportunities_processor = TopOpportunitiesProcessor(config)
        
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
            
            self.logger.info(f"🚀 Started auto-trading session: {session_id}")
            self.logger.info(f"⚙️ Configuration:")
            self.logger.info(f"   Max concurrent positions: {self.config.max_concurrent_positions}")
            self.logger.info(f"   Max executions per scan: {self.config.max_execution_per_trade}")
            self.logger.info(f"   Risk amount per trade: {self.config.risk_amount}%")
            self.logger.info(f"   Leverage: {self.config.leverage}")
            self.logger.info(f"   Auto-close profit target: {self.config.auto_close_profit_at}%")
            self.logger.info(f"   Scan interval: {self.config.scan_interval / 3600:.1f} hours")
            self.logger.info(f"🎯 Focus: TOP OPPORTUNITIES ONLY")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start trading session: {e}")
            raise
    
    async def run_scan_and_execute(self) -> Tuple[int, int]:
        """Run signal scan and execute trades - FIXED take_profit key error"""
        try:
            self.logger.info("📊 Running signal analysis...")
            
            # Run the existing signal analysis system
            results = self.trading_system.run_complete_analysis_parallel_mtf()
            
            if not results:
                self.logger.warning("❌ No analysis results returned from system")
                return 0, 0
                
            # Check both 'signals' and 'top_opportunities'
            all_signals = results.get('signals', [])
            opportunities = results.get('top_opportunities', [])
            
            if not all_signals and not opportunities:
                self.logger.warning("❌ No signals or opportunities generated in this scan")
                return 0, 0
            
            # Use all_signals count for accurate reporting
            signals_count = len(all_signals) if all_signals else len(opportunities)
            
            self.logger.info(f"📈 Generated {signals_count} signals ({len(opportunities)} top opportunities)")
            
            # Save results to database
            self.logger.info("💾 Saving results to MySQL database...")
            save_result = self.enhanced_db_manager.save_all_results(results)
            if save_result.get('error'):
                self.logger.error(f"Failed to save results: {save_result['error']}")
            else:
                self.logger.debug(f"✅ Results saved - Scan ID: {save_result.get('scan_id', 'Unknown')}")
            
            # Display ONLY top opportunities (clean console output)
            print("\n" + "=" * 80)
            print("📊 TOP TRADING OPPORTUNITIES ONLY")
            print("=" * 80)
            self.trading_system.print_comprehensive_results_with_mtf(results)
            print("=" * 80 + "\n")
            
            # Use opportunities for execution, limited by max_execution_per_trade
            if not opportunities:
                self.logger.warning("⚠️ No top opportunities available for execution")
                return signals_count, 0
            
            # FIXED: Ensure all opportunities have take_profit key for compatibility
            for opp in opportunities:
                # Ensure take_profit key exists (use take_profit_1 as primary)
                if 'take_profit' not in opp:
                    opp['take_profit'] = opp.get('take_profit_1', opp.get('take_profit_2', 0))
                
                # Ensure all required keys exist with fallbacks
                if 'take_profit_1' not in opp:
                    opp['take_profit_1'] = opp.get('take_profit', 0)
                if 'take_profit_2' not in opp:
                    opp['take_profit_2'] = opp.get('take_profit_1', 0) * 1.2  # 20% higher as TP2
            
            # FILTER OUT SYMBOLS WITH EXISTING POSITIONS/ORDERS
            filtered_opportunities = self.position_manager.filter_top_opportunities_by_existing_symbols(opportunities)
            
            if not filtered_opportunities:
                self.logger.warning("⚠️ No signals available for execution after filtering existing symbols")
                print("⚠️  ALL SIGNALS FILTERED OUT:")
                print("   Either symbols already have positions/orders")
                print("   or no valid signals generated")
                return signals_count, 0
            
            # Check position availability
            can_trade, available_slots = self.position_manager.can_open_new_positions(
                self.config.max_execution_per_trade
            )
            
            if not can_trade:
                self.logger.warning(f"⚠️ Cannot open new positions - at max capacity ({self.config.max_concurrent_positions})")
                return signals_count, 0
            
            # Limit execution to max_execution_per_trade
            execution_count = min(
                len(filtered_opportunities),
                self.config.max_execution_per_trade,  # This is the key limit
                available_slots
            )
            
            # Only execute the top N opportunities
            selected_opportunities = filtered_opportunities[:execution_count]
            
            # Execute trades for selected opportunities
            executed_count = 0
            
            print(f"\n🎯 EXECUTING TOP {execution_count} TRADES:")
            print(f"   Max Per Scan Setting: {self.config.max_execution_per_trade}")
            print(f"   Available After Filtering: {len(filtered_opportunities)}")
            print(f"   Selected for Execution: {execution_count}")
            print("=" * 60)
            
            for i, opportunity in enumerate(selected_opportunities):
                try:
                    # FIXED: Get take profit values with proper fallbacks
                    take_profit_1 = opportunity.get('take_profit_1', opportunity.get('take_profit', 0))
                    take_profit_2 = opportunity.get('take_profit_2', take_profit_1 * 1.2 if take_profit_1 > 0 else 0)
                    primary_take_profit = take_profit_1  # Use TP1 as primary target
                    
                    print(f"\n📈 TRADE {i+1}/{execution_count}: {opportunity['symbol']} {opportunity['side'].upper()}")
                    print(f"   Confidence: {opportunity['confidence']:.1f}%")
                    print(f"   MTF Status: {opportunity.get('mtf_status', 'N/A')}")
                    print(f"   Entry: ${opportunity['entry_price']:.6f}")
                    print(f"   Stop Loss: ${opportunity['stop_loss']:.6f}")
                    print(f"   Take Profit 1: ${take_profit_1:.6f}")
                    if take_profit_2 > 0 and take_profit_2 != take_profit_1:
                        print(f"   Take Profit 2: ${take_profit_2:.6f}")
                    
                    # Create a clean opportunity dict for execution
                    execution_opportunity = {
                        'symbol': opportunity['symbol'],
                        'side': opportunity['side'],
                        'entry_price': opportunity['entry_price'],
                        'stop_loss': opportunity['stop_loss'],
                        'take_profit': primary_take_profit,  # Use TP1 as primary
                        'take_profit_1': take_profit_1,
                        'take_profit_2': take_profit_2,
                        'confidence': opportunity['confidence'],
                        'mtf_status': opportunity.get('mtf_status', 'NONE'),
                        'order_type': opportunity.get('order_type', 'market'),
                        'risk_reward_ratio': opportunity.get('risk_reward_ratio', 1.0)
                    }
                    
                    # Execute the trade
                    success, message, position_data = await self.position_manager.execute_trade(execution_opportunity)
                    
                    if success:
                        executed_count += 1
                        print(f"   ✅ TRADE SUCCESSFUL!")
                        print(f"   Position ID: {position_data['position_id']}")
                        print(f"   Position Size: {position_data['position_size']:.4f} units")
                        print(f"   Risk Amount: {position_data['risk_amount']:.2f} USDT")
                        print(f"   Leverage: {position_data['leverage']}x")
                        
                        self.logger.info(
                            f"✅ Trade executed: {opportunity['symbol']} "
                            f"({opportunity['confidence']:.1f}% confidence, "
                            f"MTF: {opportunity.get('mtf_status', 'N/A')})"
                        )
                        
                        # Send Telegram notification
                        await self.send_trade_notification(execution_opportunity, position_data, success=True)
                    else:
                        print(f"   ❌ TRADE FAILED: {message}")
                        self.logger.error(f"❌ Trade failed: {opportunity['symbol']} - {message}")
                        
                        # Send failure notification
                        await self.send_trade_notification(execution_opportunity, {}, success=False, error_message=message)
                
                except Exception as e:
                    print(f"   ❌ TRADE ERROR: {e}")
                    self.logger.error(f"Error executing trade for {opportunity.get('symbol', 'unknown')}: {e}")
                    # Log the opportunity structure for debugging
                    self.logger.debug(f"Opportunity keys: {list(opportunity.keys())}")
            
            # Final execution summary
            print(f"\n🏁 EXECUTION SUMMARY:")
            print(f"   Total Signals Generated: {signals_count}")
            print(f"   Top Opportunities: {len(opportunities)}")
            print(f"   After Symbol Filtering: {len(filtered_opportunities)}")
            print(f"   Max Execution Per Scan: {self.config.max_execution_per_trade}")
            print(f"   Trades Attempted: {execution_count}")
            print(f"   Trades Executed: {executed_count}")
            print(f"   Success Rate: {(executed_count/execution_count*100) if execution_count > 0 else 0:.1f}%")
            
            return signals_count, executed_count
            
        except Exception as e:
            self.logger.error(f"Error in scan and execute: {e}")
            return 0, 0
        
    def main_trading_loop(self):
        """Main auto-trading loop - FIXED"""
        try:
            self.is_running = True
            session_id = self.start_trading_session()
            
            # Start profit monitoring
            self.profit_monitor.start_monitoring()
            
            self.logger.info("🤖 Auto-trading loop started")
            
            while self.is_running:
                try:
                    # Wait for next scheduled scan
                    next_scan_time = self.schedule_manager.wait_for_next_scan()
                    
                    if not self.is_running:
                        break
                    
                    self.logger.info(f"⏰ Scan time reached: {next_scan_time}")
                    
                    # FIXED: Run scan and execute trades with proper async handling
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        signals_count, executed_count = loop.run_until_complete(
                            self.run_scan_and_execute()
                        )
                        
                        # FIXED: Add proper logging to show the actual results
                        if signals_count > 0:
                            self.logger.info(f"✅ Scan successful: {signals_count} signals generated, {executed_count} trades executed")
                        else:
                            self.logger.warning("❌ No results returned from analysis")
                        
                    finally:
                        loop.close()
                    
                    # FIXED: Properly log the scan completion
                    self.logger.info(f"📊 Scan complete - Total Signals: {signals_count}, Top Opportunities Executed: {executed_count}")
                    
                    # Update session statistics
                    self.update_session_stats(signals_count, executed_count)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Received interrupt signal")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main trading loop: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
            
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
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
            
            self.logger.info("🛑 Auto-trading stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")
    
    async def send_trade_notification(self, opportunity: Dict, position_data: Dict, success: bool, error_message: str = None):
        """Send Telegram notification for trade execution"""
        try:
            symbol = opportunity['symbol']
            side = opportunity['side'].upper()
            confidence = opportunity.get('confidence', 0)
            mtf_status = opportunity.get('mtf_status', 'N/A')
            quality_tier = opportunity.get('quality_tier', 'Unknown')
            selection_rank = opportunity.get('selection_rank', 'Unknown')
            
            if success:
                # Success notification for TOP OPPORTUNITY
                risk_usdt = position_data.get('risk_amount', 0)
                leverage = position_data.get('leverage', 0)
                position_size = position_data.get('position_size', 0)
                entry_price = position_data.get('entry_price', 0)
                
                message = f"🏆 **TOP OPPORTUNITY EXECUTED**\n\n"
                message += f"📊 **{symbol}** {side}\n"
                message += f"💰 Entry Price: ${entry_price:.6f}\n"
                message += f"📈 Position Size: {position_size:.4f} units\n"
                message += f"⚡ Leverage: {leverage}x\n"
                message += f"💵 Risk: {risk_usdt:.2f} USDT ({self.config.risk_amount}%)\n\n"
                message += f"🎯 **TOP OPPORTUNITY QUALITY:**\n"
                message += f"   Confidence: {confidence:.1f}%\n"
                message += f"   Quality Tier: {quality_tier}\n"
                message += f"   Selection Rank: #{selection_rank}\n"
                message += f"   MTF Status: {mtf_status}\n\n"
                message += f"🎯 Profit Target: {self.config.auto_close_profit_at}%\n"
                message += f"🕒 {datetime.now().strftime('%H:%M:%S')}"
                
                # Add inline keyboard for position management
                keyboard = [
                    [{"text": "📊 Check Position", "callback_data": f"check_pos_{symbol}"}],
                    [{"text": "🔴 Close Position", "callback_data": f"close_pos_{symbol}"}]
                ]
                
            else:
                # Failure notification for TOP OPPORTUNITY
                message = f"❌ **TOP OPPORTUNITY FAILED**\n\n"
                message += f"📊 **{symbol}** {side}\n"
                message += f"🎯 Confidence: {confidence:.1f}%\n"
                message += f"📈 Quality Tier: {quality_tier}\n"
                message += f"🔢 Selection Rank: #{selection_rank}\n"
                message += f"📈 MTF Status: {mtf_status}\n\n"
                message += f"💥 **Error:** {error_message}\n\n"
                message += f"🕒 {datetime.now().strftime('%H:%M:%S')}"
                
                keyboard = None
            
            await send_trading_notification(self.config, message, keyboard)
            
        except Exception as e:
            self.logger.error(f"Failed to send trade notification: {e}")
    
    async def send_scan_notification(self, total_signals: int, executed_count: int):
        """Send scan completion notification"""
        try:
            next_scan = self.schedule_manager.calculate_next_scan_time()
            
            message = f"📊 **TOP OPPORTUNITIES SCAN COMPLETE**\n\n"
            message += f"🔍 Total Signals: {total_signals}\n"
            message += f"🏆 Top Opportunities Executed: {executed_count}\n"
            message += f"📈 Focus: Quality over Quantity\n\n"
            message += f"⏰ Next Scan: {next_scan.strftime('%H:%M')}\n"
            message += f"🕒 {datetime.now().strftime('%H:%M:%S')}"
            
            await send_trading_notification(self.config, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send scan notification: {e}")
    
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
            
            # Send scan notification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.send_scan_notification(total_signals, executed_count))
            finally:
                loop.close()
            
        except Exception as e:
            self.logger.error(f"Error updating session stats: {e}")


# Main execution function for standalone usage
def main():
    """Main function for running the auto-trader with TOP OPPORTUNITIES focus"""
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
        
        logger.info("🏆 Starting AUTO-TRADER with TOP OPPORTUNITIES focus")
        logger.info("🎯 Only the highest quality signals will be executed")
        
        # Start auto-trader
        auto_trader = AutoTrader(config)
        auto_trader.main_trading_loop()
        
    except KeyboardInterrupt:
        logger.info("Auto-trader stopped by user")
    except Exception as e:
        logger.error(f"Auto-trader failed: {e}")
        raise


if __name__ == "__main__":
    main()