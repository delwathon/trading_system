"""
Auto-Trading System for Enhanced Bybit Trading System.
Handles scheduled scanning, position management, and automated trading with leverage support.
UPDATED: Enhanced with MILESTONE-BASED TRAILING STOPS that consider leverage dynamically.
FIXED: Bybit API integration issues and integrated trailing stops with partial profit milestones.
Version: 2.1
"""

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
from database.models import DatabaseManager, TradingPosition


# ========================================
# PHASE 1: DATA STRUCTURES AND MODELS
# ========================================

@dataclass
class PositionData:
    """Data class for position tracking with milestone-based trailing stop support"""
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
    # Milestone tracking fields
    milestone_reached: str = 'none'  # 'none', 'break_even', 'profit_lock_1', 'profit_lock_2', 'profit_lock_3'
    original_stop_loss: float = 0.0  # Store original SL for reference
    last_milestone_check: float = 0.0  # Last price at which we checked milestones
    stop_order_id: str = ''  # Track stop order ID for updates
    # Position sizing
    original_size: float = 0.0  # Store original size for partial profit calculations
    # Legacy fields (kept for backward compatibility)
    trailing_stop_active: bool = False
    highest_price: float = 0.0
    lowest_price: float = 0.0


@dataclass
class TrailingStopMilestone:
    """Data class for milestone configuration"""
    name: str
    trigger_price_move_pct: float  # Price movement percentage to trigger
    stop_price_move_pct: float  # Where to place the stop (as price movement %)
    leveraged_profit_pct: float  # Leveraged profit at this milestone
    stop_leveraged_profit_pct: float  # Leveraged profit locked in


# ========================================
# PHASE 2: LEVERAGE MANAGEMENT
# ========================================

class LeverageManager:
    """Handle leverage validation and conversion"""
    
    ACCEPTABLE_LEVERAGE = ['10', '12.5', '25', '50', '100', 'max']
    
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

# ========================================
# PHASE 3: POSITION SIZING
# ========================================

class PositionSizer:
    """Calculate position sizes with leverage - ENHANCED with adaptive sizing"""
    
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
    
    def calculate_adaptive_position_size(self, risk_amount_pct: float, leverage: float, 
                                       entry_price: float, signal: Dict, df=None) -> Tuple[float, float]:
        """
        ENHANCED: Adaptive position sizing based on market conditions and signal quality
        
        Returns:
            Tuple of (position_size, adjusted_risk_pct)
        """
        try:
            base_risk_pct = risk_amount_pct
            
            # 1. MTF Quality Adjustment
            mtf_validated = signal.get('mtf_validated', False)
            analysis_details = signal.get('analysis_details', {})
            signal_strength = analysis_details.get('signal_strength', 'moderate')
            confidence = signal.get('confidence', 50)
            
            if mtf_validated and signal_strength == 'strong' and confidence > 75:
                risk_multiplier = 1.3  # Increase risk for premium signals
            elif mtf_validated and confidence > 65:
                risk_multiplier = 1.1  # Slightly increase for good MTF signals
            elif mtf_validated:
                risk_multiplier = 1.0  # Normal risk for MTF validated
            elif signal_strength == 'strong' and confidence > 70:
                risk_multiplier = 0.9  # Slightly reduce for non-MTF strong signals
            else:
                risk_multiplier = 0.7  # Reduce risk for weaker signals
            
            # 2. Volatility Adjustment (if dataframe available)
            volatility_multiplier = 1.0
            if df is not None and len(df) > 20:
                try:
                    # Calculate ATR-based volatility
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift())
                    low_close = np.abs(df['low'] - df['close'].shift())
                    import pandas as pd
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = true_range.rolling(window=14).mean().iloc[-1]
                    volatility_pct = (atr / entry_price) * 100
                    
                    if volatility_pct > 8:  # High volatility
                        volatility_multiplier = 0.6  # Reduce position size significantly
                    elif volatility_pct > 5:  # Medium volatility
                        volatility_multiplier = 0.8  # Reduce moderately
                    else:  # Low volatility
                        volatility_multiplier = 1.0
                        
                    self.logger.debug(f"📊 Volatility adjustment: {volatility_pct:.1f}% → {volatility_multiplier:.1f}x")
                except Exception as e:
                    self.logger.debug(f"Could not calculate volatility: {e}")
            
            # 3. Risk/Reward Adjustment
            rr_ratio = signal.get('risk_reward_ratio', 2.0)
            if rr_ratio >= 3.5:
                rr_multiplier = 1.2  # Increase for excellent R/R
            elif rr_ratio >= 2.8:
                rr_multiplier = 1.1
            elif rr_ratio >= 2.2:
                rr_multiplier = 1.0
            else:
                rr_multiplier = 0.8  # Reduce for poor R/R
            
            # 4. Market Conditions Adjustment
            price_change_24h = abs(signal.get('price_change_24h', 0))
            if price_change_24h > 15:  # Extreme movement
                market_multiplier = 0.5  # Very conservative
            elif price_change_24h > 8:  # High movement
                market_multiplier = 0.7
            else:
                market_multiplier = 1.0
            
            # 5. Volume Quality Adjustment
            volume_24h = signal.get('volume_24h', 0)
            if volume_24h < 500_000:  # Low liquidity
                volume_multiplier = 0.6
            elif volume_24h < 1_000_000:  # Medium liquidity
                volume_multiplier = 0.8
            else:  # Good liquidity
                volume_multiplier = 1.0
            
            # Calculate final risk percentage
            adjusted_risk_pct = (base_risk_pct * risk_multiplier * volatility_multiplier * 
                                rr_multiplier * market_multiplier * volume_multiplier)
            
            # Apply reasonable limits
            adjusted_risk_pct = max(1.0, min(10.0, adjusted_risk_pct))
            
            # Calculate position size with adjusted risk
            account_balance = self.get_available_balance()
            risk_amount_usdt = account_balance * (adjusted_risk_pct / 100)
            position_size = (risk_amount_usdt * leverage) / entry_price
            
            self.logger.info(f"📊 Adaptive Position Sizing:")
            self.logger.info(f"   Base Risk: {base_risk_pct}% → Adjusted: {adjusted_risk_pct:.1f}%")
            self.logger.info(f"   Quality: {risk_multiplier:.1f}x | Volatility: {volatility_multiplier:.1f}x")
            self.logger.info(f"   R/R: {rr_multiplier:.1f}x | Market: {market_multiplier:.1f}x | Volume: {volume_multiplier:.1f}x")
            self.logger.info(f"   Final Risk: {risk_amount_usdt:.2f} USDT ({adjusted_risk_pct:.1f}%)")
            
            return position_size, adjusted_risk_pct
            
        except Exception as e:
            self.logger.error(f"Adaptive position sizing error: {e}")
            # Fallback to regular calculation
            position_size = self.calculate_position_size(risk_amount_pct, leverage, entry_price)
            return position_size, risk_amount_pct
    
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


# ========================================
# PHASE 5: ENHANCED MILESTONE-BASED TRAILING STOP MONITOR
# ========================================

class LeveragedProfitMonitor:
    """
    Monitor leveraged profits/losses and auto-close positions
    ENHANCED: Milestone-based trailing stops with leverage-aware calculations
    FIXED: Bybit API integration and integrated with partial profit milestones
    """
    
    def __init__(self, exchange, config: EnhancedSystemConfig):
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.position_tracker = {}  # Track position states and milestones
    
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
    
    def calculate_price_from_leveraged_profit(self, position: PositionData, target_leveraged_profit_pct: float) -> float:
        """Calculate price level that corresponds to target leveraged profit percentage"""
        try:
            if position.entry_price <= 0 or position.leverage <= 0:
                return position.entry_price
            
            # Convert leveraged profit to price change percentage
            price_change_pct = target_leveraged_profit_pct / position.leverage
            
            # Consider position side
            if position.side.lower() == 'buy':
                target_price = position.entry_price * (1 + price_change_pct / 100)
            else:  # sell
                target_price = position.entry_price * (1 - price_change_pct / 100)
            
            return target_price
        except Exception as e:
            self.logger.error(f"Error calculating price from leveraged profit: {e}")
            return position.entry_price
    
    def should_auto_close(self, position: PositionData, current_price: float) -> tuple[bool, str]:
        """Check if position should be auto-closed based on profit/loss targets"""
        try:
            leveraged_profit_pct = self.calculate_leveraged_profit_pct(position, current_price)
            
            # Check profit target
            if leveraged_profit_pct >= self.config.auto_close_profit_at:
                return True, 'profit_target'
            
            # Check loss limit
            if leveraged_profit_pct <= -abs(self.config.auto_close_loss_at):
                return True, 'loss_limit'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error checking auto-close for {position.symbol}: {e}")
            return False, None
    
    def check_partial_profit_taking(self, position: PositionData, current_price: float) -> Optional[Dict]:
        """
        ENHANCED: Check for partial profit taking opportunities based on LEVERAGED profit
        Fixed to ensure proper stop loss tracking initialization
        """
        try:
            symbol = position.symbol
            
            # Initialize partial profit tracking if not exists
            if symbol not in self.position_tracker:
                # Get the actual stop loss and take profit from the position or exchange
                actual_stop_loss = position.stop_loss
                actual_take_profit = position.take_profit
                
                if actual_stop_loss == 0 or actual_take_profit == 0:
                    # Try to get from exchange
                    try:
                        positions = self.exchange.fetch_positions([symbol])
                        for pos in positions:
                            if pos['symbol'] == symbol:
                                actual_stop_loss = pos.get('stopLoss', 0) or actual_stop_loss
                                actual_take_profit = pos.get('takeProfit', 0) or actual_take_profit
                                break
                    except Exception as e:
                        self.logger.debug(f"Could not fetch stop loss/take profit from exchange: {e}")
                
                self.position_tracker[symbol] = {
                    'milestone_reached': 'none',
                    'original_stop': actual_stop_loss,
                    'original_take_profit': actual_take_profit,  # Store original TP
                    'current_stop': actual_stop_loss,
                    'current_take_profit': actual_take_profit,  # Track current TP
                    'last_check_price': current_price,
                    'partial_100_taken': False,
                    'partial_200_taken': False,
                    'partial_300_taken': False,
                    'original_size': position.size,
                    'manual_stop_needed': False,  # Track if manual intervention is needed
                    'manual_stop_price': 0  # Price for manual stop if needed
                }
                
                self.logger.debug(f"Initialized tracker for {symbol}")
                self.logger.debug(f"  Stop Loss: ${actual_stop_loss:.6f}")
                self.logger.debug(f"  Take Profit: ${actual_take_profit:.6f}")
            
            tracker = self.position_tracker[symbol]
            
            # Calculate LEVERAGED profit percentage
            leveraged_profit_pct = self.calculate_leveraged_profit_pct(position, current_price)
            
            # Skip if loss or if all partials taken
            if leveraged_profit_pct <= 0:
                return None
            
            # Check if manual stop monitoring is needed
            if tracker.get('manual_stop_needed', False):
                manual_stop = tracker.get('manual_stop_price', 0)
                if manual_stop > 0:
                    # Check if price hit manual stop
                    if (position.side.lower() == 'buy' and current_price <= manual_stop) or \
                    (position.side.lower() == 'sell' and current_price >= manual_stop):
                        self.logger.warning(f"🚨 MANUAL STOP HIT for {symbol} at ${current_price:.6f} (stop: ${manual_stop:.6f})")
                        # Return a special action to close the position
                        return {
                            'action': 'close_full',
                            'reason': 'manual_stop_hit',
                            'stop_price': manual_stop,
                            'current_price': current_price
                        }
            
            # Check partial profit levels
            if leveraged_profit_pct >= 300.0 and not tracker.get('partial_300_taken', False):
                self.logger.info(f"🎯 300% leveraged profit reached for {symbol}!")
                tracker['partial_300_taken'] = True
                return {
                    'action': 'close_partial',
                    'percentage': 50,
                    'level': 'partial_300',
                    'leveraged_profit_pct': leveraged_profit_pct,
                    'milestone': '300% leveraged profit',
                    'preserve_take_profit': tracker.get('current_take_profit', 0)  # Pass TP to preserve
                }
            
            elif leveraged_profit_pct >= 200.0 and not tracker.get('partial_200_taken', False):
                self.logger.info(f"🎯 200% leveraged profit reached for {symbol}!")
                tracker['partial_200_taken'] = True
                return {
                    'action': 'close_partial',
                    'percentage': 50,
                    'level': 'partial_200',
                    'leveraged_profit_pct': leveraged_profit_pct,
                    'milestone': '200% leveraged profit',
                    'preserve_take_profit': tracker.get('current_take_profit', 0)
                }
            
            elif leveraged_profit_pct >= 100.0 and not tracker.get('partial_100_taken', False):
                self.logger.info(f"🎯 100% leveraged profit reached for {symbol}!")
                tracker['partial_100_taken'] = True
                return {
                    'action': 'close_partial',
                    'percentage': 50,
                    'level': 'partial_100',
                    'leveraged_profit_pct': leveraged_profit_pct,
                    'milestone': '100% leveraged profit',
                    'preserve_take_profit': tracker.get('current_take_profit', 0)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Partial profit check error for {position.symbol}: {str(e)}")
            return None
    
    def execute_partial_close(self, position: PositionData, partial_info: Dict) -> bool:
        """
        ENHANCED: Execute partial position close and update trailing stops based on milestones
        """
        try:
            symbol = position.symbol
            percentage = partial_info['percentage']
            level = partial_info['level']
            leveraged_profit_pct = partial_info.get('leveraged_profit_pct', 0)
            milestone = partial_info.get('milestone', '')
            
            # Get current position size from exchange (CRITICAL for accuracy)
            positions = self.exchange.fetch_positions([symbol])
            current_position = None
            
            for pos in positions:
                if pos['symbol'] == symbol and pos['contracts'] > 0:
                    current_position = pos
                    break
            
            if not current_position:
                self.logger.warning(f"No active position found for partial close: {symbol}")
                return False
            
            # Calculate partial close size (50% of CURRENT remaining position)
            current_size = current_position['contracts']
            close_size = current_size * (percentage / 100)
            
            # Determine close side
            current_side = current_position['side']
            if current_side.lower() == 'long':
                close_side = 'sell'
            elif current_side.lower() == 'short':
                close_side = 'buy'
            else:
                self.logger.error(f"Unknown position side for partial close: {current_side}")
                return False
            
            # Execute partial close
            try:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=close_size,
                    params={'reduceOnly': True}
                )
                
                # Log successful partial profit
                self.logger.info(f"💰 PARTIAL PROFIT TAKEN: {symbol}")
                self.logger.info(f"   Milestone: {milestone}")
                self.logger.info(f"   Leveraged Profit: {leveraged_profit_pct:.1f}%")
                self.logger.info(f"   Closed: {percentage}% of remaining ({close_size:.4f} units)")
                self.logger.info(f"   Remaining Position: {current_size - close_size:.4f} units")
                
                # ENHANCED: Update trailing stop based on partial profit milestone
                self.update_trailing_stop_for_partial_milestone(position, level, leveraged_profit_pct)
                
                # Send notification about partial profit
                asyncio.run(self._send_partial_profit_notification(
                    symbol, milestone, leveraged_profit_pct, 
                    close_size, current_size - close_size
                ))
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to execute partial close order: {e}")
                # Reset the tracking flag if execution failed
                if symbol in self.position_tracker:
                    tracker = self.position_tracker[symbol]
                    if level == 'partial_100':
                        tracker['partial_100_taken'] = False
                    elif level == 'partial_200':
                        tracker['partial_200_taken'] = False
                    elif level == 'partial_300':
                        tracker['partial_300_taken'] = False
                return False
            
        except Exception as e:
            self.logger.error(f"Partial close execution error for {position.symbol}: {e}")
            return False
    
    def update_trailing_stop_for_partial_milestone(self, position: PositionData, partial_level: str, leveraged_profit_pct: float):
        """
        ENHANCED: Update trailing stop when partial profit is taken
        - partial_100 → move SL to break even (0% leveraged profit)
        - partial_200 → move SL to 100% leveraged profit level  
        - partial_300 → move SL to 200% leveraged profit level
        """
        try:
            symbol = position.symbol
            
            # Determine target leveraged profit for stop loss
            if partial_level == 'partial_100':
                target_leveraged_profit = 10.0  # Break even
                stop_description = "Break Even"
            elif partial_level == 'partial_200':
                target_leveraged_profit = 100.0  # 100% leveraged profit
                stop_description = "100% leveraged profit"
            elif partial_level == 'partial_300':
                target_leveraged_profit = 200.0  # 200% leveraged profit
                stop_description = "200% leveraged profit"
            else:
                return  # Unknown partial level
            
            # Calculate new stop loss price
            new_stop_price = self.calculate_price_from_leveraged_profit(position, target_leveraged_profit)
            
            self.logger.info(f"🎯 UPDATING TRAILING STOP FOR PARTIAL MILESTONE: {symbol}")
            self.logger.info(f"   Partial Level: {partial_level}")
            self.logger.info(f"   Target Level: {stop_description}")
            self.logger.info(f"   New Stop Price: ${new_stop_price:.6f}")
            
            # Update position tracking
            if symbol in self.position_tracker:
                tracker = self.position_tracker[symbol]
                old_stop = tracker.get('current_stop', position.stop_loss)
                tracker['current_stop'] = new_stop_price
                tracker['milestone_reached'] = f"partial_{partial_level}"
                
                self.logger.info(f"   Old Stop: ${old_stop:.6f} → New Stop: ${new_stop_price:.6f}")
            
            # Update stop order on exchange
            if self._update_stop_order_on_exchange(position, new_stop_price):
                self.logger.info(f"✅ Trailing stop updated for partial milestone: {symbol}")
                position.stop_loss = new_stop_price
                
                # Send notification
                asyncio.run(self._send_trailing_stop_update_notification(
                    symbol, stop_description, new_stop_price, leveraged_profit_pct
                ))
            else:
                self.logger.error(f"❌ Failed to update trailing stop for partial milestone: {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error updating trailing stop for partial milestone {position.symbol}: {e}")
    
    def _update_stop_order_on_exchange(self, position: PositionData, new_stop_price: float) -> bool:
        """
        FIXED: Update the stop loss order on the exchange while preserving take profit
        Uses proper Bybit V5 API parameters to maintain TP/SL binding
        """
        try:
            symbol = position.symbol
            side = position.side.lower()
            
            self.logger.debug(f"🔧 Updating stop order for {symbol}: ${new_stop_price:.6f}")
            
            # Get current position from exchange with TP/SL values
            positions = self.exchange.fetch_positions([symbol])
            current_position = None
            current_take_profit = 0
            
            for pos in positions:
                if pos['symbol'] == symbol and pos['contracts'] > 0:
                    current_position = pos
                    # CRITICAL: Preserve current take profit value
                    current_take_profit = pos.get('takeProfit', 0) or position.take_profit
                    break
            
            if not current_position:
                self.logger.warning(f"No active position found for {symbol} when updating stop")
                return False
            
            position_size = current_position['contracts']
            current_side = current_position['side'].lower()
            
            # Update position tracker with current TP if we have it
            if symbol in self.position_tracker and current_take_profit > 0:
                self.position_tracker[symbol]['current_take_profit'] = current_take_profit
            
            # METHOD 1: Use set_trading_stop with BOTH SL and TP to preserve binding
            try:
                # Convert symbol format for Bybit API
                bybit_symbol = symbol.replace('/', '').replace(':USDT', '')
                
                # Build params - CRITICAL: Include both SL and TP to maintain binding
                params = {
                    'category': 'linear',
                    'symbol': bybit_symbol,
                    'stopLoss': str(new_stop_price),
                    'tpslMode': 'Full',  # Use Full mode for entire position
                }
                
                # IMPORTANT: Include take profit if it exists to preserve it
                if current_take_profit > 0:
                    params['takeProfit'] = str(current_take_profit)
                    self.logger.debug(f"Preserving take profit at ${current_take_profit:.6f}")
                
                # Use direct API call for better control
                result = self.exchange.privatePostV5PositionTradingStop(params)
                
                if result.get('retCode') == 0:
                    self.logger.info(f"✅ Stop loss updated to ${new_stop_price:.6f} (TP preserved at ${current_take_profit:.6f})")
                    return True
                else:
                    self.logger.debug(f"Method 1 failed with retCode: {result.get('retCode')}, retMsg: {result.get('retMsg')}")
                    
            except Exception as e1:
                self.logger.debug(f"Method 1 (set_trading_stop with TP preservation) failed: {str(e1)}")
            
            # METHOD 2: Use CCXT's built-in method with proper params
            try:
                # Build request with both SL and TP
                request_params = {
                    'category': 'linear',
                    'symbol': symbol.replace('/', '').replace(':USDT', ''),
                    'stopLoss': str(new_stop_price),
                    'tpslMode': 'Full'
                }
                
                # Include TP to preserve it
                if current_take_profit > 0:
                    request_params['takeProfit'] = str(current_take_profit)
                
                # Try alternative API endpoint
                result = self.exchange.private_post_v5_position_trading_stop(request_params)
                
                if result and result.get('retCode') == 0:
                    self.logger.info(f"✅ Trading stop updated via CCXT at ${new_stop_price:.6f} (TP: ${current_take_profit:.6f})")
                    return True
                    
            except Exception as e2:
                self.logger.debug(f"Method 2 (CCXT trading stop) failed: {str(e2)}")
            
            # METHOD 3: Try with different API call structure
            try:
                # Another attempt with different parameter structure
                result = self.exchange.private_post_v5_position_set_trading_stop({
                    'category': 'linear',
                    'symbol': symbol.replace('/', '').replace(':USDT', ''),
                    'stopLoss': str(new_stop_price),
                    'takeProfit': str(current_take_profit) if current_take_profit > 0 else None
                })
                
                self.logger.info(f"✅ Trading stop updated (Method 3) at ${new_stop_price:.6f}")
                return True
                
            except Exception as e3:
                self.logger.debug(f"Method 3 (alternative API) failed: {str(e3)}")
            
            # METHOD 4: Use conditional stop order as separate order (fallback)
            try:
                # First cancel existing stop orders but NOT take profit orders
                open_orders = self.exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    # Only cancel stop orders, not take profit orders
                    if order.get('type') in ['stop', 'stop_market'] and order.get('side') != current_side:
                        # Check if it's a stop loss (opposite side of position)
                        order_stop = order.get('stopPrice', 0) or order.get('triggerPrice', 0)
                        current_mark = current_position.get('markPrice', 0)
                        
                        is_stop_loss = False
                        if current_side == 'long' and order['side'] == 'sell' and order_stop < current_mark:
                            is_stop_loss = True
                        elif current_side == 'short' and order['side'] == 'buy' and order_stop > current_mark:
                            is_stop_loss = True
                        
                        if is_stop_loss:
                            try:
                                self.exchange.cancel_order(order['id'], symbol)
                                self.logger.debug(f"Cancelled existing stop order: {order['id']}")
                            except Exception as cancel_err:
                                self.logger.debug(f"Could not cancel order {order['id']}: {cancel_err}")
                
                # Determine stop order side
                if current_side == 'long':
                    stop_side = 'sell'
                else:  # short
                    stop_side = 'buy'
                
                # Create new conditional stop order
                stop_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side=stop_side,
                    amount=position_size,
                    stopPrice=new_stop_price,
                    params={
                        'triggerPrice': new_stop_price,
                        'reduceOnly': True,
                        'triggerBy': 'LastPrice',
                        'closeOnTrigger': False  # Don't close TP orders
                    }
                )
                
                position.stop_order_id = stop_order['id']
                self.logger.info(f"✅ Conditional stop order placed: {stop_order['id']} at ${new_stop_price:.6f}")
                self.logger.info(f"   Take profit preserved at ${current_take_profit:.6f}")
                return True
                
            except Exception as e4:
                self.logger.debug(f"Method 4 (conditional stop order) failed: {str(e4)}")
            
            # If all methods fail, track internally and alert user
            self.logger.warning(f"⚠️ All API methods failed. Tracking stop internally at ${new_stop_price:.6f}")
            self.logger.warning(f"⚠️ MANUAL INTERVENTION NEEDED: Set stop loss for {symbol} at ${new_stop_price:.6f}")
            self.logger.warning(f"⚠️ Ensure take profit remains at ${current_take_profit:.6f}")
            
            # Send urgent notification
            asyncio.run(self._send_manual_intervention_notification(symbol, new_stop_price, current_side))
            
            # Update internal tracking
            position.stop_loss = new_stop_price
            
            # Store in position tracker for manual monitoring
            if symbol in self.position_tracker:
                self.position_tracker[symbol]['manual_stop_needed'] = True
                self.position_tracker[symbol]['manual_stop_price'] = new_stop_price
                # Store current TP for recovery
                self.position_tracker[symbol]['preserved_take_profit'] = current_take_profit
            
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in stop order update: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def _send_manual_intervention_notification(self, symbol: str, stop_price: float, side: str):
        """Send urgent notification when manual intervention is needed"""
        try:
            from telegram_bot_and_notification.bootstrap_manager import send_trading_notification
            
            message = (
                f"🚨 *URGENT: MANUAL INTERVENTION NEEDED\\!*\n"
                f"\n"
                f"Symbol: {escape_markdown(symbol)}\n"
                f"Position: {escape_markdown(side.upper())}\n"
                f"Required Stop Loss: ${escape_markdown(f'{stop_price:.6f}')}\n"
                f"\n"
                f"❌ All automated stop update methods failed\\!\n"
                f"🔧 Please manually set stop loss on exchange\\.\n"
                f"\n"
                f"⚠️ Position is NOT protected until stop is set\\!"
            )
            
            await send_trading_notification(self.config, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send manual intervention notification: {e}")

    # Additional helper method to check if stop order was actually placed
    def _verify_stop_order_placement(self, symbol: str, expected_stop_price: float) -> bool:
        """Verify that the stop order was actually placed on the exchange"""
        try:
            # Check open orders for stop orders
            open_orders = self.exchange.fetch_open_orders(symbol)
            stop_orders = [o for o in open_orders if o['type'] in ['stop', 'stop_market', 'conditional']]
            
            for order in stop_orders:
                # Check if any stop order is close to our expected price (within 0.1%)
                if order.get('triggerPrice') or order.get('stopPrice'):
                    order_stop_price = order.get('triggerPrice') or order.get('stopPrice')
                    price_diff_pct = abs(order_stop_price - expected_stop_price) / expected_stop_price
                    
                    if price_diff_pct < 0.001:  # Within 0.1%
                        self.logger.debug(f"✅ Verified stop order exists at ${order_stop_price:.6f}")
                        return True
            
            self.logger.warning(f"⚠️ No matching stop order found for ${expected_stop_price:.6f}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying stop order placement: {e}")
            return False
      
    async def _send_partial_profit_notification(self, symbol: str, milestone: str, 
                                               leveraged_profit: float, closed_size: float, 
                                               remaining_size: float):
        """Send notification when partial profit is taken"""
        try:
            from telegram_bot_and_notification.bootstrap_manager import send_trading_notification
            
            message = (
                f"💰 *PARTIAL PROFIT TAKEN\\!*\n"
                f"\n"
                f"Symbol: {escape_markdown(symbol)}\n"
                f"Milestone: *{escape_markdown(milestone)}*\n"
                f"Leveraged Profit: {escape_markdown(f'{leveraged_profit:.1f}')}%\n"
                f"Closed Size: {escape_markdown(f'{closed_size:.4f}')} units\n"
                f"Remaining: {escape_markdown(f'{remaining_size:.4f}')} units\n"
                f"\n"
                f"50% of position secured\\! 💎🙌\n"
                f"Trailing stop updated automatically\\! 🎯"
            )
            
            await send_trading_notification(self.config, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send partial profit notification: {e}")
    
    async def _send_trailing_stop_update_notification(self, symbol: str, stop_description: str, 
                                                     new_stop_price: float, leveraged_profit: float):
        """Send notification when trailing stop is updated"""
        try:
            from telegram_bot_and_notification.bootstrap_manager import send_trading_notification
            
            message = (
                f"🎯 *TRAILING STOP UPDATED\\!*\n"
                f"\n"
                f"Symbol: {escape_markdown(symbol)}\n"
                f"New Stop Level: *{escape_markdown(stop_description)}*\n"
                f"Stop Price: ${escape_markdown(f'{new_stop_price:.6f}')}\n"
                f"Current Profit: {escape_markdown(f'{leveraged_profit:.1f}')}%\n"
                f"\n"
                f"Your profits are protected\\! 🛡️"
            )
            
            await send_trading_notification(self.config, message)
            
        except Exception as e:
            self.logger.error(f"Failed to send trailing stop update notification: {e}")
    
    def close_position(self, position: PositionData) -> bool:
        """Close a position on the exchange"""
        try:
            # Clean up position tracker (removes both milestone and partial profit tracking)
            if position.symbol in self.position_tracker:
                # Log final partial profit status before cleanup
                tracker = self.position_tracker[position.symbol]
                partials_taken = []
                if tracker.get('partial_100_taken', False):
                    partials_taken.append('100%')
                if tracker.get('partial_200_taken', False):
                    partials_taken.append('200%')
                if tracker.get('partial_300_taken', False):
                    partials_taken.append('300%')
                
                if partials_taken:
                    self.logger.debug(f"Position {position.symbol} had partials taken at: {', '.join(partials_taken)}")
                
                # Now clean up the tracker
                del self.position_tracker[position.symbol]
            
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
            
            # Use proper lowercase sides for CCXT
            if current_side.lower() == 'long':
                close_side = 'sell'
            elif current_side.lower() == 'short':
                close_side = 'buy'
            else:
                self.logger.error(f"Unknown position side: {current_side}")
                return False
            
            self.logger.debug(f"Closing {current_side} position of {position_size} {position.symbol} with {close_side} order")
            
            # Place market order to close position
            order = self.exchange.create_order(
                symbol=position.symbol,
                type='market',
                side=close_side,
                amount=position_size,
                params={'reduceOnly': True}  # Ensure this closes the position
            )
            
            self.logger.debug(f"✅ Closed position {position.symbol}")
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
                        stop_loss=pos.get('stopLoss', 0.0),
                        take_profit=pos.get('takeProfit', 0.0),
                        position_id=pos.get('id', ''),
                        unrealized_pnl=pos.get('unrealizedPnl', 0.0),
                        unrealized_pnl_pct=pos.get('percentage', 0.0),
                        original_size=pos['contracts']  # Store original size
                    )
                    active_positions.append(position_data)
            
            return active_positions
        except Exception as e:
            self.logger.error(f"Error getting current positions: {e}")
            return []
    
    def monitor_positions(self):
        """
        Main monitoring loop - ENHANCED to handle manual stop monitoring
        """
        try:
            while self.monitoring:
                positions = self.get_current_positions()
                
                for position in positions:
                    try:
                        # Get current price
                        ticker = self.exchange.fetch_ticker(position.symbol)
                        current_price = ticker['last']
                        
                        # Add debug logging for position monitoring
                        leveraged_profit = self.calculate_leveraged_profit_pct(position, current_price)
                        
                        self.logger.debug(f"🔍 Monitoring {position.symbol}:")
                        self.logger.debug(f"   Price: ${current_price:.6f} | Entry: ${position.entry_price:.6f}")
                        self.logger.debug(f"   Leverage: {position.leverage}x | Profit: {leveraged_profit:.2f}%")
                        
                        # PHASE 1: Check for partial profit taking (with integrated trailing stops)
                        partial_taken = False
                        partial_info = self.check_partial_profit_taking(position, current_price)
                        if partial_info:
                            if partial_info.get('action') == 'close_full':
                                # Manual stop hit - close full position
                                self.logger.warning(f"🚨 Closing position due to manual stop hit: {position.symbol}")
                                if self.close_position(position):
                                    self.update_position_in_database(position, 'closed', 'manual_stop_hit')
                                    self.logger.info(f"✅ Position closed due to manual stop: {position.symbol}")
                                continue
                            elif partial_info.get('action') == 'close_partial':
                                if self.execute_partial_close(position, partial_info):
                                    self.logger.info(f"✅ Partial profit taken for {position.symbol}")
                                    partial_taken = True
                        
                        # PHASE 2: Check auto-close conditions (skip if partial was just taken)
                        if not partial_taken and self.config.auto_close_enabled:
                            should_close, close_reason = self.should_auto_close(position, current_price)
                            
                            if should_close:
                                leveraged_profit = self.calculate_leveraged_profit_pct(position, current_price)
                                
                                # Log appropriate message based on close reason
                                if close_reason == 'profit_target':
                                    self.logger.info(
                                        f"💰 Profit target reached for {position.symbol}: "
                                        f"{leveraged_profit:.2f}% profit (target: {self.config.auto_close_profit_at}%)"
                                    )
                                elif close_reason == 'loss_limit':
                                    self.logger.warning(
                                        f"🛑 Stop loss triggered for {position.symbol}: "
                                        f"{leveraged_profit:.2f}% loss (limit: -{abs(self.config.auto_close_loss_at)}%)"
                                    )
                                
                                # Attempt to close position
                                if self.close_position(position):
                                    # Update database record
                                    self.update_position_in_database(position, 'closed', close_reason)
                                    
                                    # Clean up position tracker
                                    if position.symbol in self.position_tracker:
                                        del self.position_tracker[position.symbol]
                                    
                                    self.logger.info(f"✅ Successfully closed position {position.symbol}")
                        
                        # PHASE 3: Log monitoring status for positions with manual stops
                        if position.symbol in self.position_tracker:
                            tracker = self.position_tracker[position.symbol]
                            
                            if tracker.get('manual_stop_needed', False):
                                manual_stop = tracker.get('manual_stop_price', 0)
                                self.logger.debug(
                                    f"⚠️ Manual stop monitoring for {position.symbol}: "
                                    f"Stop at ${manual_stop:.6f}, Current: ${current_price:.6f}"
                                )
                            
                    except Exception as e:
                        self.logger.error(f"Error monitoring position {position.symbol}: {str(e)}")
                
                # Sleep between monitoring cycles
                time.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {str(e)}")
            
    def start_monitoring(self):
        """Start position monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)
            self.monitor_thread.start()
            self.logger.info("📊 Started enhanced position monitoring:")
            self.logger.info("   💰 Partial profits: 50% at 100%, 200%, 300% leveraged profit")
            self.logger.info("   🎯 Trailing stops: Auto-update at partial profit milestones")
            self.logger.info("   ⏱️ Check interval: 10 seconds")
    
    def stop_monitoring(self):
        """Stop position monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛑 Stopped position monitoring")
    
    def update_position_in_database(self, position: PositionData, status: str, close_reason: str = None):
        """Update position status in database"""
        # This would be implemented to update the TradingPosition table
        # Including milestone information for analysis
        pass


# ========================================
# PHASE 6: ORDER EXECUTION
# ========================================

class OrderExecutor:
    """Execute leveraged orders with proper sizing - ENHANCED with market regime awareness"""
    
    def __init__(self, exchange, config: EnhancedSystemConfig, leverage_manager: LeverageManager, position_sizer: PositionSizer):
        self.exchange = exchange
        self.config = config
        self.leverage_manager = leverage_manager
        self.position_sizer = position_sizer
        self.logger = logging.getLogger(__name__)
    
    def check_market_regime_compatibility(self, signal: Dict) -> Tuple[bool, str]:
        """
        ENHANCED: Check if signal is compatible with current market regime
        
        Returns:
            Tuple of (is_compatible, regime_info)
        """
        try:
            # Simple market regime detection based on signal data
            price_change_24h = signal.get('price_change_24h', 0)
            side = signal['side']
            confidence = signal.get('confidence', 50)
            
            # Determine regime
            if price_change_24h > 5:
                regime = 'strong_bullish'
            elif price_change_24h > 2:
                regime = 'bullish'
            elif price_change_24h < -5:
                regime = 'strong_bearish'
            elif price_change_24h < -2:
                regime = 'bearish'
            else:
                regime = 'ranging'
            
            # Check compatibility
            if regime == 'strong_bullish' and side == 'sell':
                # Only allow very strong SHORT signals in bull market
                if confidence < 60:
                    return False, f"SHORT signal too weak for {regime} market"
            elif regime == 'strong_bearish' and side == 'buy':
                # Only allow very strong LONG signals in bear market
                if confidence < 60:
                    return False, f"LONG signal too weak for {regime} market"
            
            return True, f"Compatible with {regime} regime"
            
        except Exception as e:
            self.logger.error(f"Market regime check error: {e}")
            return True, "regime_check_failed"
    
    def check_correlation_risk(self, signal: Dict, existing_positions: List = None) -> Tuple[bool, str]:
        """
        ENHANCED: Check correlation risk before opening new position
        
        Returns:
            Tuple of (is_safe, risk_info)
        """
        try:
            if existing_positions is None:
                # Get current positions from exchange
                positions = self.exchange.fetch_positions()
                existing_positions = [pos for pos in positions if pos.get('contracts', 0) > 0]
            
            new_symbol = signal['symbol']
            new_side = signal['side']
            
            # Define correlation groups
            correlation_groups = {
                'BTC': ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'ADA/USDT:USDT'],
                'ETH': ['ETH/USDT:USDT', 'BTC/USDT:USDT', 'MATIC/USDT:USDT', 'LINK/USDT:USDT'],
                'MAJOR_ALTS': ['ADA/USDT:USDT', 'DOT/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT'],
                'DEFI': ['UNI/USDT:USDT', 'SUSHI/USDT:USDT', 'AAVE/USDT:USDT', 'COMP/USDT:USDT']
            }
            
            # Find which group the new symbol belongs to
            new_group = None
            for group_name, symbols in correlation_groups.items():
                if new_symbol in symbols:
                    new_group = group_name
                    break
            
            if not new_group:
                return True, "No correlation group found"
            
            # Count same-direction positions in the correlation group
            same_direction_count = 0
            correlated_symbols = correlation_groups[new_group]
            
            for position in existing_positions:
                pos_symbol = position['symbol']
                pos_side = 'buy' if position['side'].lower() == 'long' else 'sell'
                
                if pos_symbol in correlated_symbols and pos_side == new_side:
                    same_direction_count += 1
            
            # Risk limits
            max_correlated_positions = 2
            
            if same_direction_count >= max_correlated_positions:
                return False, f"Too many {new_side} positions in {new_group} group ({same_direction_count}/{max_correlated_positions})"
            
            return True, f"Correlation risk acceptable ({same_direction_count}/{max_correlated_positions} in {new_group})"
            
        except Exception as e:
            self.logger.error(f"Correlation risk check error: {e}")
            return True, "correlation_check_failed"
    
    def place_leveraged_order(self, signal: Dict, risk_amount_pct: float, leverage_str: str) -> Tuple[bool, str, Dict]:
        """Place a leveraged order based on signal with percentage-based risk - ENHANCED"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = 0

            take_profit = signal.get('take_profit_1', signal.get('take_profit', 0)) if self.config.default_tp_level == 'take_profit_1' else signal.get('take_profit_2', signal.get('take_profit', 0))
            
            self.logger.info(f"🚀 Placing {side.upper()} order for {symbol}")
            
            # ENHANCED: Market regime compatibility check
            regime_compatible, regime_info = self.check_market_regime_compatibility(signal)
            if not regime_compatible:
                return False, f"Market regime filter: {regime_info}", {}
            else:
                self.logger.debug(f"📊 Market regime: {regime_info}")
            
            # ENHANCED: Correlation risk check
            correlation_safe, correlation_info = self.check_correlation_risk(signal)
            if not correlation_safe:
                return False, f"Correlation risk: {correlation_info}", {}
            else:
                self.logger.debug(f"🔗 Correlation: {correlation_info}")
            
            if self.config.sandbox_mode:
                risk_amount_pct = 0.001  # Use a smaller investment capital in sandbox mode to avoid max qty limit

            # Convert leverage to float
            leverage = self.leverage_manager.convert_leverage_to_float(leverage_str, symbol)
            
            # Set leverage on exchange
            if not self.leverage_manager.set_symbol_leverage(symbol, leverage):
                return False, "Failed to set leverage", {}
            
            # ENHANCED: Use adaptive position sizing
            position_size, adjusted_risk_pct = self.position_sizer.calculate_adaptive_position_size(
                risk_amount_pct, leverage, entry_price, signal
            )
            
            # Validate position size
            is_valid, adjusted_size = self.position_sizer.validate_position_size(symbol, position_size)
            if not is_valid:
                return False, f"Invalid position size: {position_size}", {}
            
            position_size = adjusted_size
            
            # Check available balance and calculate required margin
            available_balance = self.position_sizer.get_available_balance()
            required_margin = (position_size * entry_price) / leverage
            risk_amount_usdt = available_balance * (adjusted_risk_pct / 100)
            
            if required_margin > available_balance:
                return False, f"Insufficient balance: need {required_margin}, have {available_balance}", {}
            
            self.logger.info(f"💰 Enhanced risk calculation:")
            self.logger.info(f"   Account Balance: {available_balance:.2f} USDT")
            self.logger.info(f"   Risk Percentage: {risk_amount_pct:.0f}% → {adjusted_risk_pct:.1f}% (adaptive)")
            self.logger.info(f"   Risk Amount: {risk_amount_usdt:.2f} USDT")
            self.logger.info(f"   Position Size: {position_size:.2f} units")
            self.logger.info(f"   Required Margin: {required_margin:.2f} USDT")
            self.logger.info(f"   Leverage: {leverage}x (trailing stops will activate at partial milestones)")
            
            # Use lowercase side for CCXT
            ccxt_side = side.lower()
            
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
                    side=ccxt_side,
                    amount=position_size,
                    params=order_params  # Include SL/TP in main order
                )
            else:
                entry_order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=ccxt_side,
                    amount=position_size,
                    price=entry_price,
                    params=order_params  # Include SL/TP in main order
                )
            
            self.logger.info(f"✅ Entry order placed: {entry_order['id']}")
            if stop_loss > 0:
                self.logger.debug(f"✅ Stop loss integrated: ${stop_loss:.6f}")
            if take_profit > 0:
                self.logger.debug(f"✅ Take profit integrated: ${take_profit:.6f}")
            
            # Note about trailing stops
            self.logger.info(f"📈 Trailing stops will activate at partial profit milestones (leverage: {leverage}x)")
            
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
                'risk_percentage': adjusted_risk_pct,  # Store adaptive percentage
                'original_risk_percentage': risk_amount_pct,  # Store original percentage
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_order_id': entry_order['id'],
                'stop_loss_order_id': sl_order['id'] if sl_order else None,
                'take_profit_order_id': tp_order['id'] if tp_order else None,
                'signal_confidence': signal.get('confidence', 0),
                'mtf_status': signal.get('mtf_status', ''),
                'mtf_validated': signal.get('mtf_validated', False),
                'entry_strategy': signal.get('entry_strategy', 'immediate'),
                'market_regime': regime_info,
                'correlation_info': correlation_info,
                'auto_close_profit_target': self.config.auto_close_profit_at,
                'auto_close_loss_target': self.config.auto_close_loss_at,
                'partial_trailing_stops_enabled': True,  # NEW: Flag for partial milestone trailing stops
                'leverage_for_milestones': leverage  # NEW: Store leverage for milestone calculation
            }
            self.logger.info("=" * 100 + "\n")

            return True, "Order placed successfully", position_data
            
        except Exception as e:
            error_msg = f"Failed to place order for {signal.get('symbol', 'unknown')}: {e}"
            self.logger.error(error_msg)
            return False, error_msg, {}


# ========================================
# PHASE 7: POSITION MANAGEMENT
# ========================================

class PositionManager:
    """Track and manage concurrent positions - ENHANCED with portfolio heat tracking"""
    
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
    
    def calculate_portfolio_heat(self) -> float:
        """
        ENHANCED: Calculate total portfolio risk exposure (portfolio heat)
        
        Returns:
            Portfolio heat as percentage of account balance
        """
        try:
            positions = self.exchange.fetch_positions()
            total_risk = 0
            
            for position in positions:
                if position.get('contracts', 0) > 0:
                    # Estimate risk based on position value and typical stop distance
                    position_value = position['contracts'] * position['entryPrice']
                    estimated_risk = position_value * (self.config.risk_amount / 100)  # Assume 5% risk per position
                    total_risk += estimated_risk
            
            # Get account balance
            balance = self.exchange.fetch_balance()
            account_balance = balance['USDT']['total']
            
            portfolio_heat = (total_risk / account_balance) * 100 if account_balance > 0 else 0
            
            self.logger.debug(f"🔥 Portfolio Heat: {portfolio_heat:.1f}% (${total_risk:.2f} risk on ${account_balance:.2f})")
            
            return portfolio_heat
            
        except Exception as e:
            self.logger.error(f"Portfolio heat calculation error: {e}")
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
    
    def filter_signals_by_existing_symbols(self, signals: List[Dict]) -> List[Dict]:
        """Filter out signals for symbols that already have positions or orders"""
        try:
            symbols_with_positions = self.get_symbols_with_positions()
            symbols_with_orders = self.get_symbols_with_orders()
            excluded_symbols = symbols_with_positions.union(symbols_with_orders)
            
            filtered_signals = []
            skipped_count = 0
            
            for signal in signals:
                symbol = signal['symbol']
                if symbol in excluded_symbols:
                    reason = ""
                    if symbol in symbols_with_positions:
                        reason = "existing position"
                    elif symbol in symbols_with_orders:
                        reason = "pending orders"
                    
                    self.logger.info(f"⭕️ Skipping {symbol} - {reason}")
                    skipped_count += 1
                else:
                    filtered_signals.append(signal)
            
            if skipped_count > 0:
                self.logger.info(f"🔍 Filtered out {skipped_count} symbols with existing exposure")
                
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering signals by existing symbols: {e}")
            return signals
    
    def can_open_new_positions(self, requested_count: int) -> Tuple[bool, int]:
        """Check if we can open new positions - ENHANCED with portfolio heat check"""
        try:
            current_count = self.get_current_positions_count()
            max_positions = self.config.max_concurrent_positions
            available_slots = max_positions - current_count
            
            # Check portfolio heat limit
            portfolio_heat = self.calculate_portfolio_heat()
            max_portfolio_heat = 40.0  # Max 40% of account at risk
            
            if portfolio_heat > max_portfolio_heat:
                self.logger.warning(f"🔥 Portfolio heat too high: {portfolio_heat:.1f}% > {max_portfolio_heat}%")
                return False, 0
            
            if available_slots <= 0:
                return False, 0
            
            can_open = min(requested_count, available_slots)
            
            self.logger.debug(f"📊 Position capacity: {current_count}/{max_positions} positions, {portfolio_heat:.1f}% heat")
            
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
