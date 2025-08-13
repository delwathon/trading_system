"""
Signal Generator V13.0 - Entry Monitoring and Risk Management
=============================================================
PART 4: Entry Monitoring Service, Risk Management, and Position Sizing

This module provides:
- Real-time entry condition monitoring
- Dynamic risk management
- Position sizing calculations
- Stop loss and take profit optimization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, time, timedelta
import pandas as pd
import numpy as np
from enum import Enum

# Import from previous parts
from signals.signal_gen_v13_core import (
    Signal, SignalStatus, SignalQuality, TimeFrame, MarketRegime,
    SystemConfiguration, SignalCriteria, StateManager,
    EntryMonitor, RiskManager
)

from signals.signal_gen_v13_analysis import TechnicalIndicators

# ===========================
# ENTRY MONITORING SERVICE
# ===========================

class EntryCondition(Enum):
    """Entry condition types"""
    PRICE_LEVEL = "price_level"
    INDICATOR_THRESHOLD = "indicator_threshold"
    PATTERN_COMPLETION = "pattern_completion"
    VOLUME_SURGE = "volume_surge"
    TIME_BASED = "time_based"
    MARKET_OPEN = "market_open"

@dataclass
class EntryTrigger:
    """Entry trigger configuration"""
    condition_type: EntryCondition
    parameters: Dict[str, Any]
    priority: int  # 1-10, higher = more important
    mandatory: bool
    description: str
    
    def check(self, current_data: Dict) -> bool:
        """Check if trigger condition is met"""
        if self.condition_type == EntryCondition.PRICE_LEVEL:
            return self._check_price_level(current_data)
        elif self.condition_type == EntryCondition.INDICATOR_THRESHOLD:
            return self._check_indicator(current_data)
        elif self.condition_type == EntryCondition.VOLUME_SURGE:
            return self._check_volume(current_data)
        elif self.condition_type == EntryCondition.PATTERN_COMPLETION:
            return self._check_pattern(current_data)
        elif self.condition_type == EntryCondition.TIME_BASED:
            return self._check_time(current_data)
        else:
            return False
    
    def _check_price_level(self, data: Dict) -> bool:
        """Check if price level condition is met"""
        current_price = data.get('price', 0)
        target_price = self.parameters.get('target_price', 0)
        tolerance = self.parameters.get('tolerance', 0.001)
        direction = self.parameters.get('direction', 'cross')
        
        if direction == 'cross':
            # Price crossed the level
            prev_price = data.get('prev_price', current_price)
            return (prev_price < target_price <= current_price) or \
                   (prev_price > target_price >= current_price)
        elif direction == 'above':
            return current_price > target_price * (1 - tolerance)
        elif direction == 'below':
            return current_price < target_price * (1 + tolerance)
        else:
            return abs(current_price - target_price) / target_price < tolerance
    
    def _check_indicator(self, data: Dict) -> bool:
        """Check if indicator threshold is met"""
        indicator_name = self.parameters.get('indicator')
        threshold = self.parameters.get('threshold')
        operator = self.parameters.get('operator', '>')
        
        value = data.get('indicators', {}).get(indicator_name)
        if value is None:
            return False
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 0.01
        else:
            return False
    
    def _check_volume(self, data: Dict) -> bool:
        """Check if volume condition is met"""
        volume_ratio = data.get('indicators', {}).get('volume_ratio', 1)
        min_ratio = self.parameters.get('min_ratio', 1.5)
        return volume_ratio >= min_ratio
    
    def _check_pattern(self, data: Dict) -> bool:
        """Check if pattern is completed"""
        pattern_name = self.parameters.get('pattern')
        patterns = data.get('patterns', [])
        return pattern_name in patterns
    
    def _check_time(self, data: Dict) -> bool:
        """Check if time-based condition is met"""
        current_time = datetime.now(timezone.utc)
        
        # Check if within time window
        start_hour = self.parameters.get('start_hour')
        end_hour = self.parameters.get('end_hour')
        
        if start_hour and end_hour:
            current_hour = current_time.hour
            if start_hour < end_hour:
                return start_hour <= current_hour < end_hour
            else:  # Crosses midnight
                return current_hour >= start_hour or current_hour < end_hour
        
        return True

class EntryMonitoringService(EntryMonitor):
    """Service for monitoring and executing entry conditions"""
    
    def __init__(self, state_manager: StateManager, config: SystemConfiguration):
        self.state_manager = state_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Monitoring tasks
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._is_running = False
        
        # Price update callbacks
        self._price_callbacks: Dict[str, List] = {}
    
    async def start_monitoring(self):
        """Start the entry monitoring service"""
        if self._is_running:
            return
        
        self._is_running = True
        self.logger.info("Entry monitoring service started")
        
        # Start main monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop the monitoring service"""
        self._is_running = False
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        self._monitoring_tasks.clear()
        self.logger.info("Entry monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._is_running:
            try:
                # Get signals to monitor
                monitoring_signals = self.state_manager.get_monitoring_signals()
                
                for signal in monitoring_signals:
                    if signal.id not in self._monitoring_tasks:
                        # Start monitoring this signal
                        task = asyncio.create_task(self._monitor_signal(signal))
                        self._monitoring_tasks[signal.id] = task
                
                # Clean up completed tasks
                completed_ids = [
                    sid for sid, task in self._monitoring_tasks.items() 
                    if task.done()
                ]
                for sid in completed_ids:
                    del self._monitoring_tasks[sid]
                
                # Wait before next iteration
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_signal(self, signal: Signal):
        """Monitor signal with proper lifecycle management"""
        start_time = time.time()
        max_monitoring_time = 3600  # 1 hour max monitoring
        
        try:
            self.logger.info(f"Starting monitoring for signal {signal.id}")
            
            while signal.status == SignalStatus.MONITORING:
                # Timeout check to prevent infinite monitoring
                if time.time() - start_time > max_monitoring_time:
                    self.logger.warning(f"Signal {signal.id} monitoring timeout")
                    self.state_manager.update_signal(signal.id, {'status': SignalStatus.EXPIRED})
                    break
                
                # Check if signal still exists in state
                current_signal = self.state_manager._signals.get(signal.id)
                if not current_signal:
                    self.logger.warning(f"Signal {signal.id} no longer in state")
                    break
                
                # Get current data with timeout
                try:
                    current_data = await asyncio.wait_for(
                        self._get_current_data(signal.symbol), 
                        timeout=5.0
                    )
                    
                    if current_data:
                        is_triggered, trigger_info = await self.check_entry_conditions(
                            signal, current_data
                        )
                        
                        if is_triggered:
                            self.state_manager.update_signal(
                                signal.id,
                                {
                                    'status': SignalStatus.TRIGGERED,
                                    'triggered_at': datetime.now(timezone.utc),
                                    'trigger_info': trigger_info
                                }
                            )
                            self.logger.info(f"Signal {signal.id} triggered")
                            break
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Data fetch timeout for {signal.symbol}")
                
                # Check expiry
                age = (datetime.now(timezone.utc) - signal.created_at).total_seconds() / 60
                if age > self.config.signal_expiry_minutes:
                    self.state_manager.update_signal(signal.id, {'status': SignalStatus.EXPIRED})
                    break
                
                # Wait before next check
                await asyncio.sleep(10)
        
        except asyncio.CancelledError:
            self.logger.info(f"Signal {signal.id} monitoring cancelled")
            raise  # Re-raise cancellation
        except Exception as e:
            self.logger.error(f"Signal monitoring error for {signal.id}: {e}")
            self.state_manager.update_signal(signal.id, {'status': SignalStatus.FAILED})
        finally:
            # Cleanup monitoring task reference
            self._monitoring_tasks.pop(signal.id, None)
            self.logger.debug(f"Monitoring cleanup completed for {signal.id}")

    async def check_entry_conditions(self, signal: Signal, 
                                    current_data: Dict) -> Tuple[bool, Dict]:
        """Check if entry conditions are met"""
        try:
            triggers = self._create_entry_triggers(signal)
            
            mandatory_met = []
            optional_met = []
            
            for trigger in triggers:
                if trigger.check(current_data):
                    if trigger.mandatory:
                        mandatory_met.append(trigger)
                    else:
                        optional_met.append(trigger)
            
            # All mandatory triggers must be met
            all_mandatory = all(
                t in mandatory_met for t in triggers 
                if t.mandatory
            )
            
            # At least some optional triggers should be met
            sufficient_optional = len(optional_met) >= 2 or not [
                t for t in triggers if not t.mandatory
            ]
            
            is_triggered = all_mandatory and sufficient_optional
            
            trigger_info = {
                'mandatory_met': [t.description for t in mandatory_met],
                'optional_met': [t.description for t in optional_met],
                'trigger_price': current_data.get('price'),
                'trigger_time': datetime.now(timezone.utc).isoformat()
            }
            
            return is_triggered, trigger_info
            
        except Exception as e:
            self.logger.error(f"Entry condition check error: {e}")
            return False, {}
    
    async def update_signal_prices(self, signal: Signal, 
                                  current_data: Dict) -> Signal:
        """Update signal with current market prices"""
        try:
            current_price = current_data.get('price', signal.current_price)
            
            # Update current price
            signal.current_price = current_price
            
            # Adjust entry if needed (for limit orders)
            if signal.status == SignalStatus.MONITORING:
                # Check if we need to adjust entry
                if signal.side == 'buy' and current_price < signal.entry_price * 0.995:
                    # Price dropped, might get better entry
                    signal.entry_price = min(signal.entry_price, current_price * 1.001)
                elif signal.side == 'sell' and current_price > signal.entry_price * 1.005:
                    # Price rose, might get better entry
                    signal.entry_price = max(signal.entry_price, current_price * 0.999)
            
            # Update in state manager
            self.state_manager.update_signal(
                signal.id,
                {
                    'current_price': current_price,
                    'entry_price': signal.entry_price,
                    'last_updated': datetime.now(timezone.utc)
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Price update error: {e}")
            return signal
    
    def _create_entry_triggers(self, signal: Signal) -> List[EntryTrigger]:
        """Create entry triggers for a signal"""
        triggers = []
        
        # Price level trigger (mandatory)
        triggers.append(EntryTrigger(
            condition_type=EntryCondition.PRICE_LEVEL,
            parameters={
                'target_price': signal.entry_price,
                'tolerance': 0.002,
                'direction': 'near'
            },
            priority=10,
            mandatory=True,
            description=f"Price near entry level {signal.entry_price:.6f}"
        ))
        
        # RSI confirmation (mandatory for entry timeframe)
        if signal.side == 'buy':
            triggers.append(EntryTrigger(
                condition_type=EntryCondition.INDICATOR_THRESHOLD,
                parameters={
                    'indicator': 'rsi',
                    'threshold': 35,
                    'operator': '<='
                },
                priority=8,
                mandatory=False,  # <-- CHANGE THIS
                description="RSI oversold for entry"
            ))
        else:
            triggers.append(EntryTrigger(
                condition_type=EntryCondition.INDICATOR_THRESHOLD,
                parameters={
                    'indicator': 'rsi',
                    'threshold': 65,
                    'operator': '>='
                },
                priority=8,
                mandatory=False,
                description="RSI overbought for entry"
            ))
        
        # Volume surge (optional but preferred)
        triggers.append(EntryTrigger(
            condition_type=EntryCondition.VOLUME_SURGE,
            parameters={'min_ratio': 1.3},
            priority=5,
            mandatory=False,
            description="Volume surge detected"
        ))
        
        # Stochastic confirmation (optional)
        if signal.side == 'buy':
            triggers.append(EntryTrigger(
                condition_type=EntryCondition.INDICATOR_THRESHOLD,
                parameters={
                    'indicator': 'stoch_k',
                    'threshold': 30,
                    'operator': '<'
                },
                priority=6,
                mandatory=False,
                description="Stochastic oversold"
            ))
        else:
            triggers.append(EntryTrigger(
                condition_type=EntryCondition.INDICATOR_THRESHOLD,
                parameters={
                    'indicator': 'stoch_k',
                    'threshold': 70,
                    'operator': '>'
                },
                priority=6,
                mandatory=False,
                description="Stochastic overbought"
            ))
        
        return triggers
    
    async def _get_current_data(self, symbol: str) -> Optional[Dict]:
        """Get current market data for symbol"""
        # This would connect to exchange/data provider
        # Placeholder implementation
        return {
            'price': 50000 + np.random.randn() * 100,
            'prev_price': 50000,
            'indicators': {
                'rsi': 30 + np.random.rand() * 40,
                'stoch_k': 20 + np.random.rand() * 60,
                'volume_ratio': 0.8 + np.random.rand() * 1.2,
                'macd': np.random.randn() * 10
            },
            'patterns': [],
            'timestamp': datetime.now(timezone.utc)
        }

# ===========================
# RISK MANAGEMENT SYSTEM
# ===========================

@dataclass
class RiskProfile:
    """User/System risk profile"""
    max_risk_per_trade: float = 0.02      # 2% default
    max_daily_risk: float = 0.06          # 6% daily
    max_open_positions: int = 10
    max_correlated_positions: int = 3
    max_leverage: int = 10
    preferred_risk_reward: float = 2.5
    risk_adjustment_factor: float = 1.0   # Can be adjusted based on performance

@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    position_size: float
    position_value: float
    risk_amount: float
    leverage_used: float
    margin_required: float
    max_loss: float
    max_profit: float
    kelly_fraction: float
    risk_warnings: List[str]

class AdvancedRiskManager(RiskManager):
    """Advanced risk management system"""
    
    def __init__(self, config: SystemConfiguration, risk_profile: RiskProfile):
        self.config = config
        self.risk_profile = risk_profile
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track current exposure
        self.current_positions: List[Signal] = []
        self.daily_pnl = 0.0
        self.daily_risk_used = 0.0
    
    def calculate_position_size(self, signal: Signal, 
                               account_balance: float) -> PositionSizeResult:
        """Calculate optimal position size using multiple methods"""
        try:
            # Get base position size using fixed risk
            fixed_risk_size = self._fixed_risk_position_size(
                signal, account_balance
            )
            
            # Kelly Criterion adjustment
            kelly_size = self._kelly_criterion_size(
                signal, account_balance
            )
            
            # Volatility-based adjustment
            volatility_adjusted = self._volatility_adjusted_size(
                fixed_risk_size, signal
            )
            
            # Correlation adjustment
            correlation_adjusted = self._correlation_adjusted_size(
                volatility_adjusted, signal
            )
            
            # Apply limits
            final_size = self._apply_position_limits(
                correlation_adjusted, signal, account_balance
            )
            
            # Calculate metrics
            position_value = final_size * signal.entry_price
            risk_amount = final_size * abs(signal.entry_price - signal.stop_loss)
            max_loss = risk_amount
            max_profit = final_size * abs(signal.take_profit_1 - signal.entry_price)
            
            # Determine leverage
            leverage_used = position_value / account_balance
            margin_required = position_value / self.risk_profile.max_leverage
            
            # Risk warnings
            warnings = self._check_risk_warnings(
                final_size, signal, account_balance, leverage_used
            )
            
            return PositionSizeResult(
                position_size=final_size,
                position_value=position_value,
                risk_amount=risk_amount,
                leverage_used=leverage_used,
                margin_required=margin_required,
                max_loss=max_loss,
                max_profit=max_profit,
                kelly_fraction=kelly_size / fixed_risk_size if fixed_risk_size > 0 else 1,
                risk_warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return PositionSizeResult(
                position_size=0,
                position_value=0,
                risk_amount=0,
                leverage_used=0,
                margin_required=0,
                max_loss=0,
                max_profit=0,
                kelly_fraction=0,
                risk_warnings=["Calculation error"]
            )
    
    def validate_risk_limits(self, signal: Signal, 
                           current_positions: List[Signal]) -> Tuple[bool, str]:
        """Validate if signal passes risk limits"""
        try:
            # Check max positions
            if len(current_positions) >= self.risk_profile.max_open_positions:
                return False, "Maximum open positions reached"
            
            # Check daily risk limit
            total_risk = sum(p.risk_amount for p in current_positions)
            if total_risk >= self.risk_profile.max_daily_risk:
                return False, "Daily risk limit reached"
            
            # Check correlation limits
            correlated = self._count_correlated_positions(signal, current_positions)
            if correlated >= self.risk_profile.max_correlated_positions:
                return False, "Too many correlated positions"
            
            # Check if opposite position exists
            opposite_exists = any(
                p.symbol == signal.symbol and p.side != signal.side 
                for p in current_positions
            )
            if opposite_exists:
                return False, "Opposite position already exists"
            
            # Check minimum risk/reward
            if signal.risk_reward_ratio < self.config.min_risk_reward:
                return False, f"Risk/reward too low ({signal.risk_reward_ratio:.2f})"
            
            return True, "Pass"
            
        except Exception as e:
            self.logger.error(f"Risk validation error: {e}")
            return False, "Validation error"
    
    def adjust_stops_and_targets(self, signal: Signal, 
                                market_conditions: Dict) -> Signal:
        """Dynamically adjust stops and targets based on conditions"""
        try:
            # Get market volatility
            volatility = market_conditions.get('volatility', 0.02)
            trend_strength = market_conditions.get('trend_strength', 0.5)
            
            # Adjust stop loss based on volatility
            if volatility > 0.03:  # High volatility
                # Widen stops
                if signal.side == 'buy':
                    signal.stop_loss *= 0.98  # 2% wider
                else:
                    signal.stop_loss *= 1.02
            elif volatility < 0.01:  # Low volatility
                # Tighten stops
                if signal.side == 'buy':
                    signal.stop_loss *= 1.005  # 0.5% tighter
                else:
                    signal.stop_loss *= 0.995
            
            # Adjust targets based on trend strength
            if trend_strength > 0.7:  # Strong trend
                # Extend targets
                if signal.side == 'buy':
                    signal.take_profit_1 *= 1.05
                    signal.take_profit_2 *= 1.08
                else:
                    signal.take_profit_1 *= 0.95
                    signal.take_profit_2 *= 0.92
            elif trend_strength < 0.3:  # Weak trend
                # Reduce targets
                if signal.side == 'buy':
                    signal.take_profit_1 *= 0.98
                    signal.take_profit_2 *= 0.96
                else:
                    signal.take_profit_1 *= 1.02
                    signal.take_profit_2 *= 1.04
            
            # Recalculate risk/reward
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit_1 - signal.entry_price)
            signal.risk_reward_ratio = reward / risk if risk > 0 else 0
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Stop/target adjustment error: {e}")
            return signal
    
    def calculate_portfolio_heat(self, current_positions: List[Signal]) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        if not current_positions:
            return 0.0
        
        total_risk = sum(p.risk_amount for p in current_positions)
        return total_risk
    
    def get_risk_metrics(self, current_positions: List[Signal], 
                        account_balance: float) -> Dict:
        """Get comprehensive risk metrics"""
        if not current_positions:
            return {
                'portfolio_heat': 0,
                'position_count': 0,
                'total_exposure': 0,
                'average_rr': 0,
                'correlation_risk': 'low',
                'var_95': 0,
                'max_drawdown_potential': 0
            }
        
        # Calculate metrics
        portfolio_heat = self.calculate_portfolio_heat(current_positions)
        total_exposure = sum(p.position_size * p.entry_price for p in current_positions)
        avg_rr = np.mean([p.risk_reward_ratio for p in current_positions])
        
        # VaR calculation (simplified)
        returns = [p.potential_profit / p.risk_amount - 1 for p in current_positions]
        if returns:
            var_95 = np.percentile(returns, 5) * portfolio_heat
        else:
            var_95 = 0
        
        # Max drawdown potential
        max_dd = sum(p.risk_amount for p in current_positions)
        
        # Correlation risk
        unique_symbols = len(set(p.symbol for p in current_positions))
        correlation_risk = 'high' if unique_symbols < 3 else 'medium' if unique_symbols < 5 else 'low'
        
        return {
            'portfolio_heat': portfolio_heat,
            'portfolio_heat_pct': (portfolio_heat / account_balance) * 100,
            'position_count': len(current_positions),
            'total_exposure': total_exposure,
            'exposure_pct': (total_exposure / account_balance) * 100,
            'average_rr': avg_rr,
            'correlation_risk': correlation_risk,
            'var_95': var_95,
            'max_drawdown_potential': max_dd,
            'max_drawdown_pct': (max_dd / account_balance) * 100
        }
    
    def _fixed_risk_position_size(self, signal: Signal, 
                                 account_balance: float) -> float:
        """Calculate position size using fixed risk percentage"""
        risk_amount = account_balance * self.risk_profile.max_risk_per_trade
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0
        
        return position_size
    
    def _kelly_criterion_size(self, signal: Signal, 
                             account_balance: float) -> float:
        """Calculate position size using Kelly Criterion"""
        # Estimate win probability based on signal quality
        quality_prob_map = {
            SignalQuality.ELITE: 0.70,
            SignalQuality.PREMIUM: 0.65,
            SignalQuality.STANDARD: 0.60,
            SignalQuality.MARGINAL: 0.55
        }
        
        win_prob = quality_prob_map.get(signal.quality_tier, 0.55)
        
        # Calculate Kelly fraction
        # f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = signal.risk_reward_ratio
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # Apply Kelly fraction with safety factor (use 25% of Kelly)
        kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.25))
        
        # Calculate position size
        risk_amount = account_balance * kelly_fraction
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0
        
        return position_size
    
    def _volatility_adjusted_size(self, base_size: float, 
                                 signal: Signal) -> float:
        """Adjust position size based on volatility"""
        # Get ATR-based volatility (would come from signal analysis)
        atr_percent = signal.indicators.get('atr_percent', 2.0)
        
        # Inverse volatility sizing
        # Lower volatility = larger position, higher volatility = smaller position
        target_volatility = 2.0  # 2% target
        
        if atr_percent > 0:
            adjustment = target_volatility / atr_percent
            adjustment = max(0.5, min(adjustment, 1.5))  # Cap adjustment
        else:
            adjustment = 1.0
        
        return base_size * adjustment
    
    def _correlation_adjusted_size(self, base_size: float, 
                                  signal: Signal) -> float:
        """Adjust size based on correlation with existing positions"""
        if not self.current_positions:
            return base_size
        
        # Count positions in same asset class or correlated assets
        correlated_count = self._count_correlated_positions(
            signal, self.current_positions
        )
        
        # Reduce size for each correlated position
        reduction_factor = 1.0 - (correlated_count * 0.15)
        reduction_factor = max(0.4, reduction_factor)  # Minimum 40% of base
        
        return base_size * reduction_factor
    
    def _apply_position_limits(self, size: float, signal: Signal, 
                              account_balance: float) -> float:
        """Apply various position limits"""
        # Maximum position value
        max_position_value = account_balance * 0.1  # 10% max per position
        max_size_by_value = max_position_value / signal.entry_price
        
        # Maximum based on leverage
        max_leverage_value = account_balance * self.risk_profile.max_leverage
        max_size_by_leverage = max_leverage_value / signal.entry_price
        
        # Apply limits
        final_size = min(size, max_size_by_value, max_size_by_leverage)
        
        # Minimum position size (to avoid dust)
        min_position_value = account_balance * 0.001  # 0.1% minimum
        min_size = min_position_value / signal.entry_price
        
        if final_size < min_size:
            return 0  # Too small, don't trade
        
        return final_size
    
    def _count_correlated_positions(self, signal: Signal, 
                                   positions: List[Signal]) -> int:
        """Count correlated positions"""
        correlated = 0
        
        # Same symbol
        correlated += sum(1 for p in positions if p.symbol == signal.symbol)
        
        # Same market (simplified - would need correlation matrix)
        # For crypto, consider BTC, ETH as correlated
        major_cryptos = ['BTCUSDT', 'ETHUSDT', 'BTCUSD', 'ETHUSD']
        if signal.symbol in major_cryptos:
            correlated += sum(1 for p in positions if p.symbol in major_cryptos) - 1
        
        return correlated
    
    def _check_risk_warnings(self, size: float, signal: Signal,
                            account_balance: float, leverage: float) -> List[str]:
        """Check for risk warnings"""
        warnings = []
        
        # High leverage warning
        if leverage > 5:
            warnings.append(f"High leverage used ({leverage:.1f}x)")
        
        # Large position warning
        position_pct = (size * signal.entry_price) / account_balance * 100
        if position_pct > 5:
            warnings.append(f"Large position size ({position_pct:.1f}% of account)")
        
        # Poor risk/reward warning
        if signal.risk_reward_ratio < 1.5:
            warnings.append(f"Low risk/reward ratio ({signal.risk_reward_ratio:.2f})")
        
        # Volatility warning
        if signal.indicators.get('atr_percent', 0) > 5:
            warnings.append("High volatility environment")
        
        return warnings

# ===========================
# TRAILING STOP MANAGER
# ===========================

class TrailingStopManager:
    """Manage trailing stops for active positions"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trailing_configs: Dict[str, 'TrailingStopConfig'] = {}
    
    def calculate_trailing_stop(self, signal: Signal, 
                               current_price: float,
                               highest_price: float) -> float:
        """Calculate new trailing stop level"""
        try:
            if signal.side == 'buy':
                # Long position
                profit_pct = (highest_price - signal.entry_price) / signal.entry_price
                
                if profit_pct < 0.01:  # Less than 1% profit
                    # Keep original stop
                    return signal.stop_loss
                elif profit_pct < 0.03:  # 1-3% profit
                    # Move stop to breakeven
                    return signal.entry_price
                elif profit_pct < 0.05:  # 3-5% profit
                    # Trail by 2%
                    return highest_price * 0.98
                else:  # More than 5% profit
                    # Trail by 1.5%
                    return highest_price * 0.985
            
            else:  # Short position
                profit_pct = (signal.entry_price - current_price) / signal.entry_price
                lowest_price = current_price  # For shorts, we track lowest
                
                if profit_pct < 0.01:
                    return signal.stop_loss
                elif profit_pct < 0.03:
                    return signal.entry_price
                elif profit_pct < 0.05:
                    return lowest_price * 1.02
                else:
                    return lowest_price * 1.015
            
        except Exception as e:
            self.logger.error(f"Trailing stop calculation error: {e}")
            return signal.stop_loss

@dataclass
class TrailingStopConfig:
    """Configuration for trailing stops"""
    activation_profit: float = 0.01  # 1% profit to activate
    trail_distance: float = 0.02     # Trail by 2%
    step_profits: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.10])
    step_distances: List[float] = field(default_factory=lambda: [0.03, 0.02, 0.015, 0.01])

# ===========================
# EXPORTS
# ===========================

__all__ = [
    'EntryMonitoringService',
    'EntryCondition',
    'EntryTrigger',
    'AdvancedRiskManager',
    'RiskProfile',
    'PositionSizeResult',
    'TrailingStopManager',
    'TrailingStopConfig'
]