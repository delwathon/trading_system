"""
Signal Generator V13.0 - Core Architecture & State Management
==============================================================
Complete rewrite with professional architecture, state management,
and pending signal queue system.

PART 1: Core Architecture, State Management, and Base Classes
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
from collections import deque, defaultdict
import json
import hashlib
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import PriorityQueue
import redis
import pickle

# ===========================
# ENUMS AND CONSTANTS
# ===========================

class SignalStatus(Enum):
    """Signal lifecycle states"""
    ANALYZING = "analyzing"
    PENDING = "pending"
    READY = "ready"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"

class TimeFrame(Enum):
    """Standardized timeframes"""
    M1 = ("1m", 1)
    M5 = ("5m", 5)
    M15 = ("15m", 15)
    M30 = ("30m", 30)
    H1 = ("1h", 60)
    H2 = ("2h", 120)
    H4 = ("4h", 240)
    H6 = ("6h", 360)
    H8 = ("8h", 480)
    H12 = ("12h", 720)
    D1 = ("1d", 1440)
    W1 = ("1w", 10080)
    
    def __init__(self, label: str, minutes: int):
        self.label = label
        self.minutes = minutes
    
    @classmethod
    def from_string(cls, tf_string: str) -> 'TimeFrame':
        for tf in cls:
            if tf.label == tf_string:
                return tf
        raise ValueError(f"Unknown timeframe: {tf_string}")
    
    def __lt__(self, other):
        return self.minutes < other.minutes

class AnalysisStage(Enum):
    """Analysis pipeline stages"""
    MARKET_SCAN = auto()
    PRIMARY_ANALYSIS = auto()
    CONFIRMATION_CHECK = auto()
    ENTRY_OPTIMIZATION = auto()
    RISK_CALCULATION = auto()
    FINAL_VALIDATION = auto()
    QUEUE_PLACEMENT = auto()

class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_TREND_UP = "strong_trend_up"
    TREND_UP = "trend_up"
    RANGING_BULLISH = "ranging_bullish"
    RANGING = "ranging"
    RANGING_BEARISH = "ranging_bearish"
    TREND_DOWN = "trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    VOLATILE = "volatile"
    SQUEEZE = "squeeze"

class SignalQuality(Enum):
    """Signal quality tiers"""
    ELITE = "elite"        # Top 1% signals
    PREMIUM = "premium"    # Top 5% signals
    STANDARD = "standard"  # Good signals
    MARGINAL = "marginal"  # Acceptable signals
    REJECTED = "rejected"  # Failed validation

# ===========================
# CONFIGURATION DATACLASSES
# ===========================

@dataclass
class SystemConfiguration:
    """Main system configuration"""
    # Timeframe Configuration (Top-Down Approach)
    primary_timeframe: TimeFrame = TimeFrame.H6
    confirmation_timeframes: List[TimeFrame] = field(default_factory=lambda: [TimeFrame.H4, TimeFrame.H1])
    entry_timeframe: TimeFrame = TimeFrame.H1
    
    # Signal Generation
    max_signals_per_symbol: int = 3
    max_pending_signals: int = 50
    signal_expiry_minutes: int = 120
    
    # Risk Management
    max_concurrent_positions: int = 10
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_daily_risk: float = 0.06      # 6% daily
    min_risk_reward: float = 1.5
    max_risk_reward: float = 10.0
    
    # Quality Thresholds
    min_confidence_elite: float = 80.0
    min_confidence_premium: float = 70.0
    min_confidence_standard: float = 60.0
    min_confidence_marginal: float = 50.0
    
    # Analysis Parameters
    use_ml_prediction: bool = True
    use_news_sentiment: bool = True
    use_volume_profile: bool = True
    use_ichimoku: bool = True
    use_order_flow: bool = True
    
    # Performance Tracking
    track_performance: bool = True
    performance_window_days: int = 30
    
    # System Resources
    max_workers: int = 12
    use_redis_cache: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # API Limits
    api_calls_per_minute: int = 120
    batch_size: int = 10

@dataclass
class SignalCriteria:
    """Balanced signal validation criteria"""
    # RSI Ranges (RELAXED)
    primary_rsi_long_range: Tuple[float, float] = (20, 55)
    primary_rsi_short_range: Tuple[float, float] = (45, 80)
    confirm_rsi_long_max: float = 60
    confirm_rsi_short_min: float = 40
    entry_rsi_long_range: Tuple[float, float] = (15, 45)
    entry_rsi_short_range: Tuple[float, float] = (55, 85)
    
    # Stochastic Settings (RELAXED)
    primary_stoch_long_max: float = 50
    primary_stoch_short_min: float = 50
    entry_stoch_long_max: float = 40
    entry_stoch_short_min: float = 60
    
    # Volume Requirements (RELAXED)
    min_volume_ratio: float = 1.0
    optimal_volume_ratio: float = 1.3
    max_volume_spike: float = 8.0
    
    # Price Action (RELAXED)
    max_recent_move_pct: float = 0.12
    max_parabolic_move_pct: float = 0.18
    
    # Volatility (RELAXED)
    min_atr_pct: float = 0.003
    max_atr_pct: float = 0.15
    
    # Market Structure
    respect_support_resistance: bool = True
    min_distance_from_level: float = 0.001
    
    # Multi-Timeframe (MAJOR CHANGE)
    require_all_timeframe_alignment: bool = True
    min_timeframe_agreement: float = 1.0

# ===========================
# STATE MANAGEMENT SYSTEM
# ===========================

class StateManager:
    """Centralized state management with persistence"""
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # State containers
        self._signals: Dict[str, 'Signal'] = {}
        self._pending_queue: PriorityQueue = PriorityQueue()
        self._monitoring_pool: Dict[str, 'Signal'] = {}
        self._executed_signals: deque = deque(maxlen=1000)
        self._performance_metrics: Dict[str, Any] = defaultdict(dict)
        
        # Symbol tracking
        self._symbol_states: Dict[str, 'SymbolState'] = {}
        self._active_positions: Dict[str, List['Signal']] = defaultdict(list)
        
        # System state
        self._is_running: bool = False
        self._last_update: datetime = datetime.now(timezone.utc)
        self._error_count: int = 0
        self._state_lock = threading.RLock()
        
        # Initialize Redis if configured
        self._redis_client = None
        if config.use_redis_cache:
            try:
                self._redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port,
                    db=config.redis_db,
                    decode_responses=False
                )
                self._redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self._redis_client = None
    
    def add_signal(self, signal: 'Signal') -> bool:
        """Add a new signal to state"""
        with self._state_lock:
            try:
                # Check limits
                symbol_signals = [s for s in self._signals.values() 
                                 if s.symbol == signal.symbol and 
                                 s.status in [SignalStatus.PENDING, SignalStatus.MONITORING]]
                
                if len(symbol_signals) >= self.config.max_signals_per_symbol:
                    self.logger.warning(f"Max signals reached for {signal.symbol}")
                    return False
                
                # Add to state
                self._signals[signal.id] = signal
                
                # Add to appropriate queue
                if signal.status == SignalStatus.PENDING:
                    priority = self._calculate_priority(signal)
                    self._pending_queue.put((-priority, signal.id))
                elif signal.status == SignalStatus.MONITORING:
                    self._monitoring_pool[signal.id] = signal
                
                # Update symbol state
                self._update_symbol_state(signal.symbol)
                
                # Persist to Redis
                self._persist_signal(signal)
                
                self.logger.info(f"Signal {signal.id} added to state")
                return True
                
            except Exception as e:
                self.logger.error(f"Error adding signal: {e}")
                return False
    
    def update_signal(self, signal_id: str, updates: Dict[str, Any]) -> bool:
        """Update signal state"""
        with self._state_lock:
            if signal_id not in self._signals:
                return False
            
            signal = self._signals[signal_id]
            old_status = signal.status
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(signal, key):
                    setattr(signal, key, value)
            
            # Handle status transitions
            if 'status' in updates and updates['status'] != old_status:
                self._handle_status_transition(signal, old_status, updates['status'])
            
            # Update timestamp
            signal.last_updated = datetime.now(timezone.utc)
            
            # Persist changes
            self._persist_signal(signal)
            
            return True
    
    def get_pending_signals(self, limit: int = 10) -> List['Signal']:
        """Get top pending signals"""
        with self._state_lock:
            signals = []
            temp_queue = []
            
            while not self._pending_queue.empty() and len(signals) < limit:
                priority, signal_id = self._pending_queue.get()
                temp_queue.append((priority, signal_id))
                
                if signal_id in self._signals:
                    signal = self._signals[signal_id]
                    if signal.status == SignalStatus.PENDING:
                        signals.append(signal)
            
            # Restore queue
            for item in temp_queue:
                self._pending_queue.put(item)
            
            return signals
    
    def get_monitoring_signals(self) -> List['Signal']:
        """Get signals being monitored for entry"""
        with self._state_lock:
            return list(self._monitoring_pool.values())
    
    def get_active_positions(self, symbol: Optional[str] = None) -> List['Signal']:
        """Get active positions"""
        with self._state_lock:
            if symbol:
                return self._active_positions.get(symbol, [])
            else:
                all_positions = []
                for positions in self._active_positions.values():
                    all_positions.extend(positions)
                return all_positions
    
    def cleanup_expired_signals(self) -> int:
        """Remove expired signals"""
        with self._state_lock:
            now = datetime.now(timezone.utc)
            expired_count = 0
            
            for signal_id in list(self._signals.keys()):
                signal = self._signals[signal_id]
                
                # Check expiry
                if signal.status in [SignalStatus.PENDING, SignalStatus.MONITORING]:
                    age_minutes = (now - signal.created_at).total_seconds() / 60
                    if age_minutes > self.config.signal_expiry_minutes:
                        signal.status = SignalStatus.EXPIRED
                        self._handle_status_transition(signal, signal.status, SignalStatus.EXPIRED)
                        expired_count += 1
            
            return expired_count
    
    def get_performance_metrics(self, symbol: Optional[str] = None) -> Dict:
        """Get performance metrics"""
        with self._state_lock:
            if symbol:
                return self._performance_metrics.get(symbol, {})
            return dict(self._performance_metrics)
    
    def _calculate_priority(self, signal: 'Signal') -> float:
        """Calculate signal priority for queue"""
        priority = 0.0
        
        # Quality tier contribution (0-1000)
        quality_scores = {
            SignalQuality.ELITE: 1000,
            SignalQuality.PREMIUM: 750,
            SignalQuality.STANDARD: 500,
            SignalQuality.MARGINAL: 250,
            SignalQuality.REJECTED: 0
        }
        priority += quality_scores.get(signal.quality_tier, 0)
        
        # Confidence contribution (0-100)
        priority += signal.confidence
        
        # Risk/Reward contribution (0-300)
        priority += min(signal.risk_reward_ratio * 100, 300)
        
        # Timeframe bonus (higher TF = higher priority)
        tf_bonus = {
            TimeFrame.D1: 200,
            TimeFrame.H12: 150,
            TimeFrame.H6: 100,
            TimeFrame.H4: 75,
            TimeFrame.H2: 50,
            TimeFrame.H1: 25
        }
        priority += tf_bonus.get(signal.analysis_timeframe, 0)
        
        # Recency penalty (older signals get lower priority)
        age_minutes = (datetime.now(timezone.utc) - signal.created_at).total_seconds() / 60
        priority -= age_minutes * 0.5
        
        return max(0, priority)
    
    def _handle_status_transition(self, signal: 'Signal', old_status: SignalStatus, new_status: SignalStatus):
        """Handle signal status transitions"""
        # Remove from old containers
        if old_status == SignalStatus.MONITORING:
            self._monitoring_pool.pop(signal.id, None)
        
        # Add to new containers
        if new_status == SignalStatus.MONITORING:
            self._monitoring_pool[signal.id] = signal
        elif new_status == SignalStatus.EXECUTED:
            self._executed_signals.append(signal)
            self._active_positions[signal.symbol].append(signal)
        elif new_status in [SignalStatus.CANCELLED, SignalStatus.EXPIRED, SignalStatus.FAILED]:
            # Archive the signal
            if self._redis_client:
                self._redis_client.hset(
                    f"archived_signals:{signal.symbol}",
                    signal.id,
                    pickle.dumps(signal)
                )
    
    def _update_symbol_state(self, symbol: str):
        """Update symbol-specific state"""
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = SymbolState(symbol)
        
        state = self._symbol_states[symbol]
        state.last_analysis = datetime.now(timezone.utc)
        state.pending_signals = len([s for s in self._signals.values() 
                                    if s.symbol == symbol and s.status == SignalStatus.PENDING])
        state.active_signals = len([s for s in self._signals.values() 
                                   if s.symbol == symbol and s.status in [SignalStatus.MONITORING, SignalStatus.EXECUTED]])
    
    def _persist_signal(self, signal: 'Signal'):
        """Persist signal to Redis"""
        if self._redis_client:
            try:
                key = f"signal:{signal.id}"
                value = pickle.dumps(signal)
                self._redis_client.setex(key, 3600, value)  # 1 hour TTL
            except Exception as e:
                self.logger.error(f"Redis persistence error: {e}")

# ===========================
# CORE DATA MODELS
# ===========================

@dataclass
class Signal:
    """Enhanced Signal with all required fields for display and trading"""
    
    # ============ REQUIRED FIELDS (NO DEFAULTS) ============
    # Core identification
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    status: SignalStatus
    quality_tier: SignalQuality
    
    # Timestamps (required)
    created_at: datetime
    last_updated: datetime
    
    # Price levels (required fields)
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    current_price: float
    
    # Risk metrics (required fields)
    risk_reward_ratio: float
    position_size: float
    risk_amount: float
    potential_profit: float
    
    # Analysis data (required fields)
    confidence: float
    original_confidence: float  # Before MTF boost
    analysis_timeframe: TimeFrame
    market_regime: MarketRegime
    
    # Market data (required)
    volume_24h: float
    price_change_24h: float
    volume_ratio: float
    
    # ============ OPTIONAL FIELDS (WITH DEFAULTS) ============
    # Optional timestamps
    triggered_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    
    # Order and execution fields with defaults
    mtf_boost: float = 0.0
    order_type: str = 'limit'  # 'market' or 'limit'
    quality_grade: str = 'C'  # A+, A, A-, B+, B, B-, C+, C
    entry_strategy: str = 'patient'  # 'immediate', 'patient', 'scaled'
    
    # MTF status fields with defaults
    mtf_status: str = 'NONE'  # 'STRONG', 'PARTIAL', 'NONE', 'DISABLED'
    mtf_validated: bool = False
    mtf_confirmation_count: int = 0
    mtf_total_timeframes: int = 0
    
    # Multi-timeframe validation with defaults
    mtf_alignment: Dict[str, bool] = field(default_factory=dict)
    mtf_scores: Dict[str, float] = field(default_factory=dict)
    mtf_analysis: Dict[str, List[str]] = field(default_factory=dict)
    
    # Technical indicators
    indicators: Dict[str, Any] = field(default_factory=dict)
    
    # ML and sentiment (optional)
    ml_prediction: Optional[Dict] = None
    news_sentiment: Optional[Dict] = None
    
    # Volume profile (optional)
    volume_analysis: Optional[Dict] = None
    
    # Pattern and structure with defaults
    patterns: List[Dict] = field(default_factory=list)
    breakout_detected: bool = False
    support_resistance: Dict[str, float] = field(default_factory=dict)
    
    # Entry conditions with defaults
    entry_conditions: List[Dict] = field(default_factory=list)
    entry_checklist: Dict[str, bool] = field(default_factory=dict)
    
    # Risk warnings with defaults
    warnings: List[str] = field(default_factory=list)
    
    # Metadata with defaults
    metadata: Dict[str, Any] = field(default_factory=dict)
    regime_compatibility: str = 'medium'  # 'high', 'medium', 'low'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with all fields"""
        data = asdict(self)
        # Convert enums
        data['status'] = self.status.value if hasattr(self.status, 'value') else self.status
        data['quality_tier'] = self.quality_tier.value if hasattr(self.quality_tier, 'value') else self.quality_tier
        data['market_regime'] = self.market_regime.value if hasattr(self.market_regime, 'value') else self.market_regime
        data['analysis_timeframe'] = self.analysis_timeframe.label if hasattr(self.analysis_timeframe, 'label') else str(self.analysis_timeframe)
        
        # Convert datetimes
        for key in ['created_at', 'last_updated', 'triggered_at', 'executed_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Signal':
        """Create from dictionary"""
        # Convert enums back
        data['status'] = SignalStatus(data['status'])
        data['quality_tier'] = SignalQuality(data['quality_tier'])
        data['market_regime'] = MarketRegime(data['market_regime'])
        data['analysis_timeframe'] = TimeFrame.from_string(data['analysis_timeframe'])
        # Convert datetimes
        for key in ['created_at', 'last_updated', 'triggered_at', 'executed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)
    
@dataclass
class SymbolState:
    """Track per-symbol state"""
    symbol: str
    last_analysis: Optional[datetime] = None
    pending_signals: int = 0
    active_signals: int = 0
    total_generated: int = 0
    total_executed: int = 0
    win_rate: float = 0.0
    avg_rr_achieved: float = 0.0
    last_signal_id: Optional[str] = None
    is_blacklisted: bool = False
    blacklist_reason: Optional[str] = None

@dataclass
class MarketContext:
    """Market-wide context and conditions"""
    timestamp: datetime
    overall_sentiment: str  # 'bullish', 'bearish', 'neutral'
    volatility_index: float
    dominant_trend: str
    risk_on_off: str  # 'risk_on', 'risk_off', 'neutral'
    major_events: List[Dict] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)
    
# ===========================
# ABSTRACT BASE CLASSES
# ===========================

class AnalysisModule(ABC):
    """Base class for all analysis modules"""
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def analyze(self, data: pd.DataFrame, context: Dict) -> Dict:
        """Perform analysis on data"""
        pass
    
    @abstractmethod
    def get_signals(self) -> List[Dict]:
        """Extract signals from analysis"""
        pass
    
    @abstractmethod
    def validate(self, signal: Dict) -> Tuple[bool, List[str]]:
        """Validate a signal"""
        pass

class EntryMonitor(ABC):
    """Base class for entry monitoring"""
    
    @abstractmethod
    async def check_entry_conditions(self, signal: Signal, current_data: Dict) -> Tuple[bool, Dict]:
        """Check if entry conditions are met"""
        pass
    
    @abstractmethod
    async def update_signal_prices(self, signal: Signal, current_data: Dict) -> Signal:
        """Update signal with current market prices"""
        pass

class RiskManager(ABC):
    """Base class for risk management"""
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Calculate appropriate position size"""
        pass
    
    @abstractmethod
    def validate_risk_limits(self, signal: Signal, current_positions: List[Signal]) -> Tuple[bool, str]:
        """Check if signal passes risk limits"""
        pass
    
    @abstractmethod
    def adjust_stops_and_targets(self, signal: Signal, market_conditions: Dict) -> Signal:
        """Adjust stops and targets based on conditions"""
        pass

# ===========================
# SIGNAL LIFECYCLE MANAGER
# ===========================

class SignalLifecycleManager:
    """Manages the complete lifecycle of signals"""
    
    def __init__(self, state_manager: StateManager, config: SystemConfiguration):
        self.state_manager = state_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lifecycle_tasks: Dict[str, asyncio.Task] = {}
    
    async def process_new_signal(self, signal_data: Dict) -> Optional[Signal]:
        """Process a new signal through the lifecycle"""
        try:
            # Create signal object
            signal = self._create_signal(signal_data)
            
            # Initial validation
            if not self._validate_signal(signal):
                return None
            
            # Add to state
            if not self.state_manager.add_signal(signal):
                return None
            
            # Start lifecycle monitoring
            task = asyncio.create_task(self._monitor_signal_lifecycle(signal))
            self._lifecycle_tasks[signal.id] = task
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return None
    
    async def _monitor_signal_lifecycle(self, signal: Signal):
        """Monitor signal through its lifecycle"""
        try:
            while signal.status not in [SignalStatus.EXECUTED, SignalStatus.CANCELLED, 
                                       SignalStatus.EXPIRED, SignalStatus.FAILED]:
                
                # Check expiry
                age = (datetime.now(timezone.utc) - signal.created_at).total_seconds() / 60
                if age > self.config.signal_expiry_minutes:
                    self.state_manager.update_signal(signal.id, {'status': SignalStatus.EXPIRED})
                    break
                
                # Process based on current status
                if signal.status == SignalStatus.PENDING:
                    # Wait for better entry conditions
                    await asyncio.sleep(30)
                    
                elif signal.status == SignalStatus.READY:
                    # Transition to monitoring
                    self.state_manager.update_signal(signal.id, {'status': SignalStatus.MONITORING})
                    
                elif signal.status == SignalStatus.MONITORING:
                    # Check entry conditions
                    await asyncio.sleep(10)
                    
                await asyncio.sleep(5)
            
        except Exception as e:
            self.logger.error(f"Lifecycle monitoring error for {signal.id}: {e}")
            self.state_manager.update_signal(signal.id, {'status': SignalStatus.FAILED})
        
        finally:
            # Cleanup
            self._lifecycle_tasks.pop(signal.id, None)
    
    def _create_signal(self, data: Dict) -> Signal:
        """Create a Signal object from raw data"""
        return Signal(
            id=self._generate_signal_id(data),
            symbol=data['symbol'],
            side=data['side'],
            status=SignalStatus.ANALYZING,
            quality_tier=SignalQuality.STANDARD,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            take_profit_1=data['take_profit_1'],
            take_profit_2=data['take_profit_2'],
            current_price=data['current_price'],
            risk_reward_ratio=data.get('risk_reward_ratio', 2.0),
            position_size=0.0,
            risk_amount=0.0,
            potential_profit=0.0,
            confidence=data.get('confidence', 50.0),
            analysis_timeframe=TimeFrame.from_string(data.get('timeframe', '6h')),
            market_regime=MarketRegime(data.get('market_regime', 'ranging')),
            indicators=data.get('indicators', {}),
            ml_prediction=data.get('ml_prediction'),
            news_sentiment=data.get('news_sentiment'),
            volume_analysis=data.get('volume_analysis'),
            warnings=data.get('warnings', [])
        )
    
    def _generate_signal_id(self, data: Dict) -> str:
        """Generate unique signal ID"""
        content = f"{data['symbol']}_{data['side']}_{data['entry_price']}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _validate_signal(self, signal: Signal) -> bool:
        """Much simpler signal validation"""
        # Only check the most basic requirements
        if signal.risk_reward_ratio < 1.2:  # More lenient than config
            self.logger.warning(f"Signal {signal.id} failed R/R validation")
            return False
        
        if signal.confidence < 45:  # More lenient than config
            self.logger.warning(f"Signal {signal.id} failed confidence validation")  
            return False

        return True

# ===========================
# SIGNAL QUEUE MANAGER
# ===========================

class SignalQueueManager:
    """Manages the pending signal queue with prioritization"""
    
    def __init__(self, state_manager: StateManager, config: SystemConfiguration):
        self.state_manager = state_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._processing_lock = asyncio.Lock()
    
    async def process_queue(self):
        """Process pending signals in priority order"""
        async with self._processing_lock:
            pending_signals = self.state_manager.get_pending_signals(limit=10)
            
            for signal in pending_signals:
                try:
                    # Check if ready for monitoring
                    if await self._is_ready_for_monitoring(signal):
                        self.state_manager.update_signal(
                            signal.id, 
                            {'status': SignalStatus.READY}
                        )
                        self.logger.info(f"Signal {signal.id} ready for monitoring")
                    
                except Exception as e:
                    self.logger.error(f"Error processing queued signal {signal.id}: {e}")
    
    async def _is_ready_for_monitoring(self, signal: Signal) -> bool:
        """Check if signal is ready to move to monitoring phase"""
        # Check market conditions
        # Check if entry zone is approaching
        # Check if other signals for symbol are active
        # This will be implemented with actual market data checks
        return True

# ===========================
# PERFORMANCE TRACKER
# ===========================

class PerformanceTracker:
    """Track and analyze system performance"""
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def update_signal_outcome(self, signal_id: str, outcome: Dict):
        """Update signal with execution outcome"""
        # Record win/loss
        # Update symbol statistics
        # Calculate actual R/R achieved
        # Update win rate
        pass
    
    def get_performance_report(self, period_days: int = 7) -> Dict:
        """Generate performance report"""
        metrics = self.state_manager.get_performance_metrics()
        
        return {
            'total_signals': len(metrics),
            'win_rate': self._calculate_win_rate(metrics),
            'avg_rr_achieved': self._calculate_avg_rr(metrics),
            'best_performing_symbol': self._get_best_symbol(metrics),
            'worst_performing_symbol': self._get_worst_symbol(metrics)
        }
    
    def _calculate_win_rate(self, metrics: Dict) -> float:
        """Calculate overall win rate"""
        total_wins = sum(1 for m in metrics.values() if m.get('result') == 'win')
        total_trades = len(metrics)
        return (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    
    def _calculate_avg_rr(self, metrics: Dict) -> float:
        """Calculate average R/R achieved"""
        rr_values = [m.get('rr_achieved', 0) for m in metrics.values()]
        return sum(rr_values) / len(rr_values) if rr_values else 0.0
    
    def _get_best_symbol(self, metrics: Dict) -> str:
        """Get best performing symbol"""
        # Implementation to find best symbol
        return "N/A"
    
    def _get_worst_symbol(self, metrics: Dict) -> str:
        """Get worst performing symbol"""
        # Implementation to find worst symbol
        return "N/A"

# ===========================
# ERROR HANDLER
# ===========================

class SystemErrorHandler:
    """Centralized error handling and recovery"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_counts = defaultdict(int)
        self.error_threshold = 10
    
    def handle_error(self, error: Exception, context: Dict) -> bool:
        """Handle system errors with recovery logic"""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        
        self.logger.error(f"Error in {context.get('module', 'unknown')}: {error}")
        
        # Check if error threshold exceeded
        if self.error_counts[error_type] > self.error_threshold:
            self.logger.critical(f"Error threshold exceeded for {error_type}")
            return False  # System should stop
        
        # Attempt recovery based on error type
        if isinstance(error, ConnectionError):
            self.logger.info("Attempting connection recovery...")
            # Implement reconnection logic
        
        return True  # Continue operation

# ===========================
# MAIN EXPORTS
# ===========================

__all__ = [
    # Enums
    'SignalStatus',
    'TimeFrame', 
    'AnalysisStage',
    'MarketRegime',
    'SignalQuality',
    
    # Configuration
    'SystemConfiguration',
    'SignalCriteria',
    
    # State Management
    'StateManager',
    
    # Data Models
    'Signal',
    'SymbolState',
    'MarketContext',
    
    # Abstract Base Classes
    'AnalysisModule',
    'EntryMonitor',
    'RiskManager',
    
    # Managers
    'SignalLifecycleManager',
    'SignalQueueManager',
    'PerformanceTracker',
    'SystemErrorHandler'
]

# Version and metadata
__version__ = "13.0.0"
__status__ = "DEVELOPMENT"
__description__ = "Professional Signal Generation System - Core Architecture"