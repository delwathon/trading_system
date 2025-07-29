"""
Database models for the Enhanced Bybit Trading System.
SQLAlchemy models for storing trading signals, market data, system configuration, and auto-trading data.
UPDATED: Added encryption_password field for database-stored encryption key.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import logging

Base = declarative_base()


class SystemConfig(Base):
    """System configuration stored in database - UPDATED with encryption password"""
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    config_name = Column(String(100), unique=True, nullable=False, default='default')
    
    # API Configuration
    api_key = Column(String(255), nullable=True)
    api_secret = Column(String(255), nullable=True)
    demo_api_key = Column(String(255), nullable=True)
    demo_api_secret = Column(String(255), nullable=True)
    sandbox_mode = Column(Boolean, default=False)
    
    # NEW: Encryption password for API secrets
    encryption_password = Column(String(255), default='bybit_trading_system_secure_key_2024')
    
    # Market Scanning
    min_volume_24h = Column(Float, default=5_000_000)
    max_symbols_scan = Column(Integer, default=100)
    timeframe = Column(String(10), default='30m')
    
    # Multi-Timeframe Configuration
    confirmation_timeframes = Column(JSON, default=['1h', '4h'])
    mtf_confirmation_required = Column(Boolean, default=True)
    mtf_weight_multiplier = Column(Float, default=1.5)
    
    # Rate Limiting
    max_requests_per_second = Column(Float, default=8.0)
    api_timeout = Column(Integer, default=20000)
    
    # Risk Management
    max_portfolio_risk = Column(Float, default=0.02)
    max_daily_trades = Column(Integer, default=20)
    max_single_position_risk = Column(Float, default=0.005)
    
    # Threading
    max_workers = Column(Integer, nullable=True)
    
    # Caching
    cache_ttl_seconds = Column(Integer, default=60)
    max_cache_size = Column(Integer, default=1000)
    
    # ML Model
    ml_training_samples = Column(Integer, default=400)
    ml_profitable_rate = Column(Float, default=0.45)
    
    # Chart Settings
    generate_chart = Column(Boolean, default=True)
    show_charts = Column(Boolean, default=True)
    save_charts = Column(Boolean, default=False)
    charts_per_batch = Column(Integer, default=5)
    chart_width = Column(Integer, default=1400)
    chart_height = Column(Integer, default=800)
    
    # Indicator parameters
    stoch_rsi_window = Column(Integer, default=14)
    stoch_rsi_smooth_k = Column(Integer, default=3)
    stoch_rsi_smooth_d = Column(Integer, default=3)
    ichimoku_window1 = Column(Integer, default=9)
    ichimoku_window2 = Column(Integer, default=26)
    ichimoku_window3 = Column(Integer, default=52)
    
    # OHLCV Data Limits
    ohlcv_limit_primary = Column(Integer, default=500)
    ohlcv_limit_mtf = Column(Integer, default=200)
    ohlcv_limit_analysis = Column(Integer, default=500)
    
    # Telegram & Trading Configuration
    telegram_id = Column(String(50), default='6708641837')
    telegram_bot_token = Column(String(255), default='8088506547:AAHZZxiY_wlh48IN4ldPNRJtM9qi7qxfLdM')
    
    # FIXED: Leverage - Single string value instead of JSON array
    leverage = Column(String(10), default='max')  # Must be one of: '10', '12.5', '25', '50', 'max'
    risk_amount = Column(Float, default=5.0)  # Risk percentage of account balance (e.g., 5.0 = 5%)
    
    # NEW: Auto-Trading Configuration
    max_concurrent_positions = Column(Integer, default=10)
    max_execution_per_trade = Column(Integer, default=3)
    day_trade_start_hour = Column(String(10), default='01:00')
    scan_interval = Column(Integer, default=3600)  # 3 hours in seconds
    auto_execute_trades = Column(Boolean, default=True)  # Enable auto-trading execution
    auto_close_enabled = Column(Boolean, default=True)  # Enable auto-close feature
    auto_close_profit_at = Column(Float, default=75.0)  # 20% profit target
    auto_close_loss_at = Column(Float, default=100.0)  # 100% profit target
    default_tp_level = Column(String(10), default='take_profit_1')  # Default take profit level to use
    use_old_analysis = Column(Boolean, default=True)  # Use old analysis method
    monitor_mode = Column(Boolean, default=False)  # Monitor mode only, no analysis
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self):
        """Convert to dictionary for compatibility with existing code"""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'demo_api_key': self.demo_api_key,
            'demo_api_secret': self.demo_api_secret,
            'sandbox_mode': self.sandbox_mode,
            'encryption_password': self.encryption_password,  # NEW
            'min_volume_24h': self.min_volume_24h,
            'max_symbols_scan': self.max_symbols_scan,
            'timeframe': self.timeframe,
            'confirmation_timeframes': self.confirmation_timeframes or ['1h', '4h', '6h'],
            'mtf_confirmation_required': self.mtf_confirmation_required,
            'mtf_weight_multiplier': self.mtf_weight_multiplier,
            'max_requests_per_second': self.max_requests_per_second,
            'api_timeout': self.api_timeout,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_daily_trades': self.max_daily_trades,
            'max_single_position_risk': self.max_single_position_risk,
            'max_workers': self.max_workers,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'max_cache_size': self.max_cache_size,
            'ml_training_samples': self.ml_training_samples,
            'ml_profitable_rate': self.ml_profitable_rate,
            'generate_chart': self.generate_chart,
            'show_charts': self.show_charts,
            'save_charts': self.save_charts,
            'charts_per_batch': self.charts_per_batch,
            'chart_width': self.chart_width,
            'chart_height': self.chart_height,
            'stoch_rsi_window': self.stoch_rsi_window,
            'stoch_rsi_smooth_k': self.stoch_rsi_smooth_k,
            'stoch_rsi_smooth_d': self.stoch_rsi_smooth_d,
            'ichimoku_window1': self.ichimoku_window1,
            'ichimoku_window2': self.ichimoku_window2,
            'ichimoku_window3': self.ichimoku_window3,
            'ohlcv_limit_primary': self.ohlcv_limit_primary,
            'ohlcv_limit_mtf': self.ohlcv_limit_mtf,
            'ohlcv_limit_analysis': self.ohlcv_limit_analysis,
            'telegram_id': self.telegram_id,
            'telegram_bot_token': self.telegram_bot_token,
            # FIXED: Return single leverage value
            'leverage': self.leverage,
            'risk_amount': self.risk_amount,  # Risk percentage of account balance
            # NEW: Auto-trading fields
            'max_concurrent_positions': self.max_concurrent_positions,
            'max_execution_per_trade': self.max_execution_per_trade,
            'day_trade_start_hour': self.day_trade_start_hour,
            'scan_interval': self.scan_interval,
            'auto_execute_trades': self.auto_execute_trades,  # Enable auto-trading execution
            'auto_close_enabled': self.auto_close_enabled,  # Enable auto-close feature
            'auto_close_profit_at': self.auto_close_profit_at,
            'auto_close_loss_at': self.auto_close_loss_at,  # 100% profit target
            'default_tp_level': self.default_tp_level,  # Default take profit level to use
            'use_old_analysis': self.use_old_analysis,  # Use old analysis method
            'monitor_mode': self.monitor_mode,  # Monitor mode only, no analysis
        }


class TradingPosition(Base):
    """NEW: Track auto-trading positions"""
    __tablename__ = 'trading_positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_session_id = Column(Integer, ForeignKey('scan_sessions.id'), nullable=True)
    
    # Position identification
    position_id = Column(String(100), unique=True, nullable=False)  # Bybit position ID
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    
    # Position details
    entry_price = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    leverage = Column(String(10), nullable=False)
    risk_amount = Column(Float, nullable=False)  # Actual USDT amount used for this position
    risk_percentage = Column(Float, nullable=True)  # Original risk percentage (e.g., 5.0 for 5%)
    
    # Orders
    entry_order_id = Column(String(100), nullable=True)
    stop_loss_order_id = Column(String(100), nullable=True)
    take_profit_order_id = Column(String(100), nullable=True)
    
    # Targets
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    auto_close_profit_target = Column(Float, nullable=False)  # e.g., 10.0 for 10%
    
    # Status tracking
    status = Column(String(20), default='open')  # 'open', 'closed', 'error'
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_pct = Column(Float, default=0.0)
    leveraged_pnl_pct = Column(Float, default=0.0)  # PnL considering leverage
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    signal_confidence = Column(Float, nullable=True)
    mtf_status = Column(String(20), nullable=True)
    auto_closed = Column(Boolean, default=False)
    close_reason = Column(String(100), nullable=True)  # 'profit_target', 'stop_loss', 'manual'
    
    # Relationships
    scan_session = relationship("ScanSession", back_populates="positions")


class AutoTradingSession(Base):
    """NEW: Track auto-trading sessions"""
    __tablename__ = 'auto_trading_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), unique=True, nullable=False)
    
    # Session details
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String(20), default='active')  # 'active', 'stopped', 'error'
    
    # Configuration snapshot
    config_snapshot = Column(JSON)
    
    # Statistics
    total_scans = Column(Integer, default=0)
    total_trades_placed = Column(Integer, default=0)
    total_trades_closed = Column(Integer, default=0)
    total_profit_trades = Column(Integer, default=0)
    total_loss_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    
    # Performance
    success_rate = Column(Float, default=0.0)
    average_trade_duration_minutes = Column(Float, default=0.0)
    max_concurrent_positions_reached = Column(Integer, default=0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    total_risk_exposed = Column(Float, default=0.0)
    
    # Last scan info
    last_scan_at = Column(DateTime, nullable=True)
    next_scan_at = Column(DateTime, nullable=True)


class ScanSession(Base):
    """Scan session information - UPDATED with position relationship"""
    __tablename__ = 'scan_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_id = Column(String(50), unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    execution_time_seconds = Column(Float)
    symbols_analyzed = Column(Integer)
    signals_generated = Column(Integer)
    success_rate = Column(Float)
    charts_generated = Column(Integer)
    parallel_processing = Column(Boolean, default=False)
    threads_used = Column(Integer)
    mtf_enabled = Column(Boolean, default=False)
    primary_timeframe = Column(String(10))
    confirmation_timeframes = Column(JSON)
    mtf_weight_multiplier = Column(Float)
    
    # NEW: Auto-trading session link
    auto_trading_session_id = Column(Integer, ForeignKey('auto_trading_sessions.id'), nullable=True)
    trades_executed_count = Column(Integer, default=0)
    
    # Relationships
    signals = relationship("TradingSignal", back_populates="scan_session")
    opportunities = relationship("TradingOpportunity", back_populates="scan_session")
    market_summary = relationship("MarketSummary", back_populates="scan_session", uselist=False)
    positions = relationship("TradingPosition", back_populates="scan_session")  # NEW


class TradingSignal(Base):
    """Trading signals with comprehensive analysis data"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_session_id = Column(Integer, ForeignKey('scan_sessions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Basic signal data
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    order_type = Column(String(10), nullable=False)  # 'market' or 'limit'
    
    # Price levels
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float, nullable=True)
    
    # Confidence and MTF data
    confidence = Column(Float, nullable=False)
    original_confidence = Column(Float, nullable=False)
    mtf_boost = Column(Float, default=0.0)
    mtf_status = Column(String(20), default='UNKNOWN')  # 'STRONG', 'PARTIAL', 'NONE', 'DISABLED'
    mtf_confirmed_timeframes = Column(JSON, default=[])
    mtf_conflicting_timeframes = Column(JSON, default=[])
    mtf_confirmation_count = Column(Integer, default=0)
    mtf_confirmation_strength = Column(Float, default=0.0)
    
    # Risk and reward
    risk_reward_ratio = Column(Float)
    risk_level = Column(String(20))  # 'Low', 'Medium', 'High'
    total_risk_score = Column(Float)
    
    # Market data
    volume_24h = Column(Float)
    price_change_24h = Column(Float)
    
    # System data
    signal_type = Column(String(50))
    chart_file = Column(String(255))
    priority_boost = Column(Integer, default=0)
    
    # Technical analysis scores
    technical_score = Column(Float)
    volume_score = Column(Float)
    fibonacci_score = Column(Float)
    confluence_zones_count = Column(Integer)
    
    # NEW: Auto-trading fields
    selected_for_execution = Column(Boolean, default=False)
    execution_attempted = Column(Boolean, default=False)
    execution_successful = Column(Boolean, default=False)
    execution_error = Column(String(255), nullable=True)
    position_id = Column(String(100), nullable=True)  # Link to actual position
    
    # Additional analysis data (JSON)
    technical_summary = Column(JSON)
    volume_profile_data = Column(JSON)
    fibonacci_data = Column(JSON)
    confluence_zones = Column(JSON)
    entry_methods = Column(JSON)
    
    # Relationships
    scan_session = relationship("ScanSession", back_populates="signals")


class TradingOpportunity(Base):
    """Ranked trading opportunities"""
    __tablename__ = 'trading_opportunities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_session_id = Column(Integer, ForeignKey('scan_sessions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    rank = Column(Integer, nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    order_type = Column(String(10), nullable=False)
    
    # Confidence metrics
    confidence = Column(Float, nullable=False)
    original_confidence = Column(Float, nullable=False)
    mtf_boost = Column(Float, default=0.0)
    
    # Price levels
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float, nullable=True)
    
    # Risk and scoring
    risk_reward_ratio = Column(Float)
    volume_24h = Column(Float)
    total_score = Column(Float)
    
    # MTF analysis
    mtf_status = Column(String(20))
    mtf_confirmed = Column(JSON, default=[])
    mtf_conflicting = Column(JSON, default=[])
    mtf_confirmation_count = Column(Integer, default=0)
    mtf_total_timeframes = Column(Integer, default=0)
    mtf_confirmation_strength = Column(Float, default=0.0)
    priority_boost = Column(Integer, default=0)
    
    # Additional data
    risk_level = Column(String(20))
    chart_file = Column(String(255))
    signal_type = Column(String(50))
    distance_from_current = Column(Float)
    volume_score = Column(Float)
    technical_strength = Column(Float)
    
    # NEW: Auto-trading execution tracking
    selected_for_execution = Column(Boolean, default=False)
    execution_attempted = Column(Boolean, default=False)
    execution_successful = Column(Boolean, default=False)
    
    # Relationships
    scan_session = relationship("ScanSession", back_populates="opportunities")


class MarketSummary(Base):
    """Market summary for each scan session"""
    __tablename__ = 'market_summaries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_session_id = Column(Integer, ForeignKey('scan_sessions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Market metrics
    total_market_volume = Column(Float)
    average_volume = Column(Float)
    market_sentiment_bullish_pct = Column(Float)
    
    # Signal distribution
    buy_signals = Column(Integer, default=0)
    sell_signals = Column(Integer, default=0)
    market_orders = Column(Integer, default=0)
    limit_orders = Column(Integer, default=0)
    
    # Performance metrics
    signals_per_minute = Column(Float)
    avg_confidence = Column(Float)
    avg_original_confidence = Column(Float)
    mtf_boost_avg = Column(Float)
    speedup_factor = Column(Float)
    
    # MTF metrics
    mtf_strong_signals = Column(Integer, default=0)
    mtf_partial_signals = Column(Integer, default=0)
    mtf_none_signals = Column(Integer, default=0)
    
    # Top movers
    top_gainer_symbol = Column(String(50))
    top_gainer_change = Column(Float)
    top_loser_symbol = Column(String(50))
    top_loser_change = Column(Float)
    highest_volume_symbol = Column(String(50))
    
    # Additional market data (JSON)
    market_sentiment_data = Column(JSON)
    volume_distribution = Column(JSON)
    mtf_analysis_summary = Column(JSON)
    
    # Relationships
    scan_session = relationship("ScanSession", back_populates="market_summary")


class PerformanceMetric(Base):
    """System performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_session_id = Column(Integer, ForeignKey('scan_sessions.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    metric_type = Column(String(50), nullable=False)  # 'system', 'trading', 'technical'
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20))
    timeframe = Column(String(20))
    symbol = Column(String(50))
    additional_data = Column(JSON)


class SystemLog(Base):
    """System logs and errors"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20), nullable=False)  # 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    logger_name = Column(String(100))
    message = Column(Text, nullable=False)
    scan_session_id = Column(Integer, ForeignKey('scan_sessions.id'), nullable=True)
    symbol = Column(String(50), nullable=True)
    function_name = Column(String(100), nullable=True)
    error_type = Column(String(100), nullable=True)
    stack_trace = Column(Text, nullable=True)


class DatabaseManager:
    """Database connection and session management for MySQL"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        # MySQL-specific engine configuration
        self.engine = create_engine(
            database_url, 
            echo=False,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
            connect_args={
                "charset": "utf8mb4",
                "collation": "utf8mb4_unicode_ci"
            }
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.logger = logging.getLogger(__name__)
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.debug("✅ MySQL database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating MySQL database tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session"""
        try:
            session.close()
        except Exception as e:
            self.logger.warning(f"Error closing session: {e}")
    
    def test_connection(self) -> bool:
        """Test MySQL database connection"""
        try:
            session = self.get_session()
            session.execute(text("SELECT 1"))
            session.close()
            self.logger.debug("✅ MySQL database connection successful")
            return True
        except Exception as e:
            self.logger.error(f"MySQL database connection failed: {e}")
            return False