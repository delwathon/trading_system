"""
Database models for the Enhanced Bybit Trading System.
SQLAlchemy models for storing trading signals, market data, and system configuration.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import logging

Base = declarative_base()


class SystemConfig(Base):
    """System configuration stored in database"""
    __tablename__ = 'system_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    config_name = Column(String(100), unique=True, nullable=False, default='default')
    
    # API Configuration
    api_key = Column(String(255), nullable=True)
    api_secret = Column(String(255), nullable=True)
    demo_api_key = Column(String(255), nullable=True)
    demo_api_secret = Column(String(255), nullable=True)
    sandbox_mode = Column(Boolean, default=False)
    
    # Market Scanning
    min_volume_24h = Column(Float, default=5_000_000)
    max_symbols_scan = Column(Integer, default=100)
    timeframe = Column(String(10), default='15m')
    
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
    
    # CSV Output
    csv_base_filename = Column(String(255), default="enhanced_bybit_signals")
    
    # Chart Settings
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
    
    # ===== NEW: OHLCV Data Limits =====
    ohlcv_limit_primary = Column(Integer, default=500)     # Primary timeframe data limit
    ohlcv_limit_mtf = Column(Integer, default=200)         # MTF confirmation data limit
    ohlcv_limit_analysis = Column(Integer, default=500)    # General analysis data limit
    
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
            'min_volume_24h': self.min_volume_24h,
            'max_symbols_scan': self.max_symbols_scan,
            'timeframe': self.timeframe,
            'confirmation_timeframes': self.confirmation_timeframes or ['1h', '4h'],
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
            'csv_base_filename': self.csv_base_filename,
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
            # ===== NEW: Add OHLCV limits to dictionary =====
            'ohlcv_limit_primary': self.ohlcv_limit_primary,
            'ohlcv_limit_mtf': self.ohlcv_limit_mtf,
            'ohlcv_limit_analysis': self.ohlcv_limit_analysis
        }

class ScanSession(Base):
    """Scan session information"""
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
    
    # Relationships
    signals = relationship("TradingSignal", back_populates="scan_session")
    opportunities = relationship("TradingOpportunity", back_populates="scan_session")
    market_summary = relationship("MarketSummary", back_populates="scan_session", uselist=False)


class TradingSignal(Base):
    """Trading signals with comprehensive analysis data"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_session_id = Column(Integer, ForeignKey('scan_sessions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Basic signal data
    symbol = Column(String(20), nullable=False)
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
    symbol = Column(String(20), nullable=False)
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
    top_gainer_symbol = Column(String(20))
    top_gainer_change = Column(Float)
    top_loser_symbol = Column(String(20))
    top_loser_change = Column(Float)
    highest_volume_symbol = Column(String(20))
    
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
    symbol = Column(String(20))
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
    symbol = Column(String(20), nullable=True)
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