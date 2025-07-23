"""
Database-based configuration management for the Enhanced Bybit Trading System.
Replaces YAML-based configuration with database storage.
Primary timeframe: 30m, Confirmation timeframes: 1h, 4h, 6h
UPDATED: Fixed leverage storage and added auto-trading configuration.
UPDATED: Added connection pool optimization settings.
COMPLETE: Includes all functionality from project knowledge.
"""

import os
import yaml
import logging
import psutil
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime


@dataclass
class DatabaseConfig:
    """Database connection configuration - MySQL only - Loaded from YAML"""
    # Database connection
    database_type: str = "mysql"  # mysql only
    host: str = "localhost"
    port: int = 3306
    database: str = "bybit_trading_system"
    username: str = "root"
    password: str = ""
    
    # Connection options
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo_sql: bool = False
    
    def get_database_url(self) -> str:
        """Get SQLAlchemy database URL for MySQL"""
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> 'DatabaseConfig':
        """Load database configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract database configuration
            db_config_data = data.get('database', {})
            
            # Create DatabaseConfig with YAML values
            return cls(
                database_type=db_config_data.get('type', 'mysql'),
                host=db_config_data.get('host', 'localhost'),
                port=db_config_data.get('port', 3306),
                database=db_config_data.get('database', 'bybit_trading_system'),
                username=db_config_data.get('username', 'root'),
                password=db_config_data.get('password', ''),
                pool_size=db_config_data.get('pool_size', 10),
                max_overflow=db_config_data.get('max_overflow', 20),
                pool_timeout=db_config_data.get('pool_timeout', 30),
                pool_recycle=db_config_data.get('pool_recycle', 3600),
                echo_sql=db_config_data.get('echo_sql', False)
            )
            
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {yaml_path}")
            logging.info("Using default database configuration")
            return cls()  # Return default values
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {yaml_path}: {e}")
            return cls()
        except Exception as e:
            logging.error(f"Error loading database config from YAML: {e}")
            return cls()


class EnhancedSystemConfig:
    """Enhanced system configuration loaded from MySQL database"""
    
    def __init__(self, db_config: DatabaseConfig, config_name: str = 'default'):
        self.db_config = db_config
        self.config_name = config_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize database manager
        from database.models import DatabaseManager
        self.db_manager = DatabaseManager(db_config.get_database_url())
        
        # Load configuration from database
        self._load_from_database()
    
    def _load_from_database(self):
        """Load configuration from MySQL database"""
        try:
            session = self.db_manager.get_session()
            
            # Import here to avoid circular imports
            from database.models import SystemConfig
            
            # Try to get existing configuration
            config_record = session.query(SystemConfig).filter(
                SystemConfig.config_name == self.config_name,
                SystemConfig.is_active == True
            ).first()
            
            if config_record:
                self.logger.info(f"Loading configuration '{self.config_name}' from MySQL database")
                self._apply_config(config_record)
            else:
                self.logger.info(f"Configuration '{self.config_name}' not found, creating default")
                self._create_default_config(session)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from MySQL database: {e}")
            self._apply_fallback_config()
    
    def _create_default_config(self, session: Session):
        """Create default configuration in MySQL database"""
        try:
            from database.models import SystemConfig
            
            default_config = SystemConfig(
                config_name=self.config_name,

                # API Configuration
                api_key=None,
                api_secret=None,
                demo_api_key=None,
                demo_api_secret=None,
                sandbox_mode=False,
                
                # Market Scanning
                min_volume_24h=5_000_000,
                max_symbols_scan=500,
                timeframe='30m',
                
                # Multi-Timeframe Configuration
                confirmation_timeframes=['1h', '4h', '6h'],
                mtf_confirmation_required=True,
                mtf_weight_multiplier=1.5,
                
                # Rate Limiting - UPDATED: Reduced for connection pool optimization
                max_requests_per_second=6.0,  # Reduced from 8.0 to 6.0
                api_timeout=30000,  # Increased from 20000 to 30000
                
                # Risk Management
                max_portfolio_risk=0.02,
                max_daily_trades=20,
                max_single_position_risk=0.005,
                
                # Threading
                max_workers=None,
                
                # Caching
                cache_ttl_seconds=60,
                max_cache_size=1000,
                
                # ML Model
                ml_training_samples=400,
                ml_profitable_rate=0.45,
                
                # Chart Settings
                show_charts=True,
                save_charts=False,
                charts_per_batch=5,
                chart_width=1400,
                chart_height=800,
                
                # Indicator parameters
                stoch_rsi_window=14,
                stoch_rsi_smooth_k=3,
                stoch_rsi_smooth_d=3,
                ichimoku_window1=9,
                ichimoku_window2=26,
                ichimoku_window3=52,
                
                # OHLCV Data Limits
                ohlcv_limit_primary=500,      # Primary timeframe analysis
                ohlcv_limit_mtf=200,          # MTF confirmation (faster)
                ohlcv_limit_analysis=500,      # General analysis operations

                # Telegram & Trading Configuration
                telegram_id='6708641837',
                telegram_bot_token='8088506547:AAHZZxiY_wlh48IN4ldPNRJtM9qi7qxfLdM',
                
                # FIXED: Single leverage value instead of JSON array
                leverage='max',  # Default to max leverage
                risk_amount=5.0,  # Risk percentage of account balance (5% default)
                
                # NEW: Encryption password for API secrets
                encryption_password='bybit_trading_system_secure_key_2024',
                
                # NEW: Auto-Trading Configuration
                max_concurrent_positions=5,
                max_execution_per_trade=2,
                day_trade_start_hour='01:00',
                scan_interval=3600,  # 3 hours in seconds
                auto_close_profit_at=10.0  # 10% profit target
            )
            
            session.add(default_config)
            session.commit()
            
            self.logger.info(f"Created default configuration '{self.config_name}' in MySQL database")
            self._apply_config(default_config)
            
        except Exception as e:
            self.logger.error(f"Error creating default configuration: {e}")
            session.rollback()
            self._apply_fallback_config()

    def _apply_config(self, config_record):
        """Apply configuration from database record"""
        # Apply all configuration attributes
        config_dict = config_record.to_dict()
        
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Post-processing
        self._post_init()
    
    def _apply_fallback_config(self):
        """Apply fallback configuration when database fails"""
        self.logger.warning("Using fallback configuration with 30m primary, 1h/4h/6h confirmation")
        
        # Set default values with correct timeframes
        self.api_key = None
        self.api_secret = None
        self.demo_api_key = None
        self.demo_api_secret = None
        self.sandbox_mode = False
        self.min_volume_24h = 5_000_000
        self.max_symbols_scan = 100
        self.timeframe = '30m'
        self.confirmation_timeframes = ['1h', '4h', '6h']
        self.mtf_confirmation_required = True
        self.mtf_weight_multiplier = 1.5
        self.max_requests_per_second = 6.0  # Reduced for connection pool optimization
        self.api_timeout = 30000  # Increased timeout
        self.max_portfolio_risk = 0.02
        self.max_daily_trades = 20
        self.max_single_position_risk = 0.005
        self.max_workers = None
        self.cache_ttl_seconds = 60
        self.max_cache_size = 1000
        self.ml_training_samples = 400
        self.ml_profitable_rate = 0.45
        self.show_charts = True
        self.save_charts = False
        self.charts_per_batch = 5
        self.chart_width = 1400
        self.chart_height = 800
        self.stoch_rsi_window = 14
        self.stoch_rsi_smooth_k = 3
        self.stoch_rsi_smooth_d = 3
        self.ichimoku_window1 = 9
        self.ichimoku_window2 = 26
        self.ichimoku_window3 = 52
        
        # OHLCV Data Limits Fallback
        self.ohlcv_limit_primary = 500      # Primary timeframe fallback
        self.ohlcv_limit_mtf = 200          # MTF confirmation fallback  
        self.ohlcv_limit_analysis = 500     # General analysis fallback

        # Telegram & Trading Configuration Fallback
        self.telegram_id = '6708641837'
        self.telegram_bot_token = '8088506547:AAHZZxiY_wlh48IN4ldPNRJtM9qi7qxfLdM'
        
        # FIXED: Single leverage value
        self.leverage = 'max'  # Default to max leverage
        self.risk_amount = 5.0  # Risk percentage of account balance
        
        # NEW: Encryption password fallback
        self.encryption_password = 'bybit_trading_system_secure_key_2024'
        
        # NEW: Auto-Trading Configuration Fallback
        self.max_concurrent_positions = 5
        self.max_execution_per_trade = 2
        self.day_trade_start_hour = '01:00'
        self.scan_interval = 3600  # 3 hours
        self.auto_close_profit_at = 10.0  # 10% profit
        
        self._post_init()

    def _post_init(self):
        """Validate and set derived values"""
        if self.max_workers is None:
            cpu_count = psutil.cpu_count(logical=False)
            self.max_workers = min(6, max(2, cpu_count - 1))
        
        # Set default confirmation timeframes if None
        if self.confirmation_timeframes is None:
            self.confirmation_timeframes = ['1h', '4h']
        
        # Validate ranges with connection pool optimization
        self.max_requests_per_second = max(1.0, min(8.0, self.max_requests_per_second))
        self.max_portfolio_risk = max(0.001, min(0.1, self.max_portfolio_risk))
        self.ml_profitable_rate = max(0.2, min(0.8, self.ml_profitable_rate))
        self.mtf_weight_multiplier = max(1.0, min(3.0, self.mtf_weight_multiplier))
        
        # Validate auto-trading parameters
        self.max_concurrent_positions = max(1, min(20, self.max_concurrent_positions))
        self.max_execution_per_trade = max(1, min(10, self.max_execution_per_trade))
        self.risk_amount = max(0.1, min(50.0, self.risk_amount))  # 0.1% to 50% of account balance
        self.auto_close_profit_at = max(0.5, min(100.0, self.auto_close_profit_at))  # 0.5% to 100%
        self.scan_interval = max(300, self.scan_interval)  # Minimum 5 minutes between scans
        
        # Validate leverage
        acceptable_leverage = ['10', '12.5', '25', '50', 'max']
        if self.leverage not in acceptable_leverage:
            self.logger.warning(f"Invalid leverage '{self.leverage}', defaulting to 'max'")
            self.leverage = 'max'
        
        # OPTIMIZATION: Adjust rate limiting for connection pool efficiency
        if self.max_requests_per_second > 6.0:
            self.logger.info(f"Adjusting rate limit from {self.max_requests_per_second} to 6.0 req/s for connection pool optimization")
            self.max_requests_per_second = 6.0
        
        # Ensure adequate timeout for connection reuse
        if self.api_timeout < 25000:
            self.logger.info(f"Increasing API timeout from {self.api_timeout} to 30000ms for better connection stability")
            self.api_timeout = 30000
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration in MySQL database"""
        try:
            session = self.db_manager.get_session()
            
            from database.models import SystemConfig
            
            config_record = session.query(SystemConfig).filter(
                SystemConfig.config_name == self.config_name,
                SystemConfig.is_active == True
            ).first()
            
            if config_record:
                # Update fields
                for key, value in kwargs.items():
                    if hasattr(config_record, key):
                        setattr(config_record, key, value)
                        setattr(self, key, value)
                
                config_record.updated_at = datetime.utcnow()
                session.commit()
                
                self.logger.info(f"Updated configuration '{self.config_name}' in MySQL database")
                session.close()
                return True
            else:
                self.logger.error(f"Configuration '{self.config_name}' not found for update")
                session.close()
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def get_all_configs(self) -> List[Dict[str, Any]]:
        """Get all available configurations from MySQL"""
        try:
            session = self.db_manager.get_session()
            
            from database.models import SystemConfig
            
            configs = session.query(SystemConfig).filter(
                SystemConfig.is_active == True
            ).all()
            
            result = []
            for config in configs:
                config_dict = config.to_dict()
                config_dict.update({
                    'id': config.id,
                    'config_name': config.config_name,
                    'created_at': config.created_at,
                    'updated_at': config.updated_at
                })
                result.append(config_dict)
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting configurations: {e}")
            return []
    
    def delete_config(self, config_name: str) -> bool:
        """Soft delete a configuration (set is_active=False)"""
        try:
            if config_name == 'default':
                self.logger.error("Cannot delete default configuration")
                return False
            
            session = self.db_manager.get_session()
            
            from database.models import SystemConfig
            
            config_record = session.query(SystemConfig).filter(
                SystemConfig.config_name == config_name,
                SystemConfig.is_active == True
            ).first()
            
            if config_record:
                config_record.is_active = False
                session.commit()
                
                self.logger.info(f"Deleted configuration '{config_name}'")
                session.close()
                return True
            else:
                self.logger.error(f"Configuration '{config_name}' not found")
                session.close()
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting configuration: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    def export_to_yaml(self, file_path: str) -> bool:
        """Export current configuration to YAML file (for backup)"""
        try:
            config_dict = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'demo_api_key': self.demo_api_key,
                'demo_api_secret': self.demo_api_secret,
                'sandbox_mode': self.sandbox_mode,
                'min_volume_24h': self.min_volume_24h,
                'max_symbols_scan': self.max_symbols_scan,
                'timeframe': self.timeframe,
                'confirmation_timeframes': self.confirmation_timeframes,
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
                # FIXED: Export single leverage value
                'leverage': self.leverage,
                'risk_amount': self.risk_amount,  # Risk percentage of account balance
                # NEW: Encryption password
                'encryption_password': self.encryption_password,
                # NEW: Auto-trading configuration
                'max_concurrent_positions': self.max_concurrent_positions,
                'max_execution_per_trade': self.max_execution_per_trade,
                'day_trade_start_hour': self.day_trade_start_hour,
                'scan_interval': self.scan_interval,
                'auto_close_profit_at': self.auto_close_profit_at
            }
            
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            self.logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'api_key': getattr(self, 'api_key', None),
            'api_secret': getattr(self, 'api_secret', None),
            'demo_api_key': getattr(self, 'demo_api_key', None),
            'demo_api_secret': getattr(self, 'demo_api_secret', None),
            'sandbox_mode': getattr(self, 'sandbox_mode', False),
            'min_volume_24h': getattr(self, 'min_volume_24h', 5_000_000),
            'max_symbols_scan': getattr(self, 'max_symbols_scan', 100),
            'timeframe': getattr(self, 'timeframe', '30m'),
            'confirmation_timeframes': getattr(self, 'confirmation_timeframes', ['1h', '4h', '6h']),
            'mtf_confirmation_required': getattr(self, 'mtf_confirmation_required', True),
            'mtf_weight_multiplier': getattr(self, 'mtf_weight_multiplier', 1.5),
            'max_requests_per_second': getattr(self, 'max_requests_per_second', 6.0),
            'api_timeout': getattr(self, 'api_timeout', 30000),
            'max_portfolio_risk': getattr(self, 'max_portfolio_risk', 0.02),
            'max_daily_trades': getattr(self, 'max_daily_trades', 20),
            'max_single_position_risk': getattr(self, 'max_single_position_risk', 0.005),
            'max_workers': getattr(self, 'max_workers', None),
            'cache_ttl_seconds': getattr(self, 'cache_ttl_seconds', 60),
            'max_cache_size': getattr(self, 'max_cache_size', 1000),
            'ml_training_samples': getattr(self, 'ml_training_samples', 400),
            'ml_profitable_rate': getattr(self, 'ml_profitable_rate', 0.45),
            'show_charts': getattr(self, 'show_charts', True),
            'save_charts': getattr(self, 'save_charts', False),
            'charts_per_batch': getattr(self, 'charts_per_batch', 5),
            'chart_width': getattr(self, 'chart_width', 1400),
            'chart_height': getattr(self, 'chart_height', 800),
            'stoch_rsi_window': getattr(self, 'stoch_rsi_window', 14),
            'stoch_rsi_smooth_k': getattr(self, 'stoch_rsi_smooth_k', 3),
            'stoch_rsi_smooth_d': getattr(self, 'stoch_rsi_smooth_d', 3),
            'ichimoku_window1': getattr(self, 'ichimoku_window1', 9),
            'ichimoku_window2': getattr(self, 'ichimoku_window2', 26),
            'ichimoku_window3': getattr(self, 'ichimoku_window3', 52),
            'ohlcv_limit_primary': getattr(self, 'ohlcv_limit_primary', 500),
            'ohlcv_limit_mtf': getattr(self, 'ohlcv_limit_mtf', 200),
            'ohlcv_limit_analysis': getattr(self, 'ohlcv_limit_analysis', 500),
            'telegram_id': getattr(self, 'telegram_id', '6708641837'),
            'telegram_bot_token': getattr(self, 'telegram_bot_token', '8088506547:AAHZZxiY_wlh48IN4ldPNRJtM9qi7qxfLdM'),
            # FIXED: Single leverage value
            'leverage': getattr(self, 'leverage', 'max'),
            'risk_amount': getattr(self, 'risk_amount', 5.0),  # Risk percentage of account balance
            # NEW: Encryption password
            'encryption_password': getattr(self, 'encryption_password', 'bybit_trading_system_secure_key_2024'),
            # NEW: Auto-trading configuration
            'max_concurrent_positions': getattr(self, 'max_concurrent_positions', 5),
            'max_execution_per_trade': getattr(self, 'max_execution_per_trade', 2),
            'day_trade_start_hour': getattr(self, 'day_trade_start_hour', '01:00'),
            'scan_interval': getattr(self, 'scan_interval', 3600),
            'auto_close_profit_at': getattr(self, 'auto_close_profit_at', 10.0)
        }
    
    @classmethod
    def from_database(cls, db_config: DatabaseConfig, config_name: str = 'default') -> 'EnhancedSystemConfig':
        """Create configuration instance from MySQL database"""
        return cls(db_config, config_name)
    
    @classmethod
    def load_from_yaml_and_database(cls, yaml_path: str, config_name: str = 'default') -> 'EnhancedSystemConfig':
        """Load database config from YAML, then system config from database"""
        # First load database configuration from YAML
        db_config = DatabaseConfig.from_yaml_file(yaml_path)
        
        # Then load system configuration from database using the YAML database config
        return cls.from_database(db_config, config_name)
    
    @classmethod  
    def from_yaml_file(cls, yaml_path: str) -> DatabaseConfig:
        """Load database configuration from YAML file (for backwards compatibility)"""
        return DatabaseConfig.from_yaml_file(yaml_path)