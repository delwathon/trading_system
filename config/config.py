"""
Database-based configuration management for the Enhanced Bybit Trading System.
Replaces YAML-based configuration with database storage.
Primary timeframe: 15m, Confirmation timeframes: 1h, 4h
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
    """Database connection configuration - MySQL only"""
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
                timeframe='15m',
                
                # Multi-Timeframe Configuration
                confirmation_timeframes=['1h', '4h'],
                mtf_confirmation_required=True,
                mtf_weight_multiplier=1.5,
                
                # Rate Limiting
                max_requests_per_second=8.0,
                api_timeout=20000,
                
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
                
                # CSV Output
                csv_base_filename="enhanced_bybit_signals_mtf",
                
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
                
                # ===== NEW: OHLCV Data Limits =====
                ohlcv_limit_primary=500,      # Primary timeframe analysis
                ohlcv_limit_mtf=200,          # MTF confirmation (faster)
                ohlcv_limit_analysis=500      # General analysis operations
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
        self.logger.warning("Using fallback configuration with 15m primary, 1h/4h confirmation")
        
        # Set default values with correct timeframes
        self.api_key = None
        self.api_secret = None
        self.demo_api_key = None
        self.demo_api_secret = None
        self.sandbox_mode = False
        self.min_volume_24h = 5_000_000
        self.max_symbols_scan = 100
        self.timeframe = '15m'
        self.confirmation_timeframes = ['1h', '4h']
        self.mtf_confirmation_required = True
        self.mtf_weight_multiplier = 1.5
        self.max_requests_per_second = 8.0
        self.api_timeout = 20000
        self.max_portfolio_risk = 0.02
        self.max_daily_trades = 20
        self.max_single_position_risk = 0.005
        self.max_workers = None
        self.cache_ttl_seconds = 60
        self.max_cache_size = 1000
        self.ml_training_samples = 400
        self.ml_profitable_rate = 0.45
        self.csv_base_filename = "enhanced_bybit_signals"
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
        
        # ===== NEW: OHLCV Data Limits Fallback =====
        self.ohlcv_limit_primary = 500      # Primary timeframe fallback
        self.ohlcv_limit_mtf = 200          # MTF confirmation fallback  
        self.ohlcv_limit_analysis = 500     # General analysis fallback
        
        self._post_init()

    def _post_init(self):
        """Validate and set derived values"""
        if self.max_workers is None:
            cpu_count = psutil.cpu_count(logical=False)
            self.max_workers = min(6, max(2, cpu_count - 1))
        
        # Set default confirmation timeframes if None
        if self.confirmation_timeframes is None:
            self.confirmation_timeframes = ['1h', '4h']
        
        # Validate ranges
        self.max_requests_per_second = max(1.0, min(20.0, self.max_requests_per_second))
        self.max_portfolio_risk = max(0.001, min(0.1, self.max_portfolio_risk))
        self.ml_profitable_rate = max(0.2, min(0.8, self.ml_profitable_rate))
        self.mtf_weight_multiplier = max(1.0, min(3.0, self.mtf_weight_multiplier))
    
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
                'ichimoku_window3': self.ichimoku_window3
            }
            
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            self.logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    @classmethod
    def from_database(cls, db_config: DatabaseConfig, config_name: str = 'default') -> 'EnhancedSystemConfig':
        """Create configuration instance from MySQL database"""
        return cls(db_config, config_name)
    
    @classmethod  
    def from_yaml_file(cls, yaml_path: str) -> DatabaseConfig:
        """Load database configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract database configuration
            db_config_data = data.get('database', {})
            
            return DatabaseConfig(
                database_type='mysql',  # Only MySQL supported
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
            
        except Exception as e:
            logging.error(f"Error loading database config from YAML: {e}")
            # Return default MySQL configuration
            return DatabaseConfig(
                database_type='mysql',
                host='localhost',
                port=3306,
                username='root',
                password='',
                database='bybit_trading_system'
            )