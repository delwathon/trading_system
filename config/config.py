"""
Configuration management for the Enhanced Bybit Trading System.
"""

import os
import yaml
import logging
import psutil
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EnhancedSystemConfig:
    """Enhanced system configuration with multi-timeframe settings"""
    # API Configuration
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox_mode: bool = False
    
    # Market Scanning
    min_volume_24h: float = 5_000_000
    max_symbols_scan: int = 100
    timeframe: str = '1h'  # Primary timeframe
    
    # Multi-Timeframe Configuration
    confirmation_timeframes: List[str] = None  # Additional timeframes for confirmation
    mtf_confirmation_required: bool = True  # Require multi-timeframe confirmation
    mtf_weight_multiplier: float = 1.5  # Boost confidence for confirmed signals
    
    # Rate Limiting
    max_requests_per_second: float = 8.0
    api_timeout: int = 20000
    
    # Risk Management
    max_portfolio_risk: float = 0.02
    max_daily_trades: int = 20
    max_single_position_risk: float = 0.005
    
    # Threading
    max_workers: Optional[int] = None
    
    # Caching
    cache_ttl_seconds: int = 60
    max_cache_size: int = 1000
    
    # ML Model
    ml_training_samples: int = 400
    ml_profitable_rate: float = 0.45
    
    # CSV Output
    csv_base_filename: str = "enhanced_bybit_signals"
    
    # Chart Settings
    show_charts: bool = True
    save_charts: bool = True
    charts_per_batch: int = 5
    chart_width: int = 1400
    chart_height: int = 800

    # Indicator parameters
    stoch_rsi_window: int = 14
    stoch_rsi_smooth_k: int = 3
    stoch_rsi_smooth_d: int = 3
    ichimoku_window1: int = 9
    ichimoku_window2: int = 26
    ichimoku_window3: int = 52
    
    @classmethod
    def from_file(cls, config_path: str) -> 'EnhancedSystemConfig':
        """Load configuration from YAML file, filtering out non-dataclass fields"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Get the field names that are actually defined in this dataclass
            try:
                if hasattr(cls, '__dataclass_fields__'):
                    # Python 3.7+ dataclass
                    dataclass_fields = cls.__dataclass_fields__
                    if isinstance(next(iter(dataclass_fields.values())), str):
                        # If fields are strings (some Python versions)
                        valid_fields = set(dataclass_fields.keys())
                    else:
                        # If fields are field objects (standard)
                        valid_fields = {field.name for field in dataclass_fields.values()}
                else:
                    # Fallback: get field names from __annotations__
                    valid_fields = set(cls.__annotations__.keys())
            except Exception:
                # Ultimate fallback: define the fields manually
                valid_fields = {
                    'api_key', 'api_secret', 'sandbox_mode', 'min_volume_24h', 'max_symbols_scan',
                    'timeframe', 'confirmation_timeframes', 'mtf_confirmation_required', 
                    'mtf_weight_multiplier', 'max_requests_per_second', 'api_timeout',
                    'max_portfolio_risk', 'max_daily_trades', 'max_single_position_risk',
                    'max_workers', 'cache_ttl_seconds', 'max_cache_size', 'ml_training_samples',
                    'ml_profitable_rate', 'csv_base_filename', 'show_charts', 'save_charts',
                    'charts_per_batch', 'chart_width', 'chart_height',
                    'stoch_rsi_window', 'stoch_rsi_smooth_k', 'stoch_rsi_smooth_d',
                    'ichimoku_window1', 'ichimoku_window2', 'ichimoku_window3'
                }
            
            # Filter config_data to only include valid fields
            filtered_config = {k: v for k, v in config_data.items() 
                            if k in valid_fields}
            
            # Handle confirmation_timeframes default
            if 'confirmation_timeframes' not in filtered_config or filtered_config['confirmation_timeframes'] is None:
                filtered_config['confirmation_timeframes'] = ['1h', '4h']
            
            # Log what fields were filtered out (for debugging)
            filtered_out = set(config_data.keys()) - valid_fields
            if filtered_out:
                logging.info(f"Filtered out config fields not needed by engine.py: {filtered_out}")
            
            logging.info(f"Loading config with fields: {list(filtered_config.keys())}")
            
            return cls(**filtered_config)
            
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            try:
                logging.error(f"Available config fields: {list(config_data.keys()) if 'config_data' in locals() else 'Could not load'}")
            except:
                pass
            return cls()
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        try:
            # Convert dataclass to dict
            config_dict = self.__dict__.copy()
            
            # Write to file
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logging.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def __post_init__(self):
        """Validate and set derived values"""
        if self.max_workers is None:
            cpu_count = psutil.cpu_count(logical=False)
            self.max_workers = min(6, max(2, cpu_count - 1))
        
        # Set default confirmation timeframes if None
        if self.confirmation_timeframes is None:
            self.confirmation_timeframes = ['4h', '6h']
        
        self.max_requests_per_second = max(1.0, min(20.0, self.max_requests_per_second))
        self.max_portfolio_risk = max(0.001, min(0.1, self.max_portfolio_risk))
        self.ml_profitable_rate = max(0.2, min(0.8, self.ml_profitable_rate))
        self.mtf_weight_multiplier = max(1.0, min(3.0, self.mtf_weight_multiplier))