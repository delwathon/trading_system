#!/usr/bin/env python3
"""
Auto-Trading Main Entry Point for Enhanced Bybit Trading System.
Runs the automated trading system with scheduled scanning and position management.
UPDATED: Includes Telegram bootstrap mode for API key configuration.
COMPLETE: All functionality from project knowledge included.
"""

import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables FIRST
load_dotenv()

# Initialize logging ONCE at the very beginning
from utils.logging import setup_logging, get_logger
logger = setup_logging()  # This will be the ONLY log file created

from config.config import EnhancedSystemConfig, DatabaseConfig
from trading_module.autotrader import AutoTrader
from trading_module.bybit import LeverageManager
from database.models import DatabaseManager
from utils.logging import setup_logging
from telegram_bot_and_notification.bootstrap_manager import TelegramBootstrapManager, run_bootstrap_mode, check_bootstrap_needed


def validate_auto_trading_config(config: EnhancedSystemConfig, skip_api_check: bool = False) -> bool:
    """Validate auto-trading configuration"""
    logger = get_logger()
    
    # Validate leverage
    if not config.leverage or config.leverage not in LeverageManager.ACCEPTABLE_LEVERAGE:
        logger.error(f"❌ Invalid leverage configuration: '{config.leverage}'")
        logger.error(f"   Must be one of: {LeverageManager.ACCEPTABLE_LEVERAGE}")
        return False
    
    # Validate risk amount
    if config.risk_amount <= 0 or config.risk_amount > 50:
        logger.error(f"❌ Invalid risk amount: {config.risk_amount}%")
        logger.error("   Risk amount must be between 0.1% and 50% of account balance")
        return False
    
    # Validate position limits
    if config.max_concurrent_positions <= 0 or config.max_concurrent_positions > 20:
        logger.error(f"❌ Invalid max_concurrent_positions: {config.max_concurrent_positions}")
        logger.error("   Must be between 1 and 20")
        return False
    
    if config.max_execution_per_trade <= 0 or config.max_execution_per_trade > config.max_concurrent_positions:
        logger.error(f"❌ Invalid max_execution_per_trade: {config.max_execution_per_trade}")
        logger.error(f"   Must be between 1 and {config.max_concurrent_positions}")
        return False
    
    # Validate profit target
    if config.auto_close_profit_at <= 0 or config.auto_close_profit_at > 1000:
        logger.error(f"❌ Invalid auto_close_profit_at: {config.auto_close_profit_at}%")
        logger.error("   Must be between 0.1% and 1000%")
        return False
    
    # Validate loss target
    if config.auto_close_loss_at <= 0 or config.auto_close_loss_at > 1000:
        logger.error(f"❌ Invalid auto_close_loss_at: {config.auto_close_loss_at}%")
        logger.error("   Must be between 0.1% and 1000%")
        return False
    
    # Validate scan interval
    if config.scan_interval < 300:  # Minimum 5 minutes
        logger.error(f"❌ Invalid scan_interval: {config.scan_interval} seconds")
        logger.error("   Must be at least 300 seconds (5 minutes)")
        return False
    
    # Validate Telegram configuration
    if not config.telegram_bot_token or not config.telegram_id:
        logger.error("❌ Missing Telegram configuration")
        logger.error("   Please set telegram_bot_token and telegram_id in database configuration")
        return False
    
    # Skip API credential check if in bootstrap mode
    # if not skip_api_check:
    #     # Validate API credentials
    #     if not config.bybit_live_api_key or not config.bybit_live_api_secret:
    #         logger.error("❌ Missing API credentials")
    #         logger.error("   Please set bybit_live_api_key and bybit_live_api_secret in database configuration")
    #         return False
    
    return True


def display_auto_trading_config(config: EnhancedSystemConfig):
    """Display auto-trading configuration"""
    logger = get_logger()
    
    logger.info("🤖 AUTO-TRADING CONFIGURATION:")
    logger.info("=" * 80)
    
    # Trading parameters
    logger.info("📊 Trading Parameters:")
    logger.info(f"   Leverage: {config.leverage}")
    logger.info(f"   Risk Amount per Trade: {config.risk_amount}% of account balance")
    logger.info(f"   Max Concurrent Positions: {config.max_concurrent_positions}")
    logger.info(f"   Max Executions per Scan: {config.max_execution_per_trade}")
    logger.info("   Auto-Execute Trades: " + ("Enabled" if config.auto_execute_trades else "Disabled"))
    logger.info("   Auto-Close Trades: " + ("Enabled" if config.auto_close_enabled else "Disabled"))
    if config.auto_close_enabled:
        if config.auto_close_profit_at == 1000:
            logger.info(f"   Auto-Close Profit Target: TP1")
        else:
            logger.info(f"   Auto-Close Profit Target: {config.auto_close_profit_at}%")
        logger.info(f"   Auto-Close Loss Target: {config.auto_close_loss_at}%")
    
    # Scheduling parameters
    scan_interval_hours = config.scan_interval / 3600
    logger.info("⏰ Scheduling Parameters:")
    logger.info(f"   Daily Start Time: {config.day_trade_start_hour}")
    logger.info(f"   Scan Interval: {scan_interval_hours:.1f} hours ({config.scan_interval} seconds)")
    
    # Analysis parameters
    logger.info("📈 Analysis Configuration:")
    logger.info(f"   Primary Timeframe: {config.timeframe}")
    logger.info(f"   Confirmation Timeframes: {', '.join(config.confirmation_timeframes)}")
    logger.info(f"   MTF Confirmation Required: {config.mtf_confirmation_required}")
    logger.info(f"   Min Volume 24h: ${config.min_volume_24h:,}")
    
    # Risk management
    logger.info("⚠️  Risk Management:")
    logger.info(f"   Risk per Trade: {config.risk_amount}% of account balance")
    max_risk_exposure_pct = config.max_concurrent_positions * config.risk_amount
    logger.info(f"   Max Total Risk Exposure: {max_risk_exposure_pct}% of account balance")
    if config.leverage == 'max':
        logger.info(f"   Max Leveraged Exposure: {max_risk_exposure_pct}% × Max Leverage")
    else:
        max_leveraged_exposure_pct = max_risk_exposure_pct * float(config.leverage)
        logger.info(f"   Max Leveraged Exposure: {max_leveraged_exposure_pct}% of account balance")
    
    logger.info(f"   Portfolio Risk Limit: {config.max_portfolio_risk * 100:.1f}%")
    
    # Mode
    mode = "DEMO" if config.sandbox_mode else "PRODUCTION"
    logger.info(f"🔧 Trading Mode: {mode}")
    
    # Telegram
    logger.info("📱 Telegram Configuration:")
    logger.info(f"   Bot Token: {'Configured' if config.telegram_bot_token else 'Missing'}")
    logger.info(f"   User ID: {config.telegram_id}")
    
    logger.info("=" * 80)


def create_default_config_with_autotrading():
    """Create default configuration file with auto-trading settings"""
    import yaml
    
    config = {
        'database': {
            'type': 'mysql',
            'host': 'localhost',
            'port': 3306,
            'database': 'bybit_trading_system',
            'username': 'root',
            'password': 'your_password_here',
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'echo_sql': False
        },
        'config': {
            'config_name': 'default',
            'auto_create_default': True,
            'backup_on_update': True,
            'backup_directory': 'config_backups'
        },
        'system': {
            'enable_db_logging': True,
            'auto_cleanup_enabled': True,
            'cleanup_days_to_keep': 30,
            'cleanup_schedule': 'daily',
            'auto_vacuum': True,
            'backup_on_startup': False
        },
        'auto_trading': {
            'enabled': True,
            'bootstrap_mode': True,
            'note': 'Auto-trading parameters are stored in database, not this file'
        }
    }
    
    config_path = 'enhanced_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"📄 Created auto-trading configuration file: {config_path}")
    logger.info("⚠️  Please update the MySQL password!")
    logger.info("🔑 API credentials will be configured via Telegram bootstrap mode")
    return config_path


def setup_database_with_autotrading(db_config: DatabaseConfig) -> bool:
    """Setup database with auto-trading tables"""
    try:
        # logger.info("🔗 Testing MySQL database connection...")
        
        # Test connection
        db_manager = DatabaseManager(db_config.get_database_url())
        if not db_manager.test_connection():
            logger.info("❌ Database connection failed!")
            logger.info(f"   Please ensure MySQL is running at {db_config.host}:{db_config.port}")
            logger.info(f"   Database: {db_config.database}")
            logger.info(f"   Username: {db_config.username}")
            logger.info(f"   Password: {'Set' if db_config.password else 'NOT SET'}")
            return False
        
        # Create tables (including new auto-trading tables)
        # logger.info("📊 Creating database tables (including auto-trading tables)...")
        db_manager.create_tables()
        
        # logger.info("✅ MySQL database setup complete with auto-trading support!")
        return True
        
    except Exception as e:
        logger.info(f"❌ Database setup failed: {e}")
        return False


async def handle_bootstrap_mode(config: EnhancedSystemConfig) -> bool:
    """Handle Telegram bootstrap mode for API key configuration"""
    try:
        logger.info("🔄 BOOTSTRAP MODE REQUIRED")
        logger.info("=" * 50)
        logger.info("⚠️  API credentials are missing and need to be configured")
        logger.info("📱 Telegram bootstrap mode will guide you through the setup")
        print()
        logger.info("📋 What will happen:")
        logger.info("   1. Telegram bot will send you configuration instructions")
        logger.info("   2. You'll enter your API keys securely via Telegram")
        logger.info("   3. All keys will be encrypted before storage")
        logger.info("   4. System will automatically start trading once configured")
        print()
        logger.info("🚀 Starting Telegram bootstrap mode...")
        logger.info("📱 Check your Telegram for setup instructions!")
        print()
        
        # Run bootstrap mode
        bootstrap_success = await run_bootstrap_mode(config)
        
        if bootstrap_success:
            logger.info("✅ Bootstrap mode completed successfully!")
            logger.info("🔑 API credentials have been encrypted and saved")
            logger.info("🚀 System is now ready for auto-trading")
            return True
        else:
            logger.info("❌ Bootstrap mode failed")
            logger.info("   Please check Telegram bot configuration and try again")
            return False
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Bootstrap mode interrupted by user")
        return False
    except Exception as e:
        logger.info(f"❌ Bootstrap mode error: {e}")
        return False


def load_configuration(config_path: str = 'enhanced_config.yaml'):
    """Load configuration from YAML file and database"""
    logger = setup_logging()
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return None, None
    
    try:
        # Load database configuration from YAML
        logger.debug(f"Loading database configuration from {config_path}")
        db_config = DatabaseConfig.from_yaml_file(config_path)
        
        logger.debug(f"Database config loaded:")
        logger.debug(f"   Host: {db_config.host}:{db_config.port}")
        logger.debug(f"   Database: {db_config.database}")
        logger.debug(f"   Username: {db_config.username}")
        logger.debug(f"   Password: {'***' if db_config.password else 'NOT SET'}")
        
        # Load system configuration from database using the YAML database config
        logger.debug("Loading system configuration from MySQL database...")
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        return db_config, config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None, None


def main():
    """Main execution function for auto-trading system with Telegram bootstrap"""
    try:
        logger.info("🤖 ENHANCED BYBIT AUTO-TRADING SYSTEM")
        logger.info("🚀 Automated Trading with Multi-Timeframe Analysis & Telegram Bootstrap")
        logger.info("=" * 80)
        logger.info("   ✅ Scheduled scanning (configurable intervals)")
        logger.info("   ✅ Automated position management")  
        logger.info("   ✅ Leverage-aware position sizing")
        logger.info("   ✅ Profit-based auto-closing")
        logger.info("   ✅ Multi-timeframe confirmation")
        logger.info("   ✅ MySQL database storage")
        logger.info("   ✅ Risk management controls")
        logger.info("   ✅ Telegram bootstrap mode for API setup")
        print()
        
        # Create or load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            logger.info("📄 Creating default configuration...")
            create_default_config_with_autotrading()
            logger.info("⚠️  Please update the MySQL password and run again!")
            return
        
        # Load configuration from YAML and database
        db_config, config = load_configuration(config_path)
        
        if not db_config or not config:
            logger.info("❌ Failed to load configuration")
            logger.info(f"   Please ensure {config_path} exists and has valid database settings")
            return
        
        # logger.info(f"⚙️  Database Configuration:")
        # logger.info(f"   Host: {db_config.host}:{db_config.port}")
        # logger.info(f"   Database: {db_config.database}")
        # logger.info(f"   Username: {db_config.username}")
        # print()
        
        # Setup database with auto-trading support
        if not setup_database_with_autotrading(db_config):
            logger.info("❌ Cannot proceed without database connection")
            return
        
        # Check if bootstrap mode is needed
        if check_bootstrap_needed(config):
            logger.info("🔑 API credentials missing - entering bootstrap mode")
            
            # Run bootstrap mode
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                bootstrap_success = loop.run_until_complete(handle_bootstrap_mode(config))
                
                if not bootstrap_success:
                    logger.info("❌ Bootstrap mode failed - cannot start auto-trading")
                    return
                
                # Reload configuration after bootstrap
                logger.info("🔄 Reloading configuration after bootstrap...")
                db_config, config = load_configuration(config_path)
                
            finally:
                loop.close()
        
        # Display configuration
        display_auto_trading_config(config)
        
        # Validate auto-trading configuration (skip API check since bootstrap completed)
        if not validate_auto_trading_config(config, skip_api_check=False):
            logger.info("❌ Auto-trading configuration validation failed!")
            logger.info("   Please fix the configuration issues and try again.")
            return
        
        # logger.info("✅ Auto-trading configuration validated successfully!")
        print()
        
        # Initialize and start auto-trader
        logger.info("🚀 Initializing Auto-Trading System...")
        auto_trader = AutoTrader(config)
        
        # Test exchange connection
        if not auto_trader.trading_system.exchange_manager.test_connection():
            logger.info("❌ Failed to connect to Bybit exchange")
            logger.info("   Please check your API credentials and network connection")
            return
        
        # logger.info("✅ Exchange connection successful!")
        print()
        
        # Start auto-trading
        logger.info("🤖 Starting Automated Trading Session...")
        logger.info("   📱 You'll receive Telegram notifications for all trades")
        logger.info("   Press Ctrl+C to stop the auto-trader")
        print()
        logger.info("=" * 80)
        print()
        
        # Run main trading loop
        auto_trader.main_trading_loop()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Auto-trader stopped by user")
    except Exception as e:
        logger.info(f"\n❌ Auto-trader failed: {e}")
        import traceback
        traceback.print_exc()


def run_single_scan():
    """Run a single scan without starting the auto-trading loop (for testing)"""
    try:
        logger.info("🔍 SINGLE SCAN MODE - Testing Auto-Trading System")
        logger.info("=" * 50)
        
        # Load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            logger.info("❌ Configuration file not found. Run main() first.")
            return
        
        db_config, config = load_configuration(config_path)
        if not db_config or not config:
            logger.info("❌ Failed to load configuration")
            return
        
        # Check for bootstrap requirement
        if check_bootstrap_needed(config):
            logger.info("❌ Bootstrap mode required - API credentials missing")
            logger.info("   Run main() first to configure API keys via Telegram")
            return
        
        # Validate configuration
        if not validate_auto_trading_config(config):
            logger.info("❌ Configuration validation failed!")
            return
        
        # Initialize auto-trader
        auto_trader = AutoTrader(config)
        
        # Test connection
        if not auto_trader.trading_system.exchange_manager.test_connection():
            logger.info("❌ Exchange connection failed!")
            return
        
        logger.info("✅ Running single scan and execution test...")
        
        # FIXED: Run single scan with proper results display
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the complete analysis (same as the auto-trader uses)
            logger.info("📊 Running complete multi-timeframe analysis...")
            results = auto_trader.trading_system.run_complete_analysis_parallel_mtf()
            
            if results:
                # Display the comprehensive results table
                logger.info("\n" + "=" * 110)
                auto_trader.trading_system.print_comprehensive_results_with_mtf(results)
                logger.info("=" * 110)
                
                # Show execution simulation
                if results.get('top_opportunities'):
                    signals_count = len(results.get('signals', []))
                    opportunities_count = len(results['top_opportunities'])
                    
                    logger.info(f"\n📊 SCAN SIMULATION RESULTS:")
                    logger.info(f"   Total Signals Generated: {signals_count}")
                    logger.info(f"   Top Opportunities: {opportunities_count}")
                    logger.info(f"   Would Execute: {min(config.max_execution_per_trade, opportunities_count)} trades")
                    logger.info(f"   Available Position Slots: {config.max_concurrent_positions}")
                    logger.info(f"   Risk per Trade: {config.risk_amount}% of account balance")
                    
                    # Show top opportunities that would be executed
                    execution_count = min(config.max_execution_per_trade, opportunities_count)
                    if execution_count > 0:
                        logger.info(f"\n🎯 TRADES THAT WOULD BE EXECUTED:")
                        for i, opp in enumerate(results['top_opportunities'][:execution_count]):
                            logger.info(f"   {i+1}. {opp['symbol']} {opp['side'].upper()} - {opp['confidence']:.1f}% confidence - MTF: {opp.get('mtf_status', 'N/A')}")
                else:
                    logger.info(f"\n📊 SCAN RESULTS:")
                    logger.info(f"   No trading opportunities found in this scan")
                
            else:
                logger.info("❌ No analysis results generated")
            
        finally:
            loop.close()
        
        logger.info("✅ Single scan test completed successfully!")
        
    except Exception as e:
        logger.info(f"❌ Single scan failed: {e}")
        import traceback
        traceback.print_exc()


async def test_bootstrap():
    """Test bootstrap mode functionality"""
    try:
        logger.info("🧪 BOOTSTRAP MODE TEST")
        logger.info("=" * 30)
        
        # Load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            logger.info("❌ Configuration file not found")
            return
        
        db_config, config = load_configuration(config_path)
        if not db_config or not config:
            logger.info("❌ Failed to load configuration")
            return
        
        # Test bootstrap functionality
        bootstrap_manager = TelegramBootstrapManager(config)
        
        logger.info(f"Bootstrap needed: {bootstrap_manager.should_enter_bootstrap_mode()}")
        
        cred_status = bootstrap_manager.check_api_credentials()
        logger.info(f"Credentials status: {cred_status}")
        
        if bootstrap_manager.should_enter_bootstrap_mode():
            logger.info("🔄 Would enter bootstrap mode...")
            logger.info("📱 Check Telegram for configuration instructions")
        else:
            logger.info("✅ All credentials configured - no bootstrap needed")
        
    except Exception as e:
        logger.info(f"❌ Bootstrap test failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run single scan for testing
            run_single_scan()
        elif sys.argv[1] == "bootstrap":
            # Test bootstrap mode
            asyncio.run(test_bootstrap())
        elif sys.argv[1] == "config":
            # Show configuration
            try:
                config_path = 'enhanced_config.yaml'
                db_config, config = load_configuration(config_path)
                if config:
                    display_auto_trading_config(config)
                else:
                    logger.info("❌ Failed to load configuration")
            except Exception as e:
                logger.info(f"❌ Error showing configuration: {e}")
    else:
        # Run full auto-trading system
        main()