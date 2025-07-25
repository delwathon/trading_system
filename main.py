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
from bybit.autotrader import AutoTrader, LeverageManager
from database.models import DatabaseManager
from utils.logging import setup_logging
from notifier.telegram import TelegramBootstrapManager, run_bootstrap_mode, check_bootstrap_needed


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
    if not skip_api_check:
        # Validate API credentials
        if not config.api_key or not config.api_secret:
            logger.error("❌ Missing API credentials")
            logger.error("   Please set api_key and api_secret in database configuration")
            return False
    
    return True


def display_auto_trading_config(config: EnhancedSystemConfig):
    """Display auto-trading configuration"""
    logger = get_logger()
    
    logger.info("🤖 AUTO-TRADING CONFIGURATION:")
    logger.info("=" * 50)
    
    # Trading parameters
    logger.info("📊 Trading Parameters:")
    logger.info(f"   Leverage: {config.leverage}")
    logger.info(f"   Risk Amount per Trade: {config.risk_amount}% of account balance")
    logger.info(f"   Max Concurrent Positions: {config.max_concurrent_positions}")
    logger.info(f"   Max Executions per Scan: {config.max_execution_per_trade}")
    logger.info(f"   Auto-Close Profit Target: {config.auto_close_profit_at}%")
    
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
    
    logger.info("=" * 50)


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
    
    print(f"📄 Created auto-trading configuration file: {config_path}")
    print("⚠️  Please update the MySQL password!")
    print("🔑 API credentials will be configured via Telegram bootstrap mode")
    return config_path


def setup_database_with_autotrading(db_config: DatabaseConfig) -> bool:
    """Setup database with auto-trading tables"""
    try:
        # print("🔗 Testing MySQL database connection...")
        
        # Test connection
        db_manager = DatabaseManager(db_config.get_database_url())
        if not db_manager.test_connection():
            print("❌ Database connection failed!")
            print(f"   Please ensure MySQL is running at {db_config.host}:{db_config.port}")
            print(f"   Database: {db_config.database}")
            print(f"   Username: {db_config.username}")
            print(f"   Password: {'Set' if db_config.password else 'NOT SET'}")
            return False
        
        # Create tables (including new auto-trading tables)
        # print("📊 Creating database tables (including auto-trading tables)...")
        db_manager.create_tables()
        
        # print("✅ MySQL database setup complete with auto-trading support!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


async def handle_bootstrap_mode(config: EnhancedSystemConfig) -> bool:
    """Handle Telegram bootstrap mode for API key configuration"""
    try:
        print("🔄 BOOTSTRAP MODE REQUIRED")
        print("=" * 50)
        print("⚠️  API credentials are missing and need to be configured")
        print("📱 Telegram bootstrap mode will guide you through the setup")
        print("")
        print("📋 What will happen:")
        print("   1. Telegram bot will send you configuration instructions")
        print("   2. You'll enter your API keys securely via Telegram")
        print("   3. All keys will be encrypted before storage")
        print("   4. System will automatically start trading once configured")
        print("")
        print("🚀 Starting Telegram bootstrap mode...")
        print("📱 Check your Telegram for setup instructions!")
        print("")
        
        # Run bootstrap mode
        bootstrap_success = await run_bootstrap_mode(config)
        
        if bootstrap_success:
            print("✅ Bootstrap mode completed successfully!")
            print("🔑 API credentials have been encrypted and saved")
            print("🚀 System is now ready for auto-trading")
            return True
        else:
            print("❌ Bootstrap mode failed")
            print("   Please check Telegram bot configuration and try again")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Bootstrap mode interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Bootstrap mode error: {e}")
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
        print("🤖 ENHANCED BYBIT AUTO-TRADING SYSTEM")
        print("🚀 Automated Trading with Multi-Timeframe Analysis & Telegram Bootstrap")
        print("=" * 80)
        print("   ✅ Scheduled scanning (configurable intervals)")
        print("   ✅ Automated position management")  
        print("   ✅ Leverage-aware position sizing")
        print("   ✅ Profit-based auto-closing")
        print("   ✅ Multi-timeframe confirmation")
        print("   ✅ MySQL database storage")
        print("   ✅ Risk management controls")
        print("   ✅ Telegram bootstrap mode for API setup")
        print("")
        
        # Create or load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print("📄 Creating default configuration...")
            create_default_config_with_autotrading()
            print("⚠️  Please update the MySQL password and run again!")
            return
        
        # Load configuration from YAML and database
        db_config, config = load_configuration(config_path)
        
        if not db_config or not config:
            print("❌ Failed to load configuration")
            print(f"   Please ensure {config_path} exists and has valid database settings")
            return
        
        # print(f"⚙️  Database Configuration:")
        # print(f"   Host: {db_config.host}:{db_config.port}")
        # print(f"   Database: {db_config.database}")
        # print(f"   Username: {db_config.username}")
        # print("")
        
        # Setup database with auto-trading support
        if not setup_database_with_autotrading(db_config):
            print("❌ Cannot proceed without database connection")
            return
        
        # Check if bootstrap mode is needed
        if check_bootstrap_needed(config):
            print("🔑 API credentials missing - entering bootstrap mode")
            
            # Run bootstrap mode
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                bootstrap_success = loop.run_until_complete(handle_bootstrap_mode(config))
                
                if not bootstrap_success:
                    print("❌ Bootstrap mode failed - cannot start auto-trading")
                    return
                
                # Reload configuration after bootstrap
                print("🔄 Reloading configuration after bootstrap...")
                db_config, config = load_configuration(config_path)
                
            finally:
                loop.close()
        
        # Display configuration
        display_auto_trading_config(config)
        
        # Validate auto-trading configuration (skip API check since bootstrap completed)
        if not validate_auto_trading_config(config, skip_api_check=False):
            print("❌ Auto-trading configuration validation failed!")
            print("   Please fix the configuration issues and try again.")
            return
        
        # print("✅ Auto-trading configuration validated successfully!")
        print("")
        
        # Initialize and start auto-trader
        print("🚀 Initializing Auto-Trading System...")
        auto_trader = AutoTrader(config)
        
        # Test exchange connection
        if not auto_trader.trading_system.exchange_manager.test_connection():
            print("❌ Failed to connect to Bybit exchange")
            print("   Please check your API credentials and network connection")
            return
        
        # print("✅ Exchange connection successful!")
        print("")
        
        # Start auto-trading
        print("🤖 Starting Automated Trading Session...")
        print("   📱 You'll receive Telegram notifications for all trades")
        print("   Press Ctrl+C to stop the auto-trader")
        print("=" * 50)
        
        # Run main trading loop
        auto_trader.main_trading_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Auto-trader stopped by user")
    except Exception as e:
        print(f"\n❌ Auto-trader failed: {e}")
        import traceback
        traceback.print_exc()


def run_single_scan():
    """Run a single scan without starting the auto-trading loop (for testing)"""
    try:
        print("🔍 SINGLE SCAN MODE - Testing Auto-Trading System")
        print("=" * 50)
        
        # Load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print("❌ Configuration file not found. Run main() first.")
            return
        
        db_config, config = load_configuration(config_path)
        if not db_config or not config:
            print("❌ Failed to load configuration")
            return
        
        # Check for bootstrap requirement
        if check_bootstrap_needed(config):
            print("❌ Bootstrap mode required - API credentials missing")
            print("   Run main() first to configure API keys via Telegram")
            return
        
        # Validate configuration
        if not validate_auto_trading_config(config):
            print("❌ Configuration validation failed!")
            return
        
        # Initialize auto-trader
        auto_trader = AutoTrader(config)
        
        # Test connection
        if not auto_trader.trading_system.exchange_manager.test_connection():
            print("❌ Exchange connection failed!")
            return
        
        print("✅ Running single scan and execution test...")
        
        # FIXED: Run single scan with proper results display
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the complete analysis (same as the auto-trader uses)
            print("📊 Running complete multi-timeframe analysis...")
            results = auto_trader.trading_system.run_complete_analysis_parallel_mtf()
            
            if results:
                # Display the comprehensive results table
                print("\n" + "=" * 80)
                auto_trader.trading_system.print_comprehensive_results_with_mtf(results)
                print("=" * 80)
                
                # Show execution simulation
                if results.get('top_opportunities'):
                    signals_count = len(results.get('signals', []))
                    opportunities_count = len(results['top_opportunities'])
                    
                    print(f"\n📊 SCAN SIMULATION RESULTS:")
                    print(f"   Total Signals Generated: {signals_count}")
                    print(f"   Top Opportunities: {opportunities_count}")
                    print(f"   Would Execute: {min(config.max_execution_per_trade, opportunities_count)} trades")
                    print(f"   Available Position Slots: {config.max_concurrent_positions}")
                    print(f"   Risk per Trade: {config.risk_amount}% of account balance")
                    
                    # Show top opportunities that would be executed
                    execution_count = min(config.max_execution_per_trade, opportunities_count)
                    if execution_count > 0:
                        print(f"\n🎯 TRADES THAT WOULD BE EXECUTED:")
                        for i, opp in enumerate(results['top_opportunities'][:execution_count]):
                            print(f"   {i+1}. {opp['symbol']} {opp['side'].upper()} - {opp['confidence']:.1f}% confidence - MTF: {opp.get('mtf_status', 'N/A')}")
                else:
                    print(f"\n📊 SCAN RESULTS:")
                    print(f"   No trading opportunities found in this scan")
                
            else:
                print("❌ No analysis results generated")
            
        finally:
            loop.close()
        
        print("✅ Single scan test completed successfully!")
        
    except Exception as e:
        print(f"❌ Single scan failed: {e}")
        import traceback
        traceback.print_exc()


async def test_bootstrap():
    """Test bootstrap mode functionality"""
    try:
        print("🧪 BOOTSTRAP MODE TEST")
        print("=" * 30)
        
        # Load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print("❌ Configuration file not found")
            return
        
        db_config, config = load_configuration(config_path)
        if not db_config or not config:
            print("❌ Failed to load configuration")
            return
        
        # Test bootstrap functionality
        bootstrap_manager = TelegramBootstrapManager(config)
        
        print(f"Bootstrap needed: {bootstrap_manager.should_enter_bootstrap_mode()}")
        
        cred_status = bootstrap_manager.check_api_credentials()
        print(f"Credentials status: {cred_status}")
        
        if bootstrap_manager.should_enter_bootstrap_mode():
            print("🔄 Would enter bootstrap mode...")
            print("📱 Check Telegram for configuration instructions")
        else:
            print("✅ All credentials configured - no bootstrap needed")
        
    except Exception as e:
        print(f"❌ Bootstrap test failed: {e}")


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
                    print("❌ Failed to load configuration")
            except Exception as e:
                print(f"❌ Error showing configuration: {e}")
    else:
        # Run full auto-trading system
        main()