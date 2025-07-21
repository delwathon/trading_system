#!/usr/bin/env python3
"""
Main entry point for the Enhanced Bybit Trading System.
Updated to use MySQL database storage and 15m → 1h/4h timeframe configuration.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import EnhancedSystemConfig, DatabaseConfig
from core.system import CompleteEnhancedBybitSystem
from database.models import DatabaseManager


def create_default_config_with_mysql():
    """Create default configuration file with MySQL settings"""
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
        }
    }
    
    config_path = 'enhanced_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📄 Created MySQL configuration file: {config_path}")
    print("⚠️  Please update the MySQL password in the configuration file!")
    return config_path


def setup_database(db_config: DatabaseConfig) -> bool:
    """Setup and test MySQL database connection"""
    try:
        print("🔗 Testing MySQL database connection...")
        
        # Test connection
        db_manager = DatabaseManager(db_config.get_database_url())
        if not db_manager.test_connection():
            print("❌ Database connection failed!")
            print(f"   Please ensure MySQL is running at {db_config.host}:{db_config.port}")
            print(f"   Database: {db_config.database}")
            print(f"   Username: {db_config.username}")
            return False
        
        # Create tables
        print("📊 Creating database tables...")
        db_manager.create_tables()
        
        print("✅ MySQL database setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def main():
    """Main execution function with MySQL database and 15m→1h/4h timeframes"""
    try:
        print("🌟 ENHANCED BYBIT TRADING SYSTEM WITH MYSQL DATABASE")
        print("🔧 Database-First Architecture with Multi-Timeframe Confirmation:")
        print("   ✅ Configurable Primary & Confirmation Timeframes")
        print("   ✅ MySQL Database Storage (Replaces CSV)")
        print("   ✅ Enhanced Technical Analysis (FIXED Indicators)")
        print("   ✅ Volume Profile Analysis")
        print("   ✅ Fibonacci & Confluence Analysis")
        print("   ✅ Interactive TradingView-style Charts")
        print("   ✅ Multi-Timeframe Signal Confirmation")
        print("")
        
        # Create or load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print("📄 Creating default MySQL configuration...")
            create_default_config_with_mysql()
            print("⚠️  Please update the MySQL settings in enhanced_config.yaml and run again!")
            return
        
        # Load database configuration from YAML
        db_config = EnhancedSystemConfig.from_yaml_file(config_path)
        
        # print(f"⚙️  Database Configuration:")
        # print(f"   Type: MySQL")
        # print(f"   Host: {db_config.host}:{db_config.port}")
        # print(f"   Database: {db_config.database}")
        # print(f"   Username: {db_config.username}")
        
        # Setup database
        if not setup_database(db_config):
            print("❌ Cannot proceed without database connection")
            return
        
        # Load system configuration from database
        # print("📊 Loading system configuration from MySQL database...")
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        print(f"⚙️  Trading Configuration (from database):")
        print(f"   Mode: {'DEMO' if config.sandbox_mode else 'PRODUCTION'}")
        print(f"   Primary Timeframe: {config.timeframe}")
        print(f"   Confirmation Timeframes: {', '.join(config.confirmation_timeframes)}")
        print(f"   MTF Confirmation: {'Enabled' if config.mtf_confirmation_required else 'Disabled'}")
        if config.mtf_confirmation_required:
            print(f"   MTF Weight Multiplier: {config.mtf_weight_multiplier}x")
        print(f"   Min Volume: ${config.min_volume_24h:,}")
        print(f"   Max Symbols: {config.max_symbols_scan}")
        print(f"   Show Charts: {config.show_charts}")
        print(f"   Charts per Batch: {config.charts_per_batch}")
        print("")
        
        # Initialize complete enhanced system with MySQL
        # print("🚀 Initializing Enhanced System with MySQL Database and configurable MTF Analysis...")
        system = CompleteEnhancedBybitSystem(config)
        
        if not system.exchange_manager.exchange:
            print("❌ Failed to initialize exchange connection")
            return
        
        # print(f"Using parallel processing with {system.config.max_workers} threads and MySQL storage...")
        results = system.run_complete_analysis_parallel_mtf()

        if results and results.get('signals'):
            system.print_comprehensive_results_with_mtf(results)
            
            # Save results to MySQL database (replaces CSV export)
            # print("\n💾 Saving results to MySQL database...")
            database_results = system.save_results_to_database(results)
            
            if 'error' not in database_results:
                # print(f"✅ Database Save Complete:")
                # print(f"   Scan ID: {database_results.get('scan_id', 'Unknown')}")
                # print(f"   Signals Saved: {database_results.get('signals', 0)}")
                # print(f"   Opportunities Saved: {database_results.get('opportunities', 0)}")
                # print(f"   Market Summary: {'✅' if database_results.get('market_summary') else '❌'}")
                
                # Show database statistics
                db_stats = system.enhanced_db_manager.get_database_stats()
                # print(f"\n📊 Database Statistics:")
                # print(f"   Total Scan Sessions: {db_stats.get('scan_sessions', 0)}")
                # print(f"   Total Trading Signals: {db_stats.get('trading_signals', 0)}")
                # print(f"   Total Opportunities: {db_stats.get('trading_opportunities', 0)}")
                
                # MTF Performance Report
                timeframe_display = f"{config.timeframe}→{'/'.join(config.confirmation_timeframes)}"
                # print(f"\n🔍 Multi-Timeframe Performance (Last 7 Days):")
                mtf_report = system.enhanced_db_manager.get_mtf_performance_report(7)
                # if mtf_report:
                    # print(f"   Total Signals: {mtf_report.get('total_signals', 0)}")
                    # print(f"   Timeframes: {timeframe_display}")
                    # mtf_breakdown = mtf_report.get('mtf_breakdown', {})
                    # print(f"   Strong Confirmation: {mtf_breakdown.get('STRONG', 0)}")
                    # print(f"   Partial Confirmation: {mtf_breakdown.get('PARTIAL', 0)}")
                    # print(f"   No Confirmation: {mtf_breakdown.get('NONE', 0)}")
            else:
                print(f"❌ Database save failed: {database_results['error']}")
            
            # Additional system information
            scan_info = results['scan_info']
            performance = results.get('system_performance', {})
            
            # print(f"\n📊 System Performance:")
            # print(f"   Execution Time: {scan_info.get('execution_time_seconds', 0):.1f}s")
            # print(f"   Symbols Analyzed: {scan_info.get('symbols_analyzed', 0)}")
            # print(f"   Signals Generated: {scan_info.get('signals_generated', 0)}")
            # print(f"   Success Rate: {scan_info.get('success_rate', 0):.1f}%")
            # print(f"   Charts Generated: {scan_info.get('charts_generated', 0)}")
            # print(f"   Speedup Factor: {performance.get('speedup_factor', 1):.1f}x")
            
            mtf_dist = performance.get('mtf_distribution', {})
            if mtf_dist:
                total_confirmation_timeframes = len(config.confirmation_timeframes)
                # print(f"\n🔍 Multi-Timeframe Analysis Results:")
                # print(f"   Strong Confirmation ({total_confirmation_timeframes}/{total_confirmation_timeframes}): {mtf_dist.get('strong_confirmation', 0)}")
                # print(f"   Partial Confirmation (≥{max(1, total_confirmation_timeframes//2)}/{total_confirmation_timeframes}): {mtf_dist.get('partial_confirmation', 0)}")
                # print(f"   No Confirmation (0/{total_confirmation_timeframes}): {mtf_dist.get('no_confirmation', 0)}")
                # print(f"   Confirmation Rate: {mtf_dist.get('confirmation_rate', 0):.1f}%")

            return results
        else:
            print("\n⚠️  No signals generated in this scan")
            return None
            
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()