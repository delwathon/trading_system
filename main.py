#!/usr/bin/env python3
"""
Main entry point for the Enhanced Bybit Trading System.
Modular version with complete restructuring.
"""

import sys
import os
import yaml
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import EnhancedSystemConfig
from core.system import CompleteEnhancedBybitSystem


def create_default_config_with_mtf():
    """Create default configuration file with MTF settings and indicator parameters"""
    config = {
        'api_key': None,
        'api_secret': None,
        'sandbox_mode': True,
        'min_volume_24h': 10_000_000,
        'max_symbols_scan': 50,
        'timeframe': '1h',  # Primary timeframe
        
        # Multi-Timeframe Configuration
        'confirmation_timeframes': ['4h', '6h'],  # Configurable confirmation timeframes
        'mtf_confirmation_required': True,        # Enable/disable MTF confirmation
        'mtf_weight_multiplier': 1.5,            # Confidence boost multiplier
        
        'max_requests_per_second': 8.0,
        'api_timeout': 20000,
        'max_portfolio_risk': 0.02,
        'max_daily_trades': 20,
        'max_single_position_risk': 0.005,
        'max_workers': None,
        'cache_ttl_seconds': 60,
        'max_cache_size': 1000,
        'ml_training_samples': 400,
        'ml_profitable_rate': 0.45,
        'csv_base_filename': 'enhanced_bybit_signals_mtf',
        'show_charts': True,
        'save_charts': True,
        'charts_per_batch': 5,
        'chart_width': 1400,
        'chart_height': 800,
        
        # Indicator Parameters
        'stoch_rsi_window': 14,
        'stoch_rsi_smooth_k': 3,
        'stoch_rsi_smooth_d': 3,
        'ichimoku_window1': 9,
        'ichimoku_window2': 26,
        'ichimoku_window3': 52
    }
    
    config_path = 'enhanced_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📄 Created configuration file with MTF settings and indicator parameters: {config_path}")
    return config_path


def main():
    """Main execution function with MTF support and FIXED indicators"""
    try:
        print("🌟 ENHANCED BYBIT TRADING SYSTEM WITH MULTI-TIMEFRAME CONFIRMATION")
        print("🔧 Modular Architecture - All 4 Analysis Approaches + Multi-Timeframe Confirmation + FIXED Indicators:")
        print("   ✅ Enhanced Technical Analysis (50+ indicators)")
        print("   ✅ FIXED Stochastic RSI Implementation")
        print("   ✅ FIXED Ichimoku Cloud Implementation")
        print("   ✅ Volume Profile Analysis (POC, Value Area, HVN/LVN)")
        print("   ✅ Fibonacci & Confluence Analysis (Multi-method)")
        print("   ✅ Interactive TradingView-style Charts")
        print("   ✅ Multi-Timeframe Signal Confirmation")
        print("   ✅ Modular Architecture")
        print("")
        
        # Create or load configuration
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print("📄 Creating default configuration with MTF settings and indicator parameters...")
            create_default_config_with_mtf()
        
        # Load configuration
        config = EnhancedSystemConfig.from_file(config_path)
        
        print(f"⚙️  Configuration Loaded:")
        print(f"   Mode: {'TESTNET' if config.sandbox_mode else 'PRODUCTION'}")
        print(f"   Primary Timeframe: {config.timeframe}")
        print(f"   Confirmation Timeframes: {', '.join(config.confirmation_timeframes)}")
        print(f"   MTF Confirmation: {'Enabled' if config.mtf_confirmation_required else 'Disabled'}")
        if config.mtf_confirmation_required:
            print(f"   MTF Weight Multiplier: {config.mtf_weight_multiplier}x")
        print(f"   Min Volume: ${config.min_volume_24h:,}")
        print(f"   Max Symbols: {config.max_symbols_scan}")
        print(f"   Show Charts: {config.show_charts}")
        print(f"   Charts per Batch: {config.charts_per_batch}")
        print(f"   Stochastic RSI Window: {config.stoch_rsi_window}")
        print(f"   Ichimoku Windows: {config.ichimoku_window1}/{config.ichimoku_window2}/{config.ichimoku_window3}")
        print("")
        
        # Initialize complete enhanced system with MTF
        print("🚀 Initializing Enhanced System with Multi-Timeframe Analysis and FIXED Indicators...")
        system = CompleteEnhancedBybitSystem(config)
        
        if not system.exchange_manager.exchange:
            print("❌ Failed to initialize exchange connection")
            return
        
        print(f"Using parallel processing with {system.config.max_workers} threads and MTF confirmation...")
        results = system.run_complete_analysis_parallel_mtf()

        if results and results.get('signals'):
            system.print_comprehensive_results_with_mtf(results)
            
            # Export results to CSV
            system.export_results_to_csv(results)
            
            # Additional MTF-specific information
            scan_info = results['scan_info']
            performance = results.get('system_performance', {})
            
            print(f"\n📊 System Performance:")
            print(f"   Execution Time: {scan_info.get('execution_time_seconds', 0):.1f}s")
            print(f"   Symbols Analyzed: {scan_info.get('symbols_analyzed', 0)}")
            print(f"   Signals Generated: {scan_info.get('signals_generated', 0)}")
            print(f"   Success Rate: {scan_info.get('success_rate', 0):.1f}%")
            print(f"   Charts Generated: {scan_info.get('charts_generated', 0)}")
            print(f"   Speedup Factor: {performance.get('speedup_factor', 1):.1f}x")
            
            mtf_dist = performance.get('mtf_distribution', {})
            if mtf_dist:
                print(f"\n🔍 Multi-Timeframe Analysis:")
                print(f"   Strong Confirmation: {mtf_dist.get('strong_confirmation', 0)}")
                print(f"   Partial Confirmation: {mtf_dist.get('partial_confirmation', 0)}")
                print(f"   No Confirmation: {mtf_dist.get('no_confirmation', 0)}")
                print(f"   Confirmation Rate: {mtf_dist.get('confirmation_rate', 0):.1f}%")

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