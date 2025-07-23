#!/usr/bin/env python3
"""
Test Global Bybit API Endpoint Accessibility
Quick script to verify which Bybit API endpoints work from your location
"""

import requests
import time
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_api_endpoints():
    """Test various Bybit API endpoints to find which ones work"""
    print("🌍 BYBIT API ENDPOINT ACCESSIBILITY TEST")
    print("=" * 50)
    print("Testing which Bybit API endpoints are accessible from your location...")
    print()
    
    endpoints = {
        'Global API (Production)': 'https://api.bybitglobal.com/v2/public/time',
        'Standard API (Production)': 'https://api.bybit.com/v2/public/time', 
        'Demo API': 'https://api-demo.bybit.com/v2/public/time',
        'Testnet API': 'https://api-testnet.bybit.com/v2/public/time',
        'Global API v5': 'https://api.bybitglobal.com/v5/market/time',
        'Standard API v5': 'https://api.bybit.com/v5/market/time'
    }
    
    results = {}
    response_times = {}
    
    for name, url in endpoints.items():
        print(f"Testing {name}...")
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            response_times[name] = response_time
            
            if response.status_code == 200:
                results[name] = True
                print(f"   ✅ SUCCESS - Response time: {response_time:.0f}ms")
                
                # Try to parse JSON response to verify it's working properly
                try:
                    data = response.json()
                    if 'time_now' in data or 'result' in data:
                        print(f"   📊 API Response: Valid")
                    else:
                        print(f"   ⚠️  API Response: Unexpected format")
                except:
                    print(f"   ⚠️  API Response: Not JSON")
            else:
                results[name] = False
                print(f"   ❌ FAILED - Status code: {response.status_code}")
                
        except requests.exceptions.Timeout:
            results[name] = False
            print(f"   ❌ TIMEOUT - Request took longer than 10 seconds")
        except requests.exceptions.ConnectionError as e:
            results[name] = False
            print(f"   ❌ CONNECTION ERROR - {e}")
        except Exception as e:
            results[name] = False
            print(f"   ❌ ERROR - {e}")
        
        print()
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("📊 RESULTS SUMMARY:")
    print("=" * 30)
    
    working_endpoints = [name for name, status in results.items() if status]
    failed_endpoints = [name for name, status in results.items() if not status]
    
    if working_endpoints:
        print("✅ WORKING ENDPOINTS:")
        for endpoint in working_endpoints:
            speed = response_times.get(endpoint, 0)
            print(f"   • {endpoint} ({speed:.0f}ms)")
    
    if failed_endpoints:
        print("\n❌ FAILED ENDPOINTS:")
        for endpoint in failed_endpoints:
            print(f"   • {endpoint}")
    
    print("\n💡 RECOMMENDATION:")
    if 'Global API (Production)' in working_endpoints:
        print("✅ Use Global API (api.bybitglobal.com) - RECOMMENDED")
        print("   Your trading system will work without VPN!")
        return 'global'
    elif 'Standard API (Production)' in working_endpoints:
        print("✅ Use Standard API (api.bybit.com)")
        print("   Your current system should work as-is")
        return 'standard'
    else:
        print("❌ No working endpoints found")
        print("   Check your internet connection or try with VPN")
        return 'none'


def test_with_trading_system():
    """Test the actual trading system with Global API"""
    print("\n🔧 TESTING WITH TRADING SYSTEM:")
    print("=" * 35)
    
    try:
        # Import the updated exchange manager
        from core.exchange import ExchangeManager
        from config.config import DatabaseConfig, EnhancedSystemConfig
        
        print("Loading system configuration...")
        
        # Load configuration
        config_path = 'enhanced_config.yaml'
        if not Path(config_path).exists():
            print(f"❌ Configuration file not found: {config_path}")
            print("   Run main.py first to create the configuration")
            return False
        
        # Load configs
        db_config = DatabaseConfig.from_yaml_file(config_path)
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        print("✅ Configuration loaded")
        
        # Test exchange manager initialization
        print("Initializing exchange with Global API...")
        exchange_manager = ExchangeManager(config)
        
        if not exchange_manager.exchange:
            print("❌ Failed to initialize exchange")
            return False
        
        print("✅ Exchange initialized")
        
        # Get API info
        api_info = exchange_manager.get_api_info()
        print(f"📡 API Endpoint: {api_info.get('public_api', 'Unknown')}")
        print(f"🏠 Hostname: {api_info.get('hostname', 'Not set')}")
        
        # Test connection
        print("Testing API connection...")
        if exchange_manager.test_connection():
            print("✅ API connection successful!")
            
            # Get account info
            account_info = exchange_manager.get_account_info()
            if account_info:
                print(f"💰 Account Balance: {account_info.get('free_balance', 0)} USDT")
                print(f"📊 API Endpoint: {account_info.get('api_endpoint', 'Unknown')}")
            
            return True
        else:
            print("❌ API connection failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 BYBIT GLOBAL API CONFIGURATION TEST")
    print("=" * 50)
    print("This script will test API endpoint accessibility and")
    print("verify your trading system works with the Global API.")
    print()
    
    # Test endpoints
    recommended = test_api_endpoints()
    
    if recommended == 'global':
        print("\n🎯 NEXT STEPS:")
        print("1. Your system is configured for Global API (Production)")
        print("2. No VPN required!")
        print("3. Test with your credentials below...")
        
    elif recommended == 'demo':
        print("\n🎯 NEXT STEPS:")
        print("1. Your system is configured for Demo API")
        print("2. Perfect for testing without real money!")
        print("3. No VPN required!")
        print("4. Make sure sandbox_mode=True in your config")
        
        # Test with actual trading system
        success = test_with_trading_system()
        
        if success:
            print("\n🎉 SUCCESS!")
            print("✅ Demo API is working with your trading system")
            print("✅ You can run the demo trader without VPN")
            print("\nRun the following commands:")
            print("   python main.py config    # View configuration")  
            print("   python main.py test      # Test a single scan")
            print("   python main.py           # Start demo auto-trading")
        else:
            print("\n⚠️  PARTIAL SUCCESS:")
            print("✅ Demo API endpoints are accessible")
            print("❌ Trading system connection failed")
            print("\nThis could be due to:")
            print("   • Missing or invalid Demo API credentials")
            print("   • Database not configured")
            print("   • sandbox_mode not set to True")
            print("\nRun bootstrap setup to configure demo credentials:")
            print("   python setup_bootstrap.py")
            
    elif recommended == 'global':
        print("\n🎯 NEXT STEPS:")
        print("1. Your system is already configured for Global API")
        print("2. No VPN required!")
        print("3. Test with your credentials below...")
        
        # Test with actual trading system
        success = test_with_trading_system()
        
        if success:
            print("\n🎉 SUCCESS!")
            print("✅ Global API is working with your trading system")
            print("✅ You can run the auto-trader without VPN")
            print("\nRun the following commands:")
            print("   python main.py config    # View configuration")
            print("   python main.py test      # Test a single scan")
            print("   python main.py           # Start auto-trading")
        else:
            print("\n⚠️  PARTIAL SUCCESS:")
            print("✅ Global API endpoints are accessible")
            print("❌ Trading system connection failed")
            print("\nThis could be due to:")
            print("   • Missing or invalid API credentials")
            print("   • Database not configured")
            print("   • API permissions issues")
            print("\nRun bootstrap setup to configure credentials:")
            print("   python setup_bootstrap.py")
    
    elif recommended == 'standard':
        print("\n⚠️  STANDARD API WORKING:")
        print("Your current system should work, but consider:")
        print("1. Updating to Global API for better access")
        print("2. No VPN would be required with Global API")
    
    else:
        print("\n❌ NO WORKING ENDPOINTS:")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Try with VPN")
        print("3. Contact your ISP about API access")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()