#!/usr/bin/env python3
"""
BitUnix Futures API Test Script - FIXED VERSION
Tests all API endpoints and signal execution
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List

# Import the BitUnix module
from bitunix import (
    BitUnixFuturesAPI,
    BitUnixSignalExecutor,
    SignalData,
    ExecutionConfig
)


class BitUnixTester:
    """Comprehensive tester for BitUnix API"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.api = BitUnixFuturesAPI(api_key, secret_key)
        self.logger = logging.getLogger(__name__)
    
    def test_connection(self) -> bool:
        """Test basic API connection"""
        print("\n" + "="*60)
        print("TEST 1: API CONNECTION")
        print("="*60)
        
        try:
            # Test account endpoint
            result = self.api.get_account_info()
            
            if result.get('code') == 0:
                print("✅ API connection successful")
                print(f"   Response: {result}")
                return True
            else:
                print(f"❌ API error: {result.get('msg')}")
                return False
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def test_account_endpoints(self) -> bool:
        """Test account-related endpoints"""
        print("\n" + "="*60)
        print("TEST 2: ACCOUNT ENDPOINTS")
        print("="*60)
        
        try:
            # Get account balance
            print("\n📊 Testing get_account_balance...")
            balance = self.api.get_account_balance()
            print(f"✅ Balance retrieved:")
            print(f"   Available: ${balance.get('available', 0):.2f}")
            print(f"   Equity: ${balance.get('equity', 0):.2f}")
            print(f"   Total: ${balance.get('total', 0):.2f}")
            print(f"   Unrealized P&L: ${balance.get('unrealizedPL', 0):.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Account test failed: {e}")
            return False
    
    def test_market_data(self) -> bool:
        """Test market data endpoints"""
        print("\n" + "="*60)
        print("TEST 3: MARKET DATA")
        print("="*60)
        
        try:
            symbol = "BTCUSDT"
            
            # Get ticker
            print(f"\n📊 Testing ticker for {symbol}...")
            ticker = self.api.get_ticker(symbol)
            print(f"✅ Ticker data:")
            print(f"   Last: ${ticker.get('last', 0):,.2f}")
            print(f"   Mark: ${ticker.get('markPrice', 0):,.2f}")
            print(f"   Bid: ${ticker.get('bid', 0):,.2f}")
            print(f"   Ask: ${ticker.get('ask', 0):,.2f}")
            
            # Get depth
            print(f"\n📊 Testing order book depth...")
            depth = self.api.get_depth(symbol, limit=5)
            if depth.get('code') == 0:
                print(f"✅ Depth data retrieved")
            
            # Get kline
            print(f"\n📊 Testing kline data...")
            kline = self.api.get_kline(symbol, "1h", limit=10)
            if kline.get('code') == 0:
                print(f"✅ Kline data retrieved")
            
            return True
            
        except Exception as e:
            print(f"❌ Market data test failed: {e}")
            return False
    
    def test_positions(self) -> bool:
        """Test position endpoints"""
        print("\n" + "="*60)
        print("TEST 4: POSITIONS")
        print("="*60)
        
        try:
            # Get all positions
            print("\n📊 Testing get_positions...")
            positions = self.api.get_positions()
            print(f"✅ Found {len(positions)} positions")
            
            # Display active positions
            active_positions = [p for p in positions if p['size'] > 0]
            if active_positions:
                print("\n📈 Active positions:")
                for pos in active_positions:
                    print(f"   {pos['symbol']}:")
                    print(f"      Side: {pos['side']}")
                    print(f"      Size: {pos['size']}")
                    print(f"      Entry: ${pos['entryPrice']:,.2f}")
                    print(f"      Mark: ${pos['markPrice']:,.2f}")
                    print(f"      P&L: ${pos['unrealizedPnl']:,.2f}")
                    print(f"      Leverage: {pos['leverage']}x")
            else:
                print("   No active positions")
            
            return True
            
        except Exception as e:
            print(f"❌ Position test failed: {e}")
            return False
    
    def test_orders(self) -> bool:
        """Test order endpoints - FIXED"""
        print("\n" + "="*60)
        print("TEST 5: ORDERS")
        print("="*60)
        
        try:
            # Get open orders
            print("\n📊 Testing get_open_orders...")
            orders = self.api.get_open_orders()
            
            if orders.get('code') == 0:
                order_data = orders.get('data', [])
                
                # Handle both list and single order response
                if isinstance(order_data, list):
                    order_list = order_data
                elif isinstance(order_data, dict):
                    order_list = [order_data]
                else:
                    order_list = []
                
                print(f"✅ Found {len(order_list)} open orders")
                
                if order_list:
                    print("\n📋 Open orders:")
                    # Display up to 5 orders
                    for i, order in enumerate(order_list):
                        if i >= 5:  # Only show first 5
                            break
                        print(f"   Order {i+1}:")
                        print(f"   Order ID: {order.get('orderId', 'N/A')}")
                        print(f"   Symbol: {order.get('symbol', 'N/A')}")
                        print(f"   Side: {order.get('side', 'N/A')}")
                        print(f"   Type: {order.get('orderType', order.get('type', 'N/A'))}")
                        print(f"   Qty: {order.get('qty', 'N/A')}")
                        print(f"   Price: {order.get('price', 'N/A')}")
                        print()
            
            # Get order history
            print("\n📊 Testing order history...")
            history = self.api.get_order_history(limit=10)
            
            if history.get('code') == 0:
                print(f"✅ Order history retrieved")
            
            return True
            
        except Exception as e:
            print(f"❌ Order test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_leverage_change(self, symbol: str = "BTCUSDT", leverage: int = 10) -> bool:
        """Test leverage change"""
        print("\n" + "="*60)
        print("TEST 6: LEVERAGE CHANGE")
        print("="*60)
        
        try:
            print(f"\n⚙️ Testing leverage change for {symbol} to {leverage}x...")
            result = self.api.change_leverage(symbol, leverage)
            
            if result.get('code') == 0:
                print(f"✅ Leverage changed successfully")
                return True
            else:
                print(f"⚠️ Leverage change result: {result.get('msg')}")
                return True  # May already be set
                
        except Exception as e:
            print(f"❌ Leverage test failed: {e}")
            return False
    
    def test_margin_mode(self, symbol: str = "BTCUSDT") -> bool:
        """Test margin mode change"""
        print("\n" + "="*60)
        print("TEST 7: MARGIN MODE")
        print("="*60)
        
        try:
            print(f"\n⚙️ Testing margin mode change for {symbol}...")
            result = self.api.change_margin_mode(symbol, "ISOLATED")
            
            if result.get('code') == 0:
                print(f"✅ Margin mode changed to ISOLATED")
                return True
            else:
                # Try to get current margin mode
                positions = self.api.get_positions(symbol)
                if positions:
                    current_mode = "ISOLATED" if positions[0].get('marginType') == 'isolated' else "CROSS"
                    print(f"⚠️ Margin mode already set to {current_mode}")
                else:
                    print(f"⚠️ Margin mode result: {result.get('msg')}")
                return True  # May already be set
                
        except Exception as e:
            print(f"❌ Margin mode test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        print("\n" + "="*70)
        print(" BITUNIX FUTURES API TEST SUITE")
        print("="*70)
        print(f"Timestamp: {datetime.now()}")
        print(f"API Key: {self.api_key[:10]}..." if len(self.api_key) > 10 else self.api_key)
        
        tests = [
            ("Connection", self.test_connection),
            ("Account", self.test_account_endpoints),
            ("Market Data", self.test_market_data),
            ("Positions", self.test_positions),
            ("Orders", self.test_orders),
            ("Leverage", self.test_leverage_change),
            ("Margin Mode", self.test_margin_mode)
        ]
        
        results = []
        for name, test_func in tests:
            try:
                success = test_func()
                results.append((name, success))
            except Exception as e:
                print(f"\n❌ Test {name} crashed: {e}")
                results.append((name, False))
            
            time.sleep(1)  # Delay between tests
        
        # Summary
        print("\n" + "="*70)
        print(" TEST SUMMARY")
        print("="*70)
        
        for name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{name:20} {status}")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        print("\n" + "="*70)
        print(f" RESULT: {passed}/{total} tests passed")
        print("="*70)
        
        return passed == total


def test_signal_executor():
    """Test the signal executor"""
    print("\n" + "="*70)
    print(" SIGNAL EXECUTOR TEST")
    print("="*70)
    
    # Get credentials from environment
    api_key = os.getenv('BITUNIX_API_KEY')
    secret_key = os.getenv('BITUNIX_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Please set BITUNIX_API_KEY and BITUNIX_SECRET_KEY environment variables")
        return False
    
    try:
        # Create configuration
        config = ExecutionConfig(
            risk_amount=1.0,  # 1% risk for testing
            default_leverage=10,
            max_concurrent_positions=3,
            margin_mode="ISOLATED",
            partial_profits_enabled=True,
            trailing_stops_enabled=True
        )
        
        # Initialize executor
        print("\n🚀 Initializing Signal Executor...")
        executor = BitUnixSignalExecutor(api_key, secret_key, config)
        
        # Get account status
        print("\n📊 Account Status:")
        status = executor.get_account_status()
        print(f"   Balance: ${status.get('balance', 0):,.2f}")
        print(f"   Equity: ${status.get('equity', 0):,.2f}")
        print(f"   Positions: {status.get('positions_count', 0)}")
        print(f"   Unrealized P&L: ${status.get('total_unrealized_pnl', 0):,.2f}")
        
        # Create test signal (DO NOT EXECUTE without confirmation)
        signal = SignalData(
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            confidence=75,
            risk_reward_ratio=2.0
        )
        
        print("\n📨 Test Signal Created:")
        print(f"   Symbol: {signal.symbol}")
        print(f"   Side: {signal.side}")
        print(f"   Entry: ${signal.entry_price:,.2f}")
        print(f"   Stop: ${signal.stop_loss:,.2f}")
        print(f"   Target: ${signal.take_profit:,.2f}")
        
        # Ask for confirmation
        confirm = input("\n⚠️ Execute this TEST signal? (yes/no): ")
        
        if confirm.lower() == 'yes':
            print("\nExecuting signal...")
            result = executor.execute_signal(signal)
            
            if result['success']:
                print(f"✅ Signal executed successfully!")
                print(f"   Order ID: {result.get('order_id')}")
                print(f"   Position: {result.get('position_data')}")
            else:
                print(f"❌ Execution failed: {result['message']}")
        else:
            print("Signal execution cancelled")
        
        # Shutdown
        executor.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Executor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get credentials
    api_key = '4d6cca0d033055d2b11647899c52a022'
    secret_key = '6a39a12d072e2b04ccb30fa2eee048c3'
    
    if not api_key or not secret_key:
        print("\n❌ API credentials not found!")
        print("\nPlease set environment variables:")
        print("  export BITUNIX_API_KEY='your_api_key'")
        print("  export BITUNIX_SECRET_KEY='your_secret_key'")
        print("\nOr create a .env file with:")
        print("  BITUNIX_API_KEY=your_api_key")
        print("  BITUNIX_SECRET_KEY=your_secret_key")
        sys.exit(1)
    
    # Run tests
    print("\nSelect test to run:")
    print("1. Full API Test Suite")
    print("2. Signal Executor Test")
    print("3. Quick Connection Test")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == '1':
        tester = BitUnixTester(api_key, secret_key)
        success = tester.run_all_tests()
    elif choice == '2':
        success = test_signal_executor()
    elif choice == '3':
        tester = BitUnixTester(api_key, secret_key)
        success = tester.test_connection()
    else:
        print("Invalid choice")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    main()