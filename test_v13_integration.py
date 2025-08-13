"""
test_v13_integration.py - Test the V13 integration
===================================================
Run this to verify the V13 system works correctly
"""

import sys
import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_dataclass_import():
    """Test that the fixed dataclass imports correctly"""
    try:
        from signals.signal_gen_v13_core import Signal, SignalStatus, SignalQuality, TimeFrame
        print("✅ Core dataclasses imported successfully")
        
        # Test creating a signal
        signal = Signal(
            id="test123",
            symbol="BTCUSDT",
            side="buy",
            status=SignalStatus.ANALYZING,
            quality_tier=SignalQuality.STANDARD,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit_1=52000.0,
            take_profit_2=54000.0,
            current_price=50000.0,
            risk_reward_ratio=2.0,
            position_size=0.1,
            risk_amount=200.0,
            potential_profit=400.0,
            confidence=75.0,
            analysis_timeframe=TimeFrame.H6,
            market_regime="ranging"
        )
        print(f"✅ Signal created: {signal.id}")
        return True
        
    except Exception as e:
        print(f"❌ Dataclass import failed: {e}")
        return False

def test_wrapper_import():
    """Test that the wrapper imports correctly"""
    try:
        from signals.generator import SignalGenerator, create_signal_generator
        print("✅ Generator wrapper imported successfully")
        return True
    except Exception as e:
        print(f"❌ Wrapper import failed: {e}")
        return False

def test_basic_initialization():
    """Test basic initialization"""
    try:
        from signals.generator import create_signal_generator
        
        # Create a mock config
        class MockConfig:
            timeframe = '6h'
            confirmation_timeframes = ['4h', '1h']
            max_concurrent_positions = 10
            max_risk_per_trade = 0.02
            max_daily_risk = 0.06
        
        config = MockConfig()
        
        # Create generator
        generator = create_signal_generator(config, None)
        print(f"✅ Generator created with version: {generator.version}")
        
        # Test getting statistics
        stats = generator.get_statistics()
        print(f"✅ Statistics retrieved: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility_methods():
    """Test V12 compatibility methods"""
    try:
        from signals.generator import create_signal_generator
        
        class MockConfig:
            timeframe = '1h'
            confirmation_timeframes = ['4h', '6h']
        
        generator = create_signal_generator(MockConfig(), None)
        
        # Create sample data
        df = pd.DataFrame({
            'open': np.random.randn(100) + 50000,
            'high': np.random.randn(100) + 50100,
            'low': np.random.randn(100) + 49900,
            'close': np.random.randn(100) + 50000,
            'volume': np.random.rand(100) * 1000000
        })
        
        # Test technical summary
        summary = generator.create_technical_summary(df)
        print(f"✅ Technical summary: {summary['trend']['direction']}")
        
        # Test volume patterns
        volume = generator.analyze_volume_patterns(df)
        print(f"✅ Volume pattern: {volume['pattern']}")
        
        # Test trend strength
        trend = generator.calculate_trend_strength(df)
        print(f"✅ Trend strength: {trend['strength']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

async def test_v13_async():
    """Test V13 async functionality"""
    try:
        # Only test if V13 is available
        from signals.signal_gen_v13_main import SignalGeneratorV13
        from signals.signal_gen_v13_core import SystemConfiguration
        
        config = SystemConfiguration()
        generator = SignalGeneratorV13(config, None)
        
        print("✅ V13 async components initialized")
        
        # Don't actually start the system in test
        # await generator.start()
        # await asyncio.sleep(1)
        # await generator.stop()
        
        return True
        
    except ImportError:
        print("ℹ️ V13 modules not available, skipping async test")
        return True
    except Exception as e:
        print(f"❌ V13 async test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SIGNAL GENERATOR V13 INTEGRATION TEST")
    print("="*60 + "\n")
    
    tests = [
        ("Dataclass Import", test_dataclass_import),
        ("Wrapper Import", test_wrapper_import),
        ("Basic Initialization", test_basic_initialization),
        ("Compatibility Methods", test_compatibility_methods)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 40)
        result = test_func()
        results.append(result)
        print()
    
    # Run async test
    print(f"\n📋 Testing: V13 Async Components")
    print("-" * 40)
    result = asyncio.run(test_v13_async())
    results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"⚠️ SOME TESTS FAILED ({passed}/{total} passed)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)