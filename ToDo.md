- Change CSV and configuration to database -- DONE
- Change html visualization to JPG         -- DONE
- Telegram Notification for myself only
- Price decimal point per symbol
- Auto trading using max leverage & 5%





# Scalping Setup
timeframe='1m'
confirmation_timeframes=['5m', '15m']

ohlcv_limit_primary = 100
ohlcv_limit_mtf = 50
ohlcv_limit_analysis = 100

# Day Trading Setup  
timeframe='15m'
confirmation_timeframes=['1h', '4h']

ohlcv_limit_primary = 500
ohlcv_limit_mtf = 200  
ohlcv_limit_analysis = 500

# Swing Trading Setup
timeframe='4h' 
confirmation_timeframes=['1d', '1w']

ohlcv_limit_primary = 1000
ohlcv_limit_mtf = 500
ohlcv_limit_analysis = 1000

# High-Confidence Setup
timeframe='15m'
confirmation_timeframes=['30m', '1h', '4h', '1d']  # 4 confirmations