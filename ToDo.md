- Change CSV and configuration to database -- DONE
- Change html visualization to JPG         -- DONE
- Telegram Notification for myself only    -- DONE
- Price decimal point per symbol
- Auto trading using max leverage & 5%     -- DONE
- During startup, take note of active positions and orders.... if new signal generated is already present in current position, do not place that order, take on the next signal instead.
- Ensure to update db at every closed trade, be it manual or tp, or auto-close at profit level... So that active positions and orders state can be updated
- Among the top generated signal, only place order for top max_execution_per_trade the ones that has the potential of hitting auto_close_profit_at 
- Store only the top selected signals in database.... Not the entire signals generated.
- If at any point in time, all confirmation timeframes are satisfactory, place order immediately even before the next symbol analysis




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




I want us to incorporate user management into the system. As the super user, my details are already stored in the database, inside_system config table. I do not want to share the same table with other users.

This system is going to be a paid system for other users.

There are 4 categories of users:

sapphire, platinum, premium, awoof