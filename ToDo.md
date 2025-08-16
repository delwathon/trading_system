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


Modify system_config databse table by:
    - Remove bybit_live_api_key
    - Remove bybit_live_api_secret
    - Add deposit_addresses [USDT TRC20, USDT ERC20, USDT BEP20]
    - Add subsciption_fee with default of $100
    
When sandbox mode is false, then it uses the api credential (bybit_test_api_key and bybit_test_api_secret) to trade for admin only

Ensure deposit addresses are also configured by admin during bootstrap setup using the telegram bot inline keyboard. Better still, use the provided bybit live api credentials to automatically fetch the deposit addresses, and run a confirmation with me by returning the respective fetched addresses before the confirm and inline keyboard button. I prefer this approach

User Table Structure
id
telegram_id (unique)
telegram_username (telegram username)
exchange(binance, bitunix, bybit, kucoin, weex)
api_key
api_secret
tier (admin, free, paid, awoof)
subscription_expires_at:
total_trades = Column(Integer, default=0)
max_daily_trades = (Unlimited for admin, 0 for free users, 5 for paid and awoof users)
successful_trades = Column(Integer, default=0)
total_pnl = Column(Float, default=0.0)
is_active = Column(Boolean, default=True)
is_banned = Column(Boolean, default=False)
ban_reason = Column(String(255), nullable=True)
created_at
updated_at


During bootstrap mode setup, when migrating and populating system_config, pre-populate user's table with admin data
telegram_id: 6708641837
telegram_username: SmartMoneyTraderAdmin
exchange: bybit
api_key: (My system_config.bybit_live_api_key goes here)
api_secret: (My system_config.bybit_live_api_secret goes here)
tier: admin (There can only be one admin and that's me)
subscription_expires_at: (usually a month)
is_active = True
is_banned = False
ban_reason = NULL

Other users registration is allowed via telegram only using the /start command

Every other bot interaction MUST be via the use of Telegram Inline Keyboard

All registered users are firstly registered as a free user.
Then they can use the upgrade inline keyboard to pay into any of the deposit addresses
They can as well use another inline keyboard to verify payment automatically by submitting the blockchain transaction hash. The system automatically verify the payment with blockchain and ensures it was sent to either of the deposit addresses. Amount is allowed to be greater than subscription_fee to upgrade a user from free to paid, but cannot be lesser.

Admin is the only one who has the privilege to upgrade any user to awoof tier.

Upon subscription expiry, users are automatically degraded to free user tier

All users uses the same trading configuration as stored in system_config, user independent configuration not allowed.

BEFORE IMPLEMETATION:
Could you confirm how the bootstrap mode works and how admin credentials is currently setup

IMPORTANT NOTE:
Do not create new patch files. 
Rather, modify the current files by updating them with the tasks requested above. This is VERY important.