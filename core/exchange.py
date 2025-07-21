"""
Exchange connection and data fetching for the Enhanced Bybit Trading System.
"""

import ccxt
import pandas as pd
import logging
from typing import Dict, List, Optional
from config.config import EnhancedSystemConfig


class ExchangeManager:
    """Handles Bybit exchange connection and data fetching"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchange = self.setup_exchange()
        
    def setup_exchange(self):
        """Setup Bybit exchange connection"""
        try:
            exchange_config = {
                'enableRateLimit': True,
                'rateLimit': 1000,
                'timeout': self.config.api_timeout,
                'sandbox': self.config.sandbox_mode
            }
            
            if self.config.api_key and self.config.api_secret:
                if self.config.sandbox_mode:
                    exchange_config.update({
                        'apiKey': self.config.demo_api_key,
                        'secret': self.config.demo_api_secret
                    })
                else:
                    exchange_config.update({
                        'apiKey': self.config.api_key,
                        'secret': self.config.api_secret
                    })
            
            exchange = ccxt.bybit(exchange_config)
            exchange.load_markets()
            
            mode = "DEMO" if self.config.sandbox_mode else "PRODUCTION"
            self.logger.debug(f"✅ Connected to Bybit {mode}")
            
            return exchange
            
        except Exception as e:
            self.logger.error(f"Exchange setup failed: {e}")
            return None
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = None, limit: int = None) -> pd.DataFrame:
        """Fetch OHLCV data with database-configured limit"""
        if timeframe is None:
            timeframe = self.config.timeframe
            
        # Use database-configured limit if not specified
        if limit is None:
            limit = self.config.ohlcv_limit_primary
            
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch {symbol} data: {e}")
            return pd.DataFrame()
    
    def get_top_symbols(self) -> List[Dict]:
        """Get top trading symbols by volume"""
        try:
            tickers = self.exchange.fetch_tickers()
            
            # Filter for USDT perpetual contracts
            usdt_symbols = []
            for symbol, ticker in tickers.items():
                if '/USDT:USDT' in symbol and ticker.get('quoteVolume', 0) >= self.config.min_volume_24h:
                    usdt_symbols.append({
                        'symbol': symbol,
                        'volume_24h': ticker['quoteVolume'],
                        'price_change_24h': ticker.get('percentage', 0),
                        'current_price': ticker.get('last', 0)
                    })
            
            # Sort by volume and limit
            usdt_symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
            return usdt_symbols[:self.config.max_symbols_scan]
            
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            return []