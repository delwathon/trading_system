"""
Enhanced ExchangeManager with Global Bybit API support
UPDATED: Uses api.bybitglobal.com instead of api.bybit.com for regional access
FIXED: Better credential handling and validation with global endpoint
"""

import ccxt
import pandas as pd
import logging
import time
from typing import Dict, List, Optional
from config.config import EnhancedSystemConfig
from utils.encryption import SecretManager


class ExchangeManager:
    """Handles Bybit exchange connection with Global API endpoint and encrypted API secrets"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.secret_manager = SecretManager.from_config(config)
        self.exchange = self.setup_exchange()
        
    def setup_exchange(self):
        """Setup Bybit exchange connection with decrypted API secrets and optimized connection pool"""
        try:
            exchange_config = {
                'enableRateLimit': True,
                'rateLimit': int(1000 / self.config.max_requests_per_second),  # Dynamic rate limiting
                'timeout': self.config.api_timeout,
                'sandbox': self.config.sandbox_mode,
                # FIXED: Connection pool optimization
                'options': {
                    'defaultType': 'linear',  # Use linear (USDT) contracts by default
                    'createMarketBuyOrderRequiresPrice': False,
                    # Connection pool settings
                    'maxRetries': 3,
                    'retryDelay': 1000,
                    # Session optimization
                    'keepAlive': True,
                }
            }
            
            # Test encryption/decryption first
            if not self.secret_manager.test_encryption_decryption():
                self.logger.error("❌ Encryption/decryption test failed")
                return None
            
            # Determine which credentials to use
            if self.config.sandbox_mode:
                # Use demo credentials for sandbox mode
                encrypted_api = self.config.demo_api_key
                encrypted_secret = self.config.demo_api_secret
                self.logger.debug("Using demo API credentials for sandbox mode")
            else:
                # Use production credentials
                encrypted_api = self.config.api_key
                encrypted_secret = self.config.api_secret
                self.logger.debug("Using production API credentials")
            
            # Validate credentials exist
            if not encrypted_api or not encrypted_secret:
                self.logger.error("❌ Missing API credentials")
                self.logger.error(f"   API Key: {'Present' if encrypted_api else 'Missing'}")
                self.logger.error(f"   API Secret: {'Present' if encrypted_secret else 'Missing'}")
                return None
            
            self.logger.debug(f"API Key (first 10 chars): {encrypted_api[:10]}...")
            self.logger.debug(f"Encrypted Secret (length): {len(encrypted_secret)}")
            
            # Decrypt the API secret
            decrypted_api = self.secret_manager.decrypt_secret(encrypted_api)
            decrypted_secret = self.secret_manager.decrypt_secret(encrypted_secret)
            
            if not decrypted_secret:
                self.logger.error("❌ Failed to decrypt API secret")
                self.logger.error("   This could be due to:")
                self.logger.error("   - Wrong encryption password")
                self.logger.error("   - Corrupted encrypted data")
                self.logger.error("   - API secret not properly encrypted")
                return None
            
            self.logger.debug(f"✅ API secret decrypted successfully (length: {len(decrypted_secret)})")
            
            # Add credentials to exchange config
            exchange_config.update({
                'apiKey': decrypted_api,
                'secret': decrypted_secret
            })
            
            # Initialize exchange
            self.logger.debug("Initializing Bybit exchange with optimized connection pool...")
            exchange = ccxt.bybit(exchange_config)

            # Enable demo trading if in sandbox mode
            if self.config.sandbox_mode:
                exchange.enable_demo_trading(True)
                self.logger.info("✅ Bybit demo trading mode enabled (sandbox)")
            
            # Configure connection pool settings if available
            try:
                # Try to configure the underlying requests session if it exists
                if hasattr(exchange, 'session') and exchange.session is not None:
                    import requests.adapters
                    
                    # Create adapter with larger connection pool
                    adapter = requests.adapters.HTTPAdapter(
                        pool_connections=20,
                        pool_maxsize=20,
                        pool_block=False,
                        max_retries=3
                    )
                    
                    # Mount adapter for both HTTP and HTTPS
                    exchange.session.mount('http://', adapter)
                    exchange.session.mount('https://', adapter)
                    
                    self.logger.debug("✅ Enhanced connection pool adapter configured")
                else:
                    self.logger.debug("ℹ️ Exchange session not available for optimization")
                    
            except Exception as e:
                self.logger.debug(f"Connection pool optimization not available: {e}")
            
            # Load markets with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.logger.debug(f"Loading markets (attempt {attempt + 1}/{max_retries})...")
                    exchange.load_markets()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"Market loading attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            mode = "DEMO" if self.config.sandbox_mode else "PRODUCTION"
            self.logger.info(f"✅ Connected to Bybit {mode} with decrypted credentials")
            self.logger.debug(f"   Rate limit: {exchange_config['rateLimit']}ms")
            self.logger.debug(f"   Timeout: {self.config.api_timeout}ms")
            
            return exchange
            
        except ccxt.AuthenticationError as e:
            self.logger.error(f"❌ Authentication error: {e}")
            self.logger.error("   Possible causes:")
            self.logger.error("   - Invalid API key")
            self.logger.error("   - Invalid API secret")
            self.logger.error("   - API credentials not enabled for trading")
            self.logger.error("   - Wrong sandbox mode setting")
            return None
        except ccxt.PermissionDenied as e:
            self.logger.error(f"❌ Permission denied: {e}")
            self.logger.error("   Check your API key permissions on Bybit")
            return None
        except ccxt.NetworkError as e:
            self.logger.error(f"❌ Network error: {e}")
            self.logger.error("   Check your internet connection")
            return None
        except Exception as e:
            self.logger.error(f"❌ Exchange setup failed: {e}")
            
            # Log additional debug info
            if hasattr(e, 'response'):
                self.logger.error(f"API Response: {e.response}")
            
            return None
    
    def test_connection(self) -> bool:
        """Test exchange connection with Global API and API credentials"""
        try:
            if not self.exchange:
                self.logger.error("❌ Exchange not initialized")
                return False
            
            self.logger.debug("Testing Global API connection...")
            
            # Log the endpoint being used
            api_endpoint = self.exchange.urls['api']['public']
            self.logger.debug(f"Testing connection to: {api_endpoint}")
            
            # Test API connection by fetching balance
            balance = self.exchange.fetch_balance()
            
            self.logger.info("✅ Global API connection test successful")
            # self.logger.info(f"🌍 Connected via: {api_endpoint}")
            self.logger.debug(f"   Account currencies: {list(balance.keys())[:5]}...")
            
            return True
            
        except ccxt.AuthenticationError as e:
            self.logger.error(f"❌ Authentication failed with Global API: {e}")
            return False
        except ccxt.PermissionDenied as e:
            self.logger.error(f"❌ Permission denied on Global API: {e}")
            return False
        except ccxt.NetworkError as e:
            self.logger.error(f"❌ Network error with {'Demo API' if self.config.sandbox_mode else 'Global API'}: {e}")
            self.logger.error("   This could indicate:")
            if self.config.sandbox_mode:
                self.logger.error("   - api-demo.bybit.com is not accessible from your location")
            else:
                self.logger.error("   - api.bybitglobal.com is not accessible from your location")
            self.logger.error("   - Firewall blocking the connection")
            self.logger.error("   - Internet connectivity issues")
            return False
        except Exception as e:
            self.logger.error(f"❌ Global API connection test failed: {e}")
            return False
    
    def validate_credentials(self) -> Dict[str, bool]:
        """Validate and return credential status with Global API support"""
        try:
            status = {
                'api_key_present': bool(self.config.api_key),
                'api_secret_present': bool(self.config.api_secret),
                'demo_api_key_present': bool(self.config.demo_api_key),
                'demo_api_secret_present': bool(self.config.demo_api_secret),
                'credentials_decryptable': False,
                'exchange_connection': False,
                'global_api_accessible': False
            }
            
            # Test decryption
            if self.config.sandbox_mode:
                encrypted_secret = self.config.demo_api_secret
            else:
                encrypted_secret = self.config.api_secret
            
            if encrypted_secret:
                decrypted = self.secret_manager.decrypt_secret(encrypted_secret)
                status['credentials_decryptable'] = bool(decrypted)
            
            # Test Global API accessibility
            try:
                import requests
                if self.config.sandbox_mode:
                    test_url = 'https://api-demo.bybit.com/v2/public/time'
                    api_name = 'Demo API'
                else:
                    test_url = 'https://api.bybitglobal.com/v2/public/time'
                    api_name = 'Global API'
                
                response = requests.get(test_url, timeout=10)
                status['global_api_accessible'] = response.status_code == 200
                if status['global_api_accessible']:
                    self.logger.debug(f"✅ {api_name} endpoint is accessible")
                else:
                    self.logger.warning(f"⚠️ {api_name} returned status: {response.status_code}")
            except Exception as e:
                self.logger.warning(f"⚠️ {api_name} accessibility test failed: {e}")
                status['global_api_accessible'] = False
            
            # Test exchange connection
            status['exchange_connection'] = self.test_connection()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error validating credentials: {e}")
            return {}
    
    def get_api_info(self) -> Dict[str, str]:
        """Get current API endpoint information"""
        try:
            if not self.exchange:
                return {'error': 'Exchange not initialized'}
            
            return {
                'public_api': self.exchange.urls['api']['public'],
                'private_api': self.exchange.urls['api']['private'],
                'demo_public': self.exchange.urls.get('test', {}).get('public', 'N/A'),
                'demo_private': self.exchange.urls.get('test', {}).get('private', 'N/A'),
                'sandbox_mode': self.config.sandbox_mode,
                'hostname': self.exchange.options.get('hostname', 'Not set')
            }
        except Exception as e:
            return {'error': str(e)}
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = None, limit: int = None) -> pd.DataFrame:
        """Fetch OHLCV data with Global API error handling and connection reuse"""
        if timeframe is None:
            timeframe = self.config.timeframe
            
        if limit is None:
            limit = self.config.ohlcv_limit_primary
            
        try:
            if not self.exchange:
                self.logger.error("Exchange not available for data fetching")
                return pd.DataFrame()
            
            # Add small delay to respect rate limits and reduce connection pressure
            time.sleep(0.1)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                self.logger.warning(f"No OHLCV data returned for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except ccxt.NetworkError as e:
            self.logger.warning(f"Global API network error fetching {symbol} data: {e}")
            time.sleep(1)  # Brief pause before retry
            return pd.DataFrame()
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Global API rate limit exceeded for {symbol}: {e}")
            time.sleep(2)  # Longer pause for rate limit
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            self.logger.warning(f"Global API exchange error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to fetch {symbol} data from Global API: {e}")
            return pd.DataFrame()
    
    def get_top_symbols(self) -> List[Dict]:
        """Get top trading symbols by volume with Global API connection optimization"""
        try:
            if not self.exchange:
                self.logger.error("Exchange not available for symbol fetching")
                return []
            
            # Add delay to prevent connection pool exhaustion
            time.sleep(0.2)
            
            self.logger.debug("Fetching tickers from Global API...")
            tickers = self.exchange.fetch_tickers()
            
            # Filter for USDT perpetual contracts
            usdt_symbols = []
            for symbol, ticker in tickers.items():
                if 'USDT' in symbol and ticker.get('quoteVolume', 0) >= self.config.min_volume_24h:
                    usdt_symbols.append({
                        'symbol': symbol,
                        'volume_24h': ticker['quoteVolume'],
                        'price_change_24h': ticker.get('percentage', 0),
                        'current_price': ticker.get('last', 0)
                    })
            
            # Sort by volume and limit
            usdt_symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
            result = usdt_symbols[:self.config.max_symbols_scan]
            
            self.logger.debug(f"Found {len(result)} symbols meeting volume criteria from Global API")
            return result
            
        except ccxt.NetworkError as e:
            self.logger.warning(f"Global API network error getting symbols: {e}")
            return []
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Global API rate limit exceeded getting symbols: {e}")
            time.sleep(2)
            return []
        except Exception as e:
            self.logger.error(f"Failed to get symbols from Global API: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """Get account information with Global API connection reuse"""
        try:
            if not self.exchange:
                return {}
            
            # Add delay to prevent connection pool exhaustion
            time.sleep(0.1)
            
            self.logger.debug("Fetching account info from Global API...")
            balance = self.exchange.fetch_balance()
            
            return {
                'total_equity': balance.get('USDT', {}).get('total', 0),
                'free_balance': balance.get('USDT', {}).get('free', 0),
                'used_balance': balance.get('USDT', {}).get('used', 0),
                'currencies': list(balance.keys()),
                'api_endpoint': self.exchange.urls['api']['public']
            }
            
        except ccxt.NetworkError as e:
            self.logger.warning(f"Global API network error getting account info: {e}")
            return {}
        except ccxt.RateLimitExceeded as e:
            self.logger.warning(f"Global API rate limit exceeded getting account info: {e}")
            time.sleep(1)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get account info from Global API: {e}")
            return {}
    
    def cleanup_connections(self):
        """Clean up exchange connections to prevent pool exhaustion"""
        try:
            if self.exchange:
                # Check if exchange has a session attribute and it's a valid session object
                if hasattr(self.exchange, 'session') and self.exchange.session is not None:
                    # Check if it's a requests session object
                    if hasattr(self.exchange.session, 'close'):
                        self.exchange.session.close()
                        self.logger.debug("✅ Global API session closed")
                    else:
                        self.logger.debug("ℹ️ Global API session doesn't support close method")
                else:
                    self.logger.debug("ℹ️ No Global API session to cleanup")
        except Exception as e:
            self.logger.debug(f"Global API connection cleanup note: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup_connections()
        except:
            pass
    
    @staticmethod
    def test_global_api_accessibility() -> Dict[str, bool]:
        """Static method to test Global API accessibility without credentials"""
        try:
            import requests
            
            endpoints = {
                'global_api': 'https://api.bybitglobal.com/v2/public/time',
                'standard_api': 'https://api.bybit.com/v2/public/time',
                'demo_api': 'https://api-demo.bybit.com/v2/public/time',
                'testnet': 'https://api-testnet.bybit.com/v2/public/time'
            }
            
            results = {}
            
            for name, url in endpoints.items():
                try:
                    response = requests.get(url, timeout=5)
                    results[name] = response.status_code == 200
                    print(f"{'✅' if results[name] else '❌'} {name}: {url} - Status: {response.status_code}")
                except Exception as e:
                    results[name] = False
                    print(f"❌ {name}: {url} - Error: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ API accessibility test failed: {e}")
            return {}


# Convenience function for testing API accessibility
def test_bybit_api_endpoints():
    """Test which Bybit API endpoints are accessible from your location"""
    print("🌍 Testing Bybit API Endpoint Accessibility...")
    print("=" * 50)
    
    results = ExchangeManager.test_global_api_accessibility()
    
    print("\n📊 Results Summary:")
    if results.get('global_api', False):
        print("✅ Global API (api.bybitglobal.com) - RECOMMENDED for your location")
    else:
        print("❌ Global API (api.bybitglobal.com) - Not accessible")
    
    if results.get('standard_api', False):
        print("✅ Standard API (api.bybit.com) - Available")
    else:
        print("❌ Standard API (api.bybit.com) - Not accessible (expected if you need VPN)")
    
    print("\n💡 Recommendation:")
    if results.get('global_api', False):
        print("   Use the Global API configuration - no VPN required!")
    elif results.get('standard_api', False):
        print("   Use standard API or VPN to access api.bybit.com")
    else:
        print("   Check your internet connection - no Bybit endpoints accessible")
    
    return results


if __name__ == "__main__":
    # Test API accessibility when run directly
    test_bybit_api_endpoints()