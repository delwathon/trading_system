"""
BitUnix Signal Receiver and Trade Executor System - FIXED VERSION
Implements BitUnix Futures API with proper authentication and endpoints
Features:
- MILESTONE-BASED TRAILING STOPS with leverage support
- PARTIAL PROFIT TAKING at 100%, 200%, 300% leveraged profit
- ADAPTIVE POSITION SIZING
- PORTFOLIO HEAT MONITORING
Version: 4.1-BitUnix-Fixed

API Documentation: https://openapidoc.bitunix.com/
"""

import time
import logging
import threading
import numpy as np
import requests
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urlencode


# ========================================
# BITUNIX FUTURES API WRAPPER - FIXED
# ========================================

class BitUnixFuturesAPI:
    """
    BitUnix Futures API wrapper with proper authentication - FIXED VERSION
    Documentation: https://openapidoc.bitunix.com/
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://fapi.bitunix.com"
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
    def _sha256_hex(self, input_string: str) -> str:
        """Generate SHA256 hash"""
        return hashlib.sha256(input_string.encode('utf-8')).hexdigest()
    
    def _generate_signature(self, nonce: str, timestamp: str, query_params: str, body: str) -> str:
        """
        Generate BitUnix signature using double SHA256
        
        Signature process:
        1. Concatenate: nonce + timestamp + apiKey + queryParams + body
        2. First SHA256 hash
        3. Append secretKey to digest
        4. Second SHA256 hash
        """
        # Step 1: Concatenate all parts
        digest_input = nonce + timestamp + self.api_key + query_params + body
        
        # Step 2: First SHA256
        digest = self._sha256_hex(digest_input)
        
        # Step 3: Append secret key
        sign_input = digest + self.secret_key
        
        # Step 4: Second SHA256
        signature = self._sha256_hex(sign_input)
        
        return signature
    
    def _prepare_query_params(self, params: Dict) -> str:
        """
        Prepare query parameters for signature
        Sort by key in ASCII order and remove all spaces
        """
        if not params:
            return ""
        
        # Sort parameters by key
        sorted_params = sorted(params.items())
        
        # Create query string without spaces
        query_parts = []
        for key, value in sorted_params:
            query_parts.append(f"{key}{value}")
        
        return "".join(query_parts)
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, body: Dict = None) -> Dict:
        """Make authenticated request to BitUnix API"""
        url = f"{self.base_url}{endpoint}"
        
        # Generate nonce and timestamp
        nonce = uuid.uuid4().hex[:32]  # 32-character hex string
        timestamp = str(int(time.time() * 1000))  # Milliseconds
        
        # Prepare query parameters for signature
        query_params_for_sig = ""
        query_string = ""
        if params and method == 'GET':
            query_params_for_sig = self._prepare_query_params(params)
            query_string = urlencode(params)
            url = f"{url}?{query_string}"
        
        # Prepare body for signature
        body_for_sig = ""
        json_body = None
        if body and method == 'POST':
            json_body = json.dumps(body, separators=(',', ':'), sort_keys=True)
            body_for_sig = json_body
        
        # Generate signature
        signature = self._generate_signature(nonce, timestamp, query_params_for_sig, body_for_sig)
        
        # Prepare headers
        headers = {
            'api-key': self.api_key,
            'nonce': nonce,
            'timestamp': timestamp,
            'sign': signature,
            'Content-Type': 'application/json'
        }
        
        # Make request
        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, data=json_body, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            # Check for API errors
            if result.get('code') != 0:
                error_msg = result.get('msg', 'Unknown error')
                self.logger.error(f"API error {result.get('code')}: {error_msg}")
                raise Exception(f"BitUnix API error: {error_msg}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    # ========== ACCOUNT ENDPOINTS ==========
    
    def get_account_info(self, margin_coin: str = "USDT") -> Dict:
        """Get account information"""
        params = {"marginCoin": margin_coin}
        return self._make_request('GET', '/api/v1/futures/account', params=params)
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        result = self.get_account_info()
        
        if result.get('code') == 0 and result.get('data'):
            data = result['data']
            return {
                'available': float(data.get('available', 0)),
                'equity': float(data.get('equity', 0)),
                'total': float(data.get('total', 0)),
                'frozen': float(data.get('frozen', 0)),
                'unrealizedPL': float(data.get('unrealizedPL', 0))
            }
        return {'available': 0, 'equity': 0, 'total': 0}
    
    def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """Change leverage for a symbol - FIXED"""
        body = {
            "symbol": symbol,
            "leverage": str(leverage),  # Convert to string
            "marginCoin": "USDT"  # Add marginCoin parameter
        }
        return self._make_request('POST', '/api/v1/futures/account/change_leverage', body=body)
    
    def change_margin_mode(self, symbol: str, margin_mode: str, margin_coin: str = "USDT") -> Dict:
        """Change margin mode (ISOLATED/CROSS) - FIXED"""
        # Map margin mode to expected format (use integers not strings)
        mode_map = {
            "ISOLATED": 1,  # 1 for isolated
            "CROSS": 2      # 2 for cross
        }
        
        body = {
            "symbol": symbol,
            "marginMode": mode_map.get(margin_mode.upper(), 1),  # Use integer value
            "marginCoin": margin_coin
        }
        return self._make_request('POST', '/api/v1/futures/account/change_margin_mode', body=body)
    
    # ========== MARKET DATA ENDPOINTS - FIXED ==========
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker data for a symbol - FIXED to handle list response"""
        params = {"symbols": symbol}  # Note: changed to 'symbols' (plural)
        result = self._make_request('GET', '/api/v1/futures/market/tickers', params=params)
        
        if result.get('code') == 0 and result.get('data'):
            data = result['data']
            
            # Handle list response
            if isinstance(data, list) and len(data) > 0:
                ticker = data[0]  # Get first ticker from list
            else:
                ticker = data
            
            return {
                'last': float(ticker.get('last', 0)),
                'markPrice': float(ticker.get('markPrice', 0)),
                'indexPrice': float(ticker.get('indexPrice', 0)),
                'bid': float(ticker.get('bid1', 0)),
                'ask': float(ticker.get('ask1', 0)),
                'volume24h': float(ticker.get('volume24h', 0))
            }
        return {'last': 0, 'markPrice': 0}
    
    def get_depth(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book depth"""
        params = {
            "symbol": symbol,
            "limit": limit
        }
        return self._make_request('GET', '/api/v1/futures/market/depth', params=params)
    
    def get_kline(self, symbol: str, interval: str, limit: int = 100) -> Dict:
        """Get kline/candlestick data"""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self._make_request('GET', '/api/v1/futures/market/kline', params=params)
    
    # ========== TRADING ENDPOINTS - FIXED ==========
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                   price: float = None, stop_price: float = None, take_profit: float = None,
                   time_in_force: str = "GTC", reduce_only: bool = False) -> Dict:
        """
        Place a futures order - FIXED
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "BUY" or "SELL"
            order_type: "LIMIT", "MARKET", "STOP_LIMIT", "STOP_MARKET"
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
            stop_price: Stop loss price
            take_profit: Take profit price
            time_in_force: "GTC", "IOC", "FOK"
            reduce_only: True for closing positions only
        """
        body = {
            "symbol": symbol,
            "side": side.upper(),
            "orderType": order_type.upper(),
            "qty": str(quantity),
            "effect": time_in_force,
            "reduceOnly": reduce_only
        }
        
        # Add tradeSide if reducing position
        if reduce_only:
            body["tradeSide"] = "CLOSE"
        else:
            body["tradeSide"] = "OPEN"
        
        if price:
            body["price"] = str(price)
        
        if stop_price:
            body["slPrice"] = str(stop_price)
            body["slStopType"] = "MARK"
            body["slOrderType"] = "MARKET"
            
        if take_profit:
            body["tpPrice"] = str(take_profit)
            body["tpStopType"] = "MARK"
            body["tpOrderType"] = "LIMIT"
            body["tpOrderPrice"] = str(take_profit)
        
        return self._make_request('POST', '/api/v1/futures/trade/place_order', body=body)
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order"""
        body = {
            "symbol": symbol,
            "orderId": order_id
        }
        return self._make_request('POST', '/api/v1/futures/trade/cancel_order', body=body)
    
    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancel all orders for a symbol or all symbols"""
        body = {}
        if symbol:
            body["symbol"] = symbol
        return self._make_request('POST', '/api/v1/futures/trade/cancel_all_orders', body=body)
    
    def get_open_orders(self, symbol: str = None) -> Dict:
        """Get open orders - FIXED to use correct endpoint"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        # Use get_history_orders with filter for open status
        return self._make_request('GET', '/api/v1/futures/trade/get_history_orders', params=params)
    
    def get_order_history(self, symbol: str = None, limit: int = 50) -> Dict:
        """Get order history - FIXED"""
        params = {"limit": str(limit)}
        if symbol:
            params["symbol"] = symbol
        return self._make_request('GET', '/api/v1/futures/trade/get_history_orders', params=params)
    
    # ========== POSITION ENDPOINTS - FIXED ==========
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """Get current positions - FIXED to use correct endpoint"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        # Use get_pending_positions endpoint
        result = self._make_request('GET', '/api/v1/futures/position/get_pending_positions', params=params)
        
        if result.get('code') == 0 and result.get('data'):
            positions = []
            data = result['data']
            
            # Handle both list and single position response
            if isinstance(data, list):
                pos_list = data
            else:
                pos_list = [data]
            
            for pos in pos_list:
                # Only include positions with actual size
                if float(pos.get('positionAmt', 0)) > 0:
                    positions.append({
                        'symbol': pos.get('symbol'),
                        'side': 'long' if pos.get('side', '').upper() == 'BUY' else 'short',
                        'size': float(pos.get('positionAmt', 0)),
                        'entryPrice': float(pos.get('entryPrice', 0)),
                        'markPrice': float(pos.get('markPrice', 0)),
                        'unrealizedPnl': float(pos.get('unrealizedProfit', 0)),
                        'leverage': float(pos.get('leverage', 1)),
                        'marginType': 'isolated' if pos.get('marginMode') == '1' else 'cross'
                    })
            return positions
        return []
    
    def close_position(self, symbol: str, side: str = None) -> Dict:
        """Close a position - FIXED"""
        positions = self.get_positions(symbol)
        
        for pos in positions:
            if pos['symbol'] == symbol and pos['size'] > 0:
                # Determine close side
                if pos['side'] == 'long':
                    close_side = "SELL"
                else:
                    close_side = "BUY"
                
                # Place market order to close
                return self.place_order(
                    symbol=symbol,
                    side=close_side,
                    order_type="MARKET",
                    quantity=pos['size'],
                    reduce_only=True
                )
        
        return {'code': -1, 'msg': 'No position found'}
    
    def set_stop_loss_take_profit(self, symbol: str, stop_loss: float = None, 
                                  take_profit: float = None) -> Dict:
        """Set or update stop loss and take profit for a position - FIXED"""
        body = {"symbol": symbol}
        
        if stop_loss:
            body["slPrice"] = str(stop_loss)
            body["slStopType"] = "MARK"
            body["slOrderType"] = "MARKET"
        
        if take_profit:
            body["tpPrice"] = str(take_profit)
            body["tpStopType"] = "MARK"
            body["tpOrderType"] = "LIMIT"
            body["tpOrderPrice"] = str(take_profit)
        
        # Use TP/SL specific endpoint if available
        return self._make_request('POST', '/api/v1/futures/tpsl/place_order', body=body)


# ========================================
# DATA STRUCTURES (unchanged)
# ========================================

@dataclass
class SignalData:
    """Trading signal data structure"""
    symbol: str  # e.g., "BTCUSDT"
    side: str  # "buy" or "sell"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float = 50.0
    risk_reward_ratio: float = 2.0
    signal_id: str = ""
    timestamp: datetime = None
    metadata: Dict = None
    
    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)
        if not self.metadata:
            self.metadata = {}


@dataclass
class ExecutionConfig:
    """Configuration for trade execution"""
    risk_amount: float = 5.0  # Risk percentage per trade
    default_leverage: int = 25  # Default leverage
    max_concurrent_positions: int = 10
    auto_close_profit_pct: float = 500.0  # Auto-close at 500% leveraged profit
    auto_close_loss_pct: float = 100.0  # Auto-close at 100% leveraged loss
    adaptive_sizing: bool = True
    partial_profits_enabled: bool = True
    trailing_stops_enabled: bool = True
    margin_mode: str = "ISOLATED"  # ISOLATED or CROSS


@dataclass
class PositionData:
    """Position tracking data"""
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: float
    risk_amount: float
    stop_loss: float
    take_profit: float
    position_id: str
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    milestone_reached: str = 'none'
    original_size: float = 0.0


# ========================================
# POSITION SIZING (unchanged)
# ========================================

class PositionSizer:
    """Calculate position sizes with leverage"""
    
    def __init__(self, api: BitUnixFuturesAPI):
        self.api = api
        self.logger = logging.getLogger(__name__)
    
    def get_available_balance(self) -> float:
        """Get available USDT balance"""
        try:
            balance = self.api.get_account_balance()
            return balance.get('available', 0)
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def calculate_position_size(self, risk_pct: float, leverage: int, entry_price: float) -> float:
        """Calculate position size based on risk percentage"""
        try:
            balance = self.get_available_balance()
            if balance <= 0:
                raise ValueError("Insufficient balance")
            
            # Calculate risk amount in USDT
            risk_amount = balance * (risk_pct / 100)
            
            # Calculate position size
            position_size = (risk_amount * leverage) / entry_price
            
            self.logger.debug(f"Position sizing: Balance={balance}, Risk={risk_pct}%, Size={position_size}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_adaptive_size(self, risk_pct: float, leverage: int, 
                              entry_price: float, signal: SignalData) -> Tuple[float, float]:
        """Adaptive position sizing based on signal quality"""
        base_risk = risk_pct
        
        # Adjust based on confidence
        if signal.confidence >= 80:
            risk_multiplier = 1.3
        elif signal.confidence >= 70:
            risk_multiplier = 1.1
        elif signal.confidence >= 60:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.7
        
        # Adjust based on risk/reward
        if signal.risk_reward_ratio >= 3.0:
            rr_multiplier = 1.2
        elif signal.risk_reward_ratio >= 2.0:
            rr_multiplier = 1.0
        else:
            rr_multiplier = 0.8
        
        # Calculate adjusted risk
        adjusted_risk = base_risk * risk_multiplier * rr_multiplier
        adjusted_risk = max(1.0, min(10.0, adjusted_risk))
        
        # Calculate position size
        position_size = self.calculate_position_size(adjusted_risk, leverage, entry_price)
        
        return position_size, adjusted_risk


# ========================================
# PROFIT MONITOR (unchanged)
# ========================================

class ProfitMonitor:
    """Monitor positions for profit targets and trailing stops"""
    
    def __init__(self, api: BitUnixFuturesAPI, config: ExecutionConfig):
        self.api = api
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.position_tracker = {}
    
    def calculate_leveraged_profit_pct(self, entry_price: float, current_price: float, 
                                      leverage: int, side: str) -> float:
        """Calculate leveraged profit percentage"""
        if entry_price <= 0:
            return 0.0
        
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        
        if side.lower() == 'sell' or side.lower() == 'short':
            price_change_pct = -price_change_pct
        
        return price_change_pct * leverage
    
    def check_partial_profit(self, symbol: str, leveraged_profit: float) -> Optional[Dict]:
        """Check if partial profit should be taken"""
        if not self.config.partial_profits_enabled:
            return None
        
        if symbol not in self.position_tracker:
            self.position_tracker[symbol] = {
                'partial_100_taken': False,
                'partial_200_taken': False,
                'partial_300_taken': False
            }
        
        tracker = self.position_tracker[symbol]
        
        if leveraged_profit >= 300 and not tracker['partial_300_taken']:
            tracker['partial_300_taken'] = True
            return {'level': 300, 'percentage': 50}
        elif leveraged_profit >= 200 and not tracker['partial_200_taken']:
            tracker['partial_200_taken'] = True
            return {'level': 200, 'percentage': 40}
        elif leveraged_profit >= 100 and not tracker['partial_100_taken']:
            tracker['partial_100_taken'] = True
            return {'level': 100, 'percentage': 30}
        
        return None
    
    def execute_partial_close(self, symbol: str, percentage: float) -> bool:
        """Execute partial position close"""
        try:
            positions = self.api.get_positions(symbol)
            
            for pos in positions:
                if pos['size'] > 0:
                    close_size = pos['size'] * (percentage / 100)
                    close_side = "SELL" if pos['side'] == 'long' else "BUY"
                    
                    result = self.api.place_order(
                        symbol=symbol,
                        side=close_side,
                        order_type="MARKET",
                        quantity=close_size,
                        reduce_only=True
                    )
                    
                    if result.get('code') == 0:
                        self.logger.info(f"✅ Partial profit taken: {percentage}% of {symbol}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing partial close: {e}")
            return False
    
    def update_trailing_stop(self, symbol: str, level: int) -> bool:
        """Update trailing stop based on profit level"""
        if not self.config.trailing_stops_enabled:
            return False
        
        try:
            positions = self.api.get_positions(symbol)
            
            for pos in positions:
                if pos['size'] > 0:
                    entry_price = pos['entryPrice']
                    leverage = pos['leverage']
                    side = pos['side']
                    
                    # Calculate new stop price based on level
                    if level == 100:
                        # Move to break-even
                        new_stop = entry_price
                    elif level == 200:
                        # Lock in 100% leveraged profit
                        if side == 'long':
                            new_stop = entry_price * (1 + 1.0/leverage)
                        else:
                            new_stop = entry_price * (1 - 1.0/leverage)
                    elif level == 300:
                        # Lock in 200% leveraged profit
                        if side == 'long':
                            new_stop = entry_price * (1 + 2.0/leverage)
                        else:
                            new_stop = entry_price * (1 - 2.0/leverage)
                    else:
                        return False
                    
                    # Update stop loss
                    result = self.api.set_stop_loss_take_profit(symbol, stop_loss=new_stop)
                    
                    if result.get('code') == 0:
                        self.logger.info(f"✅ Trailing stop updated for {symbol} at level {level}%")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}")
            return False
    
    def monitor_positions(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                positions = self.api.get_positions()
                
                for pos in positions:
                    if pos['size'] <= 0:
                        continue
                    
                    symbol = pos['symbol']
                    entry_price = pos['entryPrice']
                    current_price = pos['markPrice']
                    leverage = pos['leverage']
                    side = pos['side']
                    
                    # Calculate leveraged profit
                    leveraged_profit = self.calculate_leveraged_profit_pct(
                        entry_price, current_price, leverage, side
                    )
                    
                    # Check for partial profit
                    partial_info = self.check_partial_profit(symbol, leveraged_profit)
                    if partial_info:
                        if self.execute_partial_close(symbol, partial_info['percentage']):
                            self.update_trailing_stop(symbol, partial_info['level'])
                    
                    # Check auto-close conditions
                    if leveraged_profit >= self.config.auto_close_profit_pct:
                        self.logger.info(f"💰 Auto-closing {symbol} at {leveraged_profit:.1f}% profit")
                        self.api.close_position(symbol, side)
                    elif leveraged_profit <= -self.config.auto_close_loss_pct:
                        self.logger.warning(f"🛑 Auto-closing {symbol} at {leveraged_profit:.1f}% loss")
                        self.api.close_position(symbol, side)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(10)
    
    def start_monitoring(self):
        """Start position monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)
            self.monitor_thread.start()
            self.logger.info("📊 Started position monitoring")
    
    def stop_monitoring(self):
        """Stop position monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛑 Stopped position monitoring")


# ========================================
# ORDER EXECUTOR (unchanged)
# ========================================

class OrderExecutor:
    """Execute orders with risk management"""
    
    def __init__(self, api: BitUnixFuturesAPI, config: ExecutionConfig):
        self.api = api
        self.config = config
        self.position_sizer = PositionSizer(api)
        self.logger = logging.getLogger(__name__)
    
    def execute_signal(self, signal: SignalData) -> Dict:
        """Execute a trading signal"""
        result = {
            'success': False,
            'signal_id': signal.signal_id,
            'message': '',
            'order_id': None,
            'position_data': {}
        }
        
        try:
            symbol = signal.symbol
            side = signal.side.upper()
            
            # Check if position already exists
            positions = self.api.get_positions(symbol)
            for pos in positions:
                if pos['size'] > 0:
                    result['message'] = f"Position already exists for {symbol}"
                    return result
            
            # Set margin mode
            self.api.change_margin_mode(symbol, self.config.margin_mode)
            
            # Set leverage
            leverage = self.config.default_leverage
            self.api.change_leverage(symbol, leverage)
            
            # Calculate position size
            if self.config.adaptive_sizing:
                position_size, risk_pct = self.position_sizer.calculate_adaptive_size(
                    self.config.risk_amount, leverage, signal.entry_price, signal
                )
            else:
                position_size = self.position_sizer.calculate_position_size(
                    self.config.risk_amount, leverage, signal.entry_price
                )
                risk_pct = self.config.risk_amount
            
            # Round position size to appropriate decimals
            position_size = round(position_size, 4)
            
            if position_size <= 0:
                result['message'] = "Invalid position size calculated"
                return result
            
            # Place order
            self.logger.info(f"📈 Placing {side} order: {symbol}, Size: {position_size}")
            
            order_result = self.api.place_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=position_size,
                stop_price=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_result.get('code') == 0:
                result['success'] = True
                result['message'] = "Order executed successfully"
                result['order_id'] = order_result.get('data', {}).get('orderId')
                result['position_data'] = {
                    'symbol': symbol,
                    'side': side,
                    'size': position_size,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'leverage': leverage,
                    'risk_pct': risk_pct
                }
                
                self.logger.info(f"✅ Order executed: {result['order_id']}")
            else:
                result['message'] = f"Order failed: {order_result.get('msg', 'Unknown error')}"
                self.logger.error(result['message'])
            
            return result
            
        except Exception as e:
            result['message'] = f"Execution error: {e}"
            self.logger.error(result['message'])
            return result


# ========================================
# POSITION MANAGER (unchanged)
# ========================================

class PositionManager:
    """Manage active positions"""
    
    def __init__(self, api: BitUnixFuturesAPI, config: ExecutionConfig):
        self.api = api
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_positions_count(self) -> int:
        """Get number of active positions"""
        try:
            positions = self.api.get_positions()
            return sum(1 for pos in positions if pos['size'] > 0)
        except Exception as e:
            self.logger.error(f"Error getting positions count: {e}")
            return 0
    
    def can_open_position(self) -> bool:
        """Check if new position can be opened"""
        return self.get_positions_count() < self.config.max_concurrent_positions
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get position information for symbol"""
        try:
            positions = self.api.get_positions(symbol)
            for pos in positions:
                if pos['size'] > 0:
                    return pos
            return None
        except Exception as e:
            self.logger.error(f"Error getting position info: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """Close position for symbol"""
        try:
            positions = self.api.get_positions(symbol)
            for pos in positions:
                if pos['size'] > 0:
                    result = self.api.close_position(symbol, pos['side'])
                    if result.get('code') == 0:
                        self.logger.info(f"✅ Closed position: {symbol}")
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False


# ========================================
# MAIN SIGNAL EXECUTOR (unchanged)
# ========================================

class BitUnixSignalExecutor:
    """
    Main class for executing trading signals on BitUnix Futures
    """
    
    def __init__(self, api_key: str, secret_key: str, config: ExecutionConfig = None):
        """
        Initialize the signal executor
        
        Args:
            api_key: BitUnix API key
            secret_key: BitUnix API secret
            config: Execution configuration
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize API
        self.api = BitUnixFuturesAPI(api_key, secret_key)
        
        # Use default config if not provided
        self.config = config or ExecutionConfig()
        
        # Initialize components
        self.order_executor = OrderExecutor(self.api, self.config)
        self.position_manager = PositionManager(self.api, self.config)
        self.profit_monitor = ProfitMonitor(self.api, self.config)
        
        # Test connection
        self._test_connection()
        
        # Start monitoring
        if self.config.partial_profits_enabled or self.config.trailing_stops_enabled:
            self.profit_monitor.start_monitoring()
        
        self.logger.info("🚀 BitUnix Signal Executor initialized")
        self._log_configuration()
    
    def _test_connection(self):
        """Test API connection"""
        try:
            balance = self.api.get_account_balance()
            self.logger.info(f"✅ API connected. Balance: ${balance.get('available', 0):.2f}")
        except Exception as e:
            self.logger.error(f"❌ API connection failed: {e}")
            raise
    
    def _log_configuration(self):
        """Log current configuration"""
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Max positions: {self.config.max_concurrent_positions}")
        self.logger.info(f"  Risk per trade: {self.config.risk_amount}%")
        self.logger.info(f"  Default leverage: {self.config.default_leverage}x")
        self.logger.info(f"  Margin mode: {self.config.margin_mode}")
        self.logger.info(f"  Partial profits: {'✅' if self.config.partial_profits_enabled else '❌'}")
        self.logger.info(f"  Trailing stops: {'✅' if self.config.trailing_stops_enabled else '❌'}")
    
    def execute_signal(self, signal: SignalData) -> Dict:
        """Execute a trading signal"""
        # Check if can open position
        if not self.position_manager.can_open_position():
            return {
                'success': False,
                'signal_id': signal.signal_id,
                'message': f"Maximum positions ({self.config.max_concurrent_positions}) reached"
            }
        
        # Check if position exists
        if self.position_manager.get_position_info(signal.symbol):
            return {
                'success': False,
                'signal_id': signal.signal_id,
                'message': f"Position already exists for {signal.symbol}"
            }
        
        # Execute the signal
        return self.order_executor.execute_signal(signal)
    
    def execute_batch(self, signals: List[SignalData]) -> List[Dict]:
        """Execute multiple signals"""
        results = []
        
        for signal in signals:
            if not self.position_manager.can_open_position():
                self.logger.warning("Max positions reached, stopping batch execution")
                break
            
            result = self.execute_signal(signal)
            results.append(result)
            
            if result['success']:
                time.sleep(1)  # Small delay between orders
        
        return results
    
    def get_account_status(self) -> Dict:
        """Get account status"""
        try:
            balance = self.api.get_account_balance()
            positions = self.api.get_positions()
            
            active_positions = [p for p in positions if p['size'] > 0]
            total_pnl = sum(p.get('unrealizedPnl', 0) for p in active_positions)
            
            return {
                'balance': balance.get('available', 0),
                'equity': balance.get('equity', 0),
                'positions_count': len(active_positions),
                'positions': active_positions,
                'total_unrealized_pnl': total_pnl
            }
        except Exception as e:
            self.logger.error(f"Error getting account status: {e}")
            return {}
    
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        return self.position_manager.close_position(symbol)
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position information"""
        return self.position_manager.get_position_info(symbol)
    
    def shutdown(self):
        """Shutdown the executor"""
        self.logger.info("Shutting down BitUnix Signal Executor...")
        self.profit_monitor.stop_monitoring()
        self.logger.info("✅ Shutdown complete")


# ========================================
# USAGE EXAMPLE
# ========================================

def example_usage():
    """Example of how to use the signal executor"""
    
    # Configuration
    config = ExecutionConfig(
        risk_amount=5.0,
        default_leverage=25,
        max_concurrent_positions=5,
        margin_mode="ISOLATED",
        partial_profits_enabled=True,
        trailing_stops_enabled=True
    )
    
    # Initialize executor
    executor = BitUnixSignalExecutor(
        api_key="your_api_key",
        secret_key="your_secret_key",
        config=config
    )
    
    # Create a signal
    signal = SignalData(
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000,
        confidence=75,
        risk_reward_ratio=2.0
    )
    
    # Execute signal
    result = executor.execute_signal(signal)
    
    if result['success']:
        print(f"✅ Trade executed: {result}")
    else:
        print(f"❌ Trade failed: {result['message']}")
    
    # Get account status
    status = executor.get_account_status()
    print(f"Account status: {status}")
    
    # Shutdown
    executor.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    example_usage()