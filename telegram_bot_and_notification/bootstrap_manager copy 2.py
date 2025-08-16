"""
Telegram Bootstrap Manager for Multi-User Trading System
CORRECTED VERSION - Only /start command, all interactions via inline keyboards
Admin buttons only visible to admin user
"""

import asyncio
import logging
import requests
import json
import base58
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Tuple
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ParseMode

from config.config import EnhancedSystemConfig
from database.models import DatabaseManager, SystemConfig, User, UserTier, ExchangeType, Subscription
from utils.encryption import SecretManager
from utils.logging import get_logger
from core.exchange import ExchangeManager


class BlockchainVerifier:
    """Production-ready blockchain payment verification for TRC20, ERC20, and BEP20"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # API endpoints for different networks
        self.api_endpoints = {
            'TRC20': {
                'main': 'https://api.trongrid.io',
                'backup': 'https://api.shasta.trongrid.io',
                'explorer': 'https://tronscan.org'
            },
            'ERC20': {
                'main': 'https://api.etherscan.io/api',
                'backup': 'https://api-goerli.etherscan.io/api',
                'explorer': 'https://etherscan.io',
                'api_key': 'YOUR_ETHERSCAN_API_KEY'  # Replace with your API key
            },
            'BEP20': {
                'main': 'https://api.bscscan.com/api',
                'backup': 'https://api-testnet.bscscan.com/api',
                'explorer': 'https://bscscan.com',
                'api_key': 'YOUR_BSCSCAN_API_KEY'  # Replace with your API key
            }
        }
        
        # USDT contract addresses on different networks
        self.usdt_contracts = {
            'TRC20': 'TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t',  # USDT on Tron
            'ERC20': '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT on Ethereum
            'BEP20': '0x55d398326f99059ff775485246999027b3197955'  # USDT on BSC
        }
    
    async def verify_payment(self, tx_hash: str, expected_address: str, 
                            min_amount: float, network: str = None) -> Dict:
        """Verify blockchain payment with production APIs"""
        try:
            # Clean transaction hash
            tx_hash = tx_hash.strip()
            
            # Auto-detect network if not specified
            if not network:
                network = self.detect_network(tx_hash)
                if not network:
                    # Try all networks
                    for net in ['TRC20', 'BEP20', 'ERC20']:
                        result = await self.verify_payment(tx_hash, expected_address, min_amount, net)
                        if result.get('verified'):
                            return result
                    return {'verified': False, 'error': 'Transaction not found on any network'}
            
            self.logger.info(f"Verifying {network} transaction: {tx_hash}")
            
            if network == 'TRC20':
                return await self.verify_tron_payment(tx_hash, expected_address, min_amount)
            elif network == 'ERC20':
                return await self.verify_ethereum_payment(tx_hash, expected_address, min_amount)
            elif network == 'BEP20':
                return await self.verify_bsc_payment(tx_hash, expected_address, min_amount)
            else:
                return {'verified': False, 'error': f'Unknown network: {network}'}
                
        except Exception as e:
            self.logger.error(f"Payment verification error: {e}")
            return {'verified': False, 'error': str(e)}
    
    def detect_network(self, tx_hash: str) -> Optional[str]:
        """Detect network based on transaction hash format"""
        tx_hash = tx_hash.strip()
        
        # Tron: 64 chars without 0x
        if len(tx_hash) == 64 and not tx_hash.startswith('0x'):
            return 'TRC20'
        # Ethereum/BSC: 66 chars with 0x
        elif tx_hash.startswith('0x') and len(tx_hash) == 66:
            # Default to BSC for USDT transfers (more common)
            return 'BEP20'
        
        return None
    
    async def verify_tron_payment(self, tx_hash: str, expected_address: str, 
                                 min_amount: float) -> Dict:
        """Verify TRC20 USDT payment on Tron network"""
        try:
            url = f"{self.api_endpoints['TRC20']['main']}/v1/transactions/{tx_hash}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return {'verified': False, 'error': 'Transaction not found'}
            
            data = response.json()
            
            if not data.get('data') or len(data['data']) == 0:
                return {'verified': False, 'error': 'Transaction not found'}
            
            tx = data['data'][0]
            
            # Check confirmation
            if not tx.get('blockNumber'):
                return {'verified': False, 'error': 'Transaction not confirmed'}
            
            # Parse contract data
            contract = tx.get('raw_data', {}).get('contract', [{}])[0]
            
            if contract.get('type') != 'TriggerSmartContract':
                return {'verified': False, 'error': 'Not a smart contract transaction'}
            
            contract_params = contract.get('parameter', {}).get('value', {})
            contract_address = contract_params.get('contract_address', '')
            data_hex = contract_params.get('data', '')
            
            # Verify USDT contract
            usdt_hex = self.tron_address_to_hex(self.usdt_contracts['TRC20'])
            if contract_address != usdt_hex:
                return {'verified': False, 'error': 'Not a USDT transfer'}
            
            # Parse transfer data (method_id: a9059cbb)
            if not data_hex.startswith('a9059cbb'):
                return {'verified': False, 'error': 'Not a transfer transaction'}
            
            # Extract recipient (32 bytes after method id)
            to_address_hex = '41' + data_hex[32:72]
            to_address = self.hex_to_tron_address(to_address_hex)
            
            # Extract amount (next 32 bytes)
            amount_hex = data_hex[72:136] if len(data_hex) >= 136 else '0'
            amount = int(amount_hex, 16) / 1e6  # USDT has 6 decimals
            
            # Get sender
            from_address_hex = contract_params.get('owner_address', '')
            from_address = self.hex_to_tron_address(from_address_hex)
            
            # Verify recipient
            if to_address.lower() != expected_address.lower():
                return {
                    'verified': False,
                    'error': f'Wrong recipient. Expected: {expected_address}, Got: {to_address}'
                }
            
            # Verify amount
            if amount < min_amount:
                return {
                    'verified': False,
                    'error': f'Insufficient amount: ${amount:.2f} < ${min_amount}'
                }
            
            return {
                'verified': True,
                'network': 'TRC20',
                'amount': amount,
                'from_address': from_address,
                'to_address': to_address,
                'block_number': str(tx.get('blockNumber')),
                'timestamp': tx.get('block_timestamp', 0) / 1000,
                'tx_url': f"https://tronscan.org/#/transaction/{tx_hash}"
            }
            
        except Exception as e:
            self.logger.error(f"Tron verification error: {e}")
            return {'verified': False, 'error': f'Tron API error: {str(e)}'}
    
    async def verify_ethereum_payment(self, tx_hash: str, expected_address: str, 
                                     min_amount: float) -> Dict:
        """Verify ERC20 USDT payment on Ethereum network"""
        try:
            api_key = self.api_endpoints['ERC20'].get('api_key', '')
            
            # Get transaction
            url = self.api_endpoints['ERC20']['main']
            params = {
                'module': 'proxy',
                'action': 'eth_getTransactionByHash',
                'txhash': tx_hash,
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data.get('result'):
                return {'verified': False, 'error': 'Transaction not found'}
            
            tx = data['result']
            
            # Verify USDT contract
            if tx.get('to', '').lower() != self.usdt_contracts['ERC20'].lower():
                return {'verified': False, 'error': 'Not a USDT transfer'}
            
            # Parse input data
            input_data = tx.get('input', '')
            
            # Check transfer method (0xa9059cbb)
            if not input_data.startswith('0xa9059cbb'):
                return {'verified': False, 'error': 'Not a transfer transaction'}
            
            # Extract recipient and amount
            recipient = '0x' + input_data[34:74]
            amount_hex = input_data[74:138] if len(input_data) >= 138 else '0'
            amount = int(amount_hex, 16) / 1e6  # USDT has 6 decimals
            
            # Verify recipient
            if recipient.lower() != expected_address.lower():
                return {
                    'verified': False,
                    'error': f'Wrong recipient. Expected: {expected_address}, Got: {recipient}'
                }
            
            # Verify amount
            if amount < min_amount:
                return {
                    'verified': False,
                    'error': f'Insufficient amount: ${amount:.2f} < ${min_amount}'
                }
            
            # Get receipt to confirm success
            params['action'] = 'eth_getTransactionReceipt'
            response = requests.get(url, params=params, timeout=10)
            receipt_data = response.json()
            
            if receipt_data.get('result'):
                receipt = receipt_data['result']
                
                if receipt.get('status') != '0x1':
                    return {'verified': False, 'error': 'Transaction failed'}
                
                return {
                    'verified': True,
                    'network': 'ERC20',
                    'amount': amount,
                    'from_address': tx.get('from'),
                    'to_address': recipient,
                    'block_number': str(int(receipt.get('blockNumber', '0x0'), 16)),
                    'gas_used': int(receipt.get('gasUsed', '0x0'), 16),
                    'tx_url': f"https://etherscan.io/tx/{tx_hash}"
                }
            
            return {'verified': False, 'error': 'Could not verify receipt'}
            
        except Exception as e:
            self.logger.error(f"Ethereum verification error: {e}")
            return {'verified': False, 'error': f'Ethereum API error: {str(e)}'}
    
    async def verify_bsc_payment(self, tx_hash: str, expected_address: str, 
                                min_amount: float) -> Dict:
        """Verify BEP20 USDT payment on BSC network"""
        try:
            api_key = self.api_endpoints['BEP20'].get('api_key', '')
            
            # Get transaction
            url = self.api_endpoints['BEP20']['main']
            params = {
                'module': 'proxy',
                'action': 'eth_getTransactionByHash',
                'txhash': tx_hash,
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data.get('result'):
                return {'verified': False, 'error': 'Transaction not found'}
            
            tx = data['result']
            
            # Verify USDT contract
            if tx.get('to', '').lower() != self.usdt_contracts['BEP20'].lower():
                return {'verified': False, 'error': 'Not a USDT transfer'}
            
            # Parse input data
            input_data = tx.get('input', '')
            
            if not input_data.startswith('0xa9059cbb'):
                return {'verified': False, 'error': 'Not a transfer transaction'}
            
            # Extract recipient and amount
            recipient = '0x' + input_data[34:74]
            amount_hex = input_data[74:138] if len(input_data) >= 138 else '0'
            amount = int(amount_hex, 16) / 1e18  # USDT on BSC has 18 decimals
            
            # Verify recipient
            if recipient.lower() != expected_address.lower():
                return {
                    'verified': False,
                    'error': f'Wrong recipient. Expected: {expected_address}, Got: {recipient}'
                }
            
            # Verify amount
            if amount < min_amount:
                return {
                    'verified': False,
                    'error': f'Insufficient amount: ${amount:.2f} < ${min_amount}'
                }
            
            # Get receipt
            params['action'] = 'eth_getTransactionReceipt'
            response = requests.get(url, params=params, timeout=10)
            receipt_data = response.json()
            
            if receipt_data.get('result'):
                receipt = receipt_data['result']
                
                if receipt.get('status') != '0x1':
                    return {'verified': False, 'error': 'Transaction failed'}
                
                return {
                    'verified': True,
                    'network': 'BEP20',
                    'amount': amount,
                    'from_address': tx.get('from'),
                    'to_address': recipient,
                    'block_number': str(int(receipt.get('blockNumber', '0x0'), 16)),
                    'tx_url': f"https://bscscan.com/tx/{tx_hash}"
                }
            
            return {'verified': False, 'error': 'Could not verify receipt'}
            
        except Exception as e:
            self.logger.error(f"BSC verification error: {e}")
            return {'verified': False, 'error': f'BSC API error: {str(e)}'}
    
    def tron_address_to_hex(self, address: str) -> str:
        """Convert Tron base58 address to hex"""
        try:
            return base58.b58decode_check(address).hex()
        except:
            return address
    
    def hex_to_tron_address(self, hex_addr: str) -> str:
        """Convert hex to Tron base58 address"""
        try:
            if hex_addr.startswith('0x'):
                hex_addr = hex_addr[2:]
            if not hex_addr.startswith('41'):
                hex_addr = '41' + hex_addr
            return base58.b58encode_check(bytes.fromhex(hex_addr)).decode()
        except:
            return hex_addr


class TelegramBootstrapManager:
    """Complete Telegram bot manager for multi-user trading system - Only /start command"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.bot = None
        self.application = None
        self.bootstrap_mode = False
        
        # Admin user ID
        self.ADMIN_ID = "6708641837"
        
        # Initialize managers
        self.db_manager = DatabaseManager(config.db_config.get_database_url())
        self.encryption_manager = SecretManager(config.encryption_password)
        self.blockchain_verifier = BlockchainVerifier()
        
        # State management
        self.pending_configurations = {}
        self.pending_payments = {}
        self.user_states = {}
        self.broadcast_states = {}
        
        # Pagination settings
        self.users_per_page = 5
        self.payments_per_page = 5
        
        # Initialize bot if token exists
        if config.telegram_bot_token:
            self.bot = Bot(token=config.telegram_bot_token)
    
    def should_enter_bootstrap_mode(self) -> bool:
        """Check if bootstrap mode is needed"""
        session = self.db_manager.get_session()
        try:
            config = session.query(SystemConfig).filter(
                SystemConfig.config_name == 'default'
            ).first()
            
            if not config:
                return True
            
            # Check deposit addresses
            if not config.deposit_addresses or not config.deposit_addresses.get('USDT_TRC20'):
                return True
            
            # Check admin user
            admin = session.query(User).filter(
                User.tier == UserTier.ADMIN
            ).first()
            
            if not admin or not admin.api_key:
                return True
                
            return False
            
        finally:
            session.close()
    
    async def fetch_deposit_addresses_from_exchange(self, api_key: str, api_secret: str) -> Dict[str, str]:
        """Auto-fetch deposit addresses using ExchangeManager"""
        try:
            self.logger.info("Fetching deposit addresses from exchange...")
            
            # Create temporary admin user for fetching
            session = self.db_manager.get_session()
            admin = session.query(User).filter(
                User.telegram_id == self.ADMIN_ID
            ).first()
            
            if not admin:
                admin = User(
                    telegram_id=self.ADMIN_ID,
                    telegram_username="SmartMoneyTraderAdmin",
                    tier=UserTier.ADMIN,
                    exchange=ExchangeType.BYBIT
                )
                session.add(admin)
            
            # Temporarily save encrypted credentials
            admin.api_key = self.encryption_manager.encrypt_secret(api_key)
            admin.api_secret = self.encryption_manager.encrypt_secret(api_secret)
            session.commit()
            
            # Initialize exchange manager
            exchange_manager = ExchangeManager(self.config, self.ADMIN_ID)
            
            if not exchange_manager.exchange:
                self.logger.error("Failed to initialize exchange")
                session.close()
                return {}
            
            # Fetch deposit addresses
            addresses = exchange_manager.fetch_deposit_addresses("USDT")
            
            # Cleanup
            exchange_manager.cleanup_connections()
            session.close()
            
            return addresses
            
        except Exception as e:
            self.logger.error(f"Error fetching deposit addresses: {e}")
            return {}
    
    async def start_bootstrap_mode(self) -> bool:
        """Start bootstrap mode for initial configuration"""
        try:
            self.bootstrap_mode = True
            self.logger.info("🔄 Bootstrap mode activated")
            
            # Database setup
            session = self.db_manager.get_session()
            self.db_manager.migrate_system_config(session)
            self.db_manager.migrate_admin_user(session)
            self.db_manager.check_and_downgrade_expired_users(session)
            session.close()
            
            # Initialize Telegram bot
            self.application = Application.builder().token(self.config.telegram_bot_token).build()
            
            # Register ONLY /start command handler
            self.application.add_handler(CommandHandler("start", self.handle_start))
            
            # Register callback handler for inline keyboards
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            # Register message handler for text input
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            # Send initial notification to admin
            await self.send_bootstrap_notification()
            
            # Start the bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            self.logger.info("✅ Bootstrap mode started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start bootstrap mode: {e}")
            return False
    
    async def send_bootstrap_notification(self):
        """Send bootstrap notification to admin"""
        try:
            message = "🔧 **System Bootstrap Mode Active**\n\n"
            message += "Welcome to Smart Money Trader Multi-User System!\n\n"
            message += "Please complete the initial setup:\n\n"
            message += "1️⃣ Configure your admin API credentials\n"
            message += "2️⃣ Set up deposit addresses\n"
            message += "3️⃣ Start managing users\n\n"
            message += "Click the button below to begin:"
            
            keyboard = [
                [InlineKeyboardButton("🔑 Configure Admin API", callback_data="config_admin_api")],
                [InlineKeyboardButton("📊 Check Status", callback_data="check_status")],
                [InlineKeyboardButton("❓ Help", callback_data="show_help")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.bot.send_message(
                chat_id=self.config.telegram_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send bootstrap notification: {e}")
    
    # Main Command Handler - ONLY /start
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - The ONLY command handler"""
        try:
            user = update.effective_user
            user_id = str(user.id)
            username = user.username or user.first_name or "User"
            
            session = self.db_manager.get_session()
            
            # Get or create user
            db_user = session.query(User).filter(
                User.telegram_id == user_id
            ).first()
            
            if not db_user:
                # New user registration
                new_user = User(
                    telegram_id=user_id,
                    telegram_username=username,
                    tier=UserTier.FREE,
                    exchange=ExchangeType.BYBIT,
                    is_active=True
                )
                session.add(new_user)
                session.commit()
                db_user = new_user
                
                message = "🤖 **Welcome to Smart Money Trader!**\n\n"
                message += f"Hello {username}! You've been registered successfully.\n\n"
                message += "📊 **Current Status:** FREE User\n\n"
                message += "**Free Tier Features:**\n"
                message += "• View trading signals\n"
                message += "• Market analysis updates\n"
                message += "• Educational content\n\n"
                message += "**💎 Upgrade to PAID ($100/month) for:**\n"
                message += "• 5 automated trades daily\n"
                message += "• API integration\n"
                message += "• Priority notifications\n"
                message += "• Advanced analytics\n\n"
                message += "Choose an option below:"
                
                keyboard = [
                    [InlineKeyboardButton("💳 Upgrade to PAID", callback_data="upgrade_account")],
                    [InlineKeyboardButton("📊 My Status", callback_data="view_status")],
                    [InlineKeyboardButton("⚙️ Configure API", callback_data="configure_api")],
                    [InlineKeyboardButton("❓ Help", callback_data="show_help")]
                ]
                
            else:
                # Existing user - Check if admin
                if user_id == self.ADMIN_ID:
                    # ADMIN USER - Show admin menu
                    message = f"👑 **Welcome back, Admin!**\n\n"
                    message += "System Status: ✅ Operational\n"
                    message += f"Total Users: {session.query(User).count()}\n"
                    message += f"Active Subscriptions: {session.query(User).filter(User.tier.in_([UserTier.PAID, UserTier.AWOOF])).count()}\n\n"
                    message += "Select an action:"
                    
                    keyboard = [
                        [InlineKeyboardButton("👥 Manage Users", callback_data="manage_users")],
                        [InlineKeyboardButton("💰 View Payments", callback_data="view_payments")],
                        [InlineKeyboardButton("📊 System Stats", callback_data="system_stats")],
                        [InlineKeyboardButton("⚙️ Admin Settings", callback_data="admin_settings")],
                        [InlineKeyboardButton("🔑 Update API", callback_data="config_admin_api")],
                        [InlineKeyboardButton("📢 Broadcast Message", callback_data="broadcast_message")]
                    ]
                    
                else:
                    # REGULAR USER - Show user menu
                    tier_info = {
                        UserTier.FREE: ("🆓", "FREE", "Upgrade to unlock trading"),
                        UserTier.PAID: ("💎", "PAID", "Premium features active"),
                        UserTier.AWOOF: ("🎁", "AWOOF", "Special access granted")
                    }
                    
                    emoji, tier_name, tier_desc = tier_info.get(db_user.tier, ("", "Unknown", ""))
                    
                    message = f"{emoji} **Welcome back, {username}!**\n\n"
                    message += f"**Tier:** {tier_name}\n"
                    message += f"**Status:** {tier_desc}\n"
                    
                    if db_user.subscription_expires_at and db_user.tier != UserTier.FREE:
                        days_left = (db_user.subscription_expires_at - datetime.utcnow()).days
                        if days_left > 0:
                            message += f"**Subscription:** {days_left} days remaining\n"
                        else:
                            message += f"**Subscription:** Expired\n"
                    
                    message += f"\n**Trading Stats:**\n"
                    message += f"• Total Trades: {db_user.total_trades}\n"
                    message += f"• Successful: {db_user.successful_trades}\n"
                    
                    if db_user.total_trades > 0:
                        success_rate = (db_user.successful_trades / db_user.total_trades * 100)
                        message += f"• Success Rate: {success_rate:.1f}%\n"
                        
                    if db_user.total_pnl != 0:
                        message += f"• Total P&L: ${db_user.total_pnl:,.2f}\n"
                    
                    message += "\nSelect an option:"
                    
                    keyboard = [
                        [InlineKeyboardButton("📊 My Status", callback_data="view_status")],
                        [InlineKeyboardButton("⚙️ Configure API", callback_data="configure_api")]
                    ]
                    
                    if db_user.tier == UserTier.FREE:
                        keyboard.insert(0, [InlineKeyboardButton("💳 Upgrade to PAID", callback_data="upgrade_account")])
                    else:
                        keyboard.append([InlineKeyboardButton("📈 Trading History", callback_data="trading_history")])
                    
                    keyboard.append([InlineKeyboardButton("❓ Help", callback_data="show_help")])
            
            session.close()
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                message, 
                reply_markup=reply_markup, 
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            self.logger.error(f"Error in handle_start: {e}")
            await update.message.reply_text(
                "❌ An error occurred. Please try again later.",
                parse_mode=ParseMode.MARKDOWN
            )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all inline keyboard callbacks"""
        try:
            query = update.callback_query
            await query.answer()
            
            user_id = str(query.from_user.id)
            callback_data = query.data
            
            # Check if admin-only function
            admin_only_callbacks = [
                "config_admin_api", "manage_users", "view_payments", "system_stats",
                "admin_settings", "broadcast_message", "search_user"
            ]
            
            # Also check dynamic admin callbacks
            admin_only_prefixes = [
                "upgrade_user_", "downgrade_user_", "ban_user_", "unban_user_",
                "page_users_", "page_payments_", "view_user_"
            ]
            
            # Check if this is an admin-only callback
            is_admin_callback = callback_data in admin_only_callbacks or \
                              any(callback_data.startswith(prefix) for prefix in admin_only_prefixes)
            
            # Verify admin access for admin-only functions
            if is_admin_callback and user_id != self.ADMIN_ID:
                await query.edit_message_text("❌ Unauthorized: Admin access only")
                return
            
            # Main handlers dictionary
            handlers = {
                # Admin functions
                "config_admin_api": self.start_admin_api_config,
                "manage_users": self.show_user_management,
                "view_payments": self.show_payments,
                "system_stats": self.show_system_stats,
                "admin_settings": self.show_admin_settings,
                "broadcast_message": self.start_broadcast_message,
                "search_user": self.start_user_search,
                
                # User functions
                "upgrade_account": self.show_upgrade_options,
                "view_status": self.show_user_status,
                "configure_api": self.start_user_api_config,
                "trading_history": self.show_trading_history,
                
                # Payment functions
                "verify_payment": self.start_payment_verification,
                "retry_payment": self.start_payment_verification,
                
                # System functions
                "check_status": self.show_bootstrap_status,
                "show_help": self.show_help,
                "cancel_config": self.cancel_configuration,
                "confirm_deposit_save": self.confirm_deposit_addresses,
                "back_to_menu": self.back_to_main_menu,
                "main_menu": self.back_to_main_menu,
                
                # Cancel functions
                "cancel_broadcast": self.cancel_broadcast,
                "cancel_payment": self.cancel_payment,
            }
            
            # Handle static callbacks
            if callback_data in handlers:
                await handlers[callback_data](query)
            # Handle dynamic callbacks
            elif callback_data.startswith("upgrade_user_"):
                await self.admin_upgrade_user(query)
            elif callback_data.startswith("downgrade_user_"):
                await self.admin_downgrade_user(query)
            elif callback_data.startswith("ban_user_"):
                await self.admin_ban_user(query)
            elif callback_data.startswith("unban_user_"):
                await self.admin_unban_user(query)
            elif callback_data.startswith("view_user_"):
                await self.view_user_details(query)
            elif callback_data.startswith("page_users_"):
                await self.paginate_users(query)
            elif callback_data.startswith("page_payments_"):
                await self.paginate_payments(query)
            elif callback_data.startswith("select_exchange_"):
                await self.select_exchange(query)
            else:
                await query.edit_message_text("⚠️ Unknown action")
                
        except Exception as e:
            self.logger.error(f"Error in handle_callback: {e}")
            await query.edit_message_text("❌ An error occurred. Please try again.")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages for configuration and payments"""
        try:
            user_id = str(update.effective_user.id)
            text = update.message.text.strip()
            
            # Check for pending operations
            if user_id in self.pending_configurations:
                await self.handle_configuration_input(update, user_id, text)
            elif user_id in self.pending_payments:
                await self.handle_payment_input(update, user_id, text)
            elif user_id in self.broadcast_states:
                await self.handle_broadcast_input(update, user_id, text)
            elif user_id in self.user_states:
                await self.handle_user_state_input(update, user_id, text)
            else:
                # No pending operation - remind user to use /start
                await update.message.reply_text(
                    "Please use /start to access the menu.",
                    parse_mode=ParseMode.MARKDOWN
                )
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    # Configuration Handlers
    async def handle_configuration_input(self, update: Update, user_id: str, text: str):
        """Handle API configuration input"""
        try:
            config = self.pending_configurations[user_id]
            step = config['step']
            
            if step == 'api_key':
                config['data']['api_key'] = text
                config['step'] = 'api_secret'
                
                await update.message.reply_text("✅ API Key received!\n\nNow send your API Secret:")
                await update.message.delete()  # Delete sensitive data
                
            elif step == 'api_secret':
                config['data']['api_secret'] = text
                await update.message.delete()  # Delete sensitive data
                
                # Check if admin configuration
                if config.get('type') == 'admin' and user_id == self.ADMIN_ID:
                    status_msg = await update.message.reply_text("🔄 Fetching deposit addresses...")
                    
                    addresses = await self.fetch_deposit_addresses_from_exchange(
                        config['data']['api_key'],
                        config['data']['api_secret']
                    )
                    
                    if addresses:
                        config['data']['addresses'] = addresses
                        
                        message = "✅ **Deposit Addresses Found!**\n\n"
                        for key, addr in addresses.items():
                            network = key.replace('USDT_', '')
                            message += f"**{network}:**\n`{addr}`\n\n"
                        
                        message += "⚠️ **Please verify these addresses!**"
                        
                        keyboard = [
                            [InlineKeyboardButton("✅ Confirm & Save", callback_data="confirm_deposit_save")],
                            [InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]
                        ]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        
                        await status_msg.edit_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
                    else:
                        # Manual entry fallback
                        await status_msg.edit_text("⚠️ Could not fetch addresses. Manual setup required.")
                        config['step'] = 'manual_trc20'
                        await update.message.reply_text("Enter USDT TRC20 address:")
                else:
                    # Regular user - save API
                    await self.save_user_api_credentials(update, user_id, config['data'])
                    
            elif step.startswith('manual_'):
                # Handle manual address entry
                network = step.replace('manual_', '').upper()
                
                if 'addresses' not in config['data']:
                    config['data']['addresses'] = {}
                
                config['data']['addresses'][f'USDT_{network}'] = text
                
                # Move to next network
                if network == 'TRC20':
                    config['step'] = 'manual_erc20'
                    await update.message.reply_text("Enter USDT ERC20 address:")
                elif network == 'ERC20':
                    config['step'] = 'manual_bep20'
                    await update.message.reply_text("Enter USDT BEP20 address:")
                else:
                    # All addresses collected
                    message = "✅ **Addresses Collected**\n\n"
                    for key, addr in config['data']['addresses'].items():
                        network = key.replace('USDT_', '')
                        message += f"**{network}:**\n`{addr}`\n\n"
                    
                    keyboard = [
                        [InlineKeyboardButton("✅ Save", callback_data="confirm_deposit_save")],
                        [InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await update.message.reply_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
                    
        except Exception as e:
            self.logger.error(f"Error in configuration input: {e}")
            await update.message.reply_text("❌ Configuration error. Please try again.")
    
    async def handle_payment_input(self, update: Update, user_id: str, tx_hash: str):
        """Handle payment verification input"""
        try:
            status_msg = await update.message.reply_text("🔄 Verifying transaction on blockchain...")
            
            # Get system config
            session = self.db_manager.get_session()
            config = session.query(SystemConfig).filter(
                SystemConfig.config_name == 'default'
            ).first()
            
            if not config or not config.deposit_addresses:
                await status_msg.edit_text("❌ System not configured. Contact admin.")
                del self.pending_payments[user_id]
                session.close()
                return
            
            # Try to verify payment on all networks
            verification_result = None
            for network, address in config.deposit_addresses.items():
                if address:
                    network_type = network.replace('USDT_', '')
                    result = await self.blockchain_verifier.verify_payment(
                        tx_hash, address, config.subscription_fee, network_type
                    )
                    
                    if result.get('verified'):
                        verification_result = result
                        break
            
            if verification_result and verification_result['verified']:
                # Check if already used
                existing = session.query(Subscription).filter(
                    Subscription.blockchain_hash == tx_hash
                ).first()
                
                if existing:
                    await status_msg.edit_text("❌ This transaction has already been used!")
                else:
                    # Process successful payment
                    await self.process_successful_payment(
                        status_msg, session, user_id, tx_hash, verification_result
                    )
            else:
                # Payment failed
                error = verification_result.get('error', 'Unknown error') if verification_result else 'Transaction not found'
                
                message = f"❌ **Payment Verification Failed**\n\n"
                message += f"**Error:** {error}\n\n"
                message += "Please check:\n"
                message += "• Transaction hash is correct\n"
                message += "• Sent to correct address\n"
                message += f"• Amount ≥ ${config.subscription_fee} USDT\n"
                message += "• Transaction is confirmed\n"
                
                keyboard = [
                    [InlineKeyboardButton("🔄 Try Again", callback_data="verify_payment")],
                    [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await status_msg.edit_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            del self.pending_payments[user_id]
            
        except Exception as e:
            self.logger.error(f"Error verifying payment: {e}")
            await update.message.reply_text("❌ Verification error. Contact support.")
            if user_id in self.pending_payments:
                del self.pending_payments[user_id]
    
    async def handle_broadcast_input(self, update: Update, user_id: str, text: str):
        """Handle broadcast message input"""
        try:
            if user_id != self.ADMIN_ID:
                return
            
            # Send broadcast to all users
            session = self.db_manager.get_session()
            users = session.query(User).filter(User.is_active == True).all()
            
            success_count = 0
            fail_count = 0
            
            status_msg = await update.message.reply_text(f"📤 Sending to {len(users)} users...")
            
            for user in users:
                if user.telegram_id != user_id:  # Don't send to admin
                    try:
                        await self.bot.send_message(
                            chat_id=user.telegram_id,
                            text=f"📢 **Announcement**\n\n{text}",
                            parse_mode=ParseMode.MARKDOWN
                        )
                        success_count += 1
                    except:
                        fail_count += 1
            
            session.close()
            del self.broadcast_states[user_id]
            
            message = f"✅ Broadcast complete!\n\n"
            message += f"Sent: {success_count}\n"
            message += f"Failed: {fail_count}"
            
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await status_msg.edit_text(message, reply_markup=reply_markup)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting: {e}")
    
    async def handle_user_state_input(self, update: Update, user_id: str, text: str):
        """Handle various user state inputs"""
        state = self.user_states.get(user_id, {})
        
        if state.get('action') == 'search_user':
            # Search for user
            session = self.db_manager.get_session()
            users = session.query(User).filter(
                (User.telegram_username.like(f'%{text}%')) |
                (User.telegram_id.like(f'%{text}%'))
            ).limit(10).all()
            
            if users:
                message = "🔍 **Search Results:**\n\n"
                keyboard = []
                
                for user in users:
                    message += f"@{user.telegram_username} ({user.tier.value})\n"
                    keyboard.append([
                        InlineKeyboardButton(
                            f"@{user.telegram_username}",
                            callback_data=f"view_user_{user.telegram_id}"
                        )
                    ])
                
                keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="manage_users")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text("No users found. Use /start to go back.")
            
            session.close()
            del self.user_states[user_id]
    
    # Payment Processing
    async def process_successful_payment(self, status_msg, session, user_id: str, 
                                        tx_hash: str, verification_result: Dict):
        """Process a successful payment verification"""
        try:
            user = session.query(User).filter(User.telegram_id == user_id).first()
            
            if not user:
                await status_msg.edit_text("❌ User not found!")
                return
            
            # Update user tier
            old_tier = user.tier
            user.tier = UserTier.PAID
            user.extend_subscription(30)
            
            # Record subscription
            subscription = Subscription(
                telegram_id=user_id,
                payment_date=datetime.utcnow(),
                blockchain_hash=tx_hash,
                expiry_date=user.subscription_expires_at,
                amount=verification_result['amount'],
                network=verification_result['network'],
                is_verified=True,
                verification_time=datetime.utcnow(),
                block_number=str(verification_result.get('block_number', '')),
                from_address=verification_result.get('from_address', ''),
                to_address=verification_result.get('to_address', '')
            )
            session.add(subscription)
            session.commit()
            
            # Success message
            message = "✅ **Payment Verified Successfully!**\n\n"
            message += f"**Amount:** ${verification_result['amount']:.2f} USDT\n"
            message += f"**Network:** {verification_result['network']}\n"
            message += f"**Block:** {verification_result.get('block_number', 'N/A')}\n\n"
            message += "🎉 **Account Upgraded to PAID Tier!**\n\n"
            message += f"**Valid Until:** {user.subscription_expires_at.strftime('%Y-%m-%d')}\n\n"
            message += "**You now have access to:**\n"
            message += "• 5 automated trades per day\n"
            message += "• Full API integration\n"
            message += "• Priority notifications\n"
            message += "• Advanced analytics\n\n"
            
            if verification_result.get('tx_url'):
                message += f"[View Transaction]({verification_result['tx_url']})"
            
            keyboard = [
                [InlineKeyboardButton("⚙️ Configure API", callback_data="configure_api")],
                [InlineKeyboardButton("📊 View Status", callback_data="view_status")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await status_msg.edit_text(
                message, 
                reply_markup=reply_markup, 
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            
            # Log the upgrade
            self.logger.info(f"User {user.telegram_username} upgraded from {old_tier.value} to PAID")
            
        except Exception as e:
            self.logger.error(f"Error processing payment: {e}")
            await status_msg.edit_text("❌ Error processing payment. Contact admin.")
    
    # Admin Functions - All require admin check
    async def start_admin_api_config(self, query: CallbackQuery):
        """Start admin API configuration"""
        try:
            user_id = str(query.from_user.id)
            
            # Admin check already done in handle_callback
            
            self.pending_configurations[user_id] = {
                'step': 'api_key',
                'data': {},
                'type': 'admin'
            }
            
            message = "🔑 **Admin API Configuration**\n\n"
            message += "Please send your Bybit Live API Key.\n\n"
            message += "⚠️ **Requirements:**\n"
            message += "• Spot trading permissions\n"
            message += "• Wallet permissions (for deposits)\n"
            message += "• Futures trading (if needed)\n\n"
            message += "Your API key will be encrypted before storage."
            
            keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error starting admin API config: {e}")
    
    async def show_user_management(self, query: CallbackQuery):
        """Show user management panel - Admin only"""
        try:
            session = self.db_manager.get_session()
            
            # Get user statistics
            total_users = session.query(User).count()
            free_users = session.query(User).filter(User.tier == UserTier.FREE).count()
            paid_users = session.query(User).filter(User.tier == UserTier.PAID).count()
            awoof_users = session.query(User).filter(User.tier == UserTier.AWOOF).count()
            banned_users = session.query(User).filter(User.is_banned == True).count()
            
            message = "👥 **User Management**\n\n"
            message += f"**Total Users:** {total_users}\n"
            message += f"• Free: {free_users}\n"
            message += f"• Paid: {paid_users}\n"
            message += f"• Awoof: {awoof_users}\n"
            message += f"• Banned: {banned_users}\n\n"
            message += "Select an option:"
            
            keyboard = [
                [InlineKeyboardButton("📋 List All Users", callback_data="page_users_all_0")],
                [InlineKeyboardButton("💎 Paid Users", callback_data="page_users_paid_0")],
                [InlineKeyboardButton("🎁 Awoof Users", callback_data="page_users_awoof_0")],
                [InlineKeyboardButton("🚫 Banned Users", callback_data="page_users_banned_0")],
                [InlineKeyboardButton("🔍 Search User", callback_data="search_user")],
                [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error showing user management: {e}")
    
    async def show_payments(self, query: CallbackQuery):
        """Show payment history - Admin only"""
        try:
            await self.paginate_payments(query, page=0)
            
        except Exception as e:
            self.logger.error(f"Error showing payments: {e}")
    
    async def show_system_stats(self, query: CallbackQuery):
        """Show system statistics - Admin only"""
        try:
            await self.show_system_statistics(query.message)
            
        except Exception as e:
            self.logger.error(f"Error showing system stats: {e}")
    
    async def show_system_statistics(self, message):
        """Display detailed system statistics"""
        try:
            session = self.db_manager.get_session()
            
            # User stats
            total_users = session.query(User).count()
            active_users = session.query(User).filter(User.is_active == True).count()
            tier_stats = {}
            for tier in UserTier:
                count = session.query(User).filter(User.tier == tier).count()
                tier_stats[tier.value] = count
            
            # Payment stats
            total_payments = session.query(Subscription).count()
            total_revenue = session.query(Subscription).with_entities(
                Subscription.amount
            ).all()
            revenue_sum = sum(p[0] for p in total_revenue) if total_revenue else 0
            
            # Recent activity
            recent_users = session.query(User).order_by(
                User.created_at.desc()
            ).limit(3).all()
            
            recent_payments = session.query(Subscription).order_by(
                Subscription.created_at.desc()
            ).limit(3).all()
            
            message_text = "📊 **System Statistics**\n\n"
            
            message_text += "**👥 Users:**\n"
            message_text += f"• Total: {total_users}\n"
            message_text += f"• Active: {active_users}\n"
            for tier, count in tier_stats.items():
                message_text += f"• {tier.capitalize()}: {count}\n"
            
            message_text += f"\n**💰 Revenue:**\n"
            message_text += f"• Total Payments: {total_payments}\n"
            message_text += f"• Total Revenue: ${revenue_sum:,.2f}\n"
            message_text += f"• Average Payment: ${(revenue_sum/total_payments if total_payments else 0):,.2f}\n"
            
            message_text += f"\n**🕐 Recent Activity:**\n"
            
            if recent_users:
                message_text += "\nNew Users:\n"
                for user in recent_users[:3]:
                    message_text += f"• @{user.telegram_username} ({user.created_at.strftime('%Y-%m-%d')})\n"
            
            if recent_payments:
                message_text += "\nRecent Payments:\n"
                for payment in recent_payments[:3]:
                    message_text += f"• ${payment.amount:.2f} ({payment.network})\n"
            
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if hasattr(message, 'edit_text'):
                await message.edit_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            else:
                await message.reply_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error showing statistics: {e}")
    
    # Admin User Management Methods
    async def admin_upgrade_user(self, query: CallbackQuery):
        """Upgrade user to awoof tier - Admin only"""
        try:
            target_user_id = query.data.replace("upgrade_user_", "")
            
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == target_user_id).first()
            
            if user and user.tier != UserTier.ADMIN:
                old_tier = user.tier
                user.tier = UserTier.AWOOF
                user.extend_subscription(30)
                session.commit()
                
                message = f"✅ User @{user.telegram_username} upgraded!\n\n"
                message += f"From: {old_tier.value}\n"
                message += f"To: AWOOF\n"
                message += f"Valid for: 30 days"
                
                keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f"view_user_{target_user_id}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
                
                # Notify user
                try:
                    await self.bot.send_message(
                        chat_id=target_user_id,
                        text="🎁 **Great News!**\n\nYou've been upgraded to AWOOF tier by admin!\n\n30 days of premium access activated!",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except:
                    pass
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error upgrading user: {e}")
    
    async def admin_downgrade_user(self, query: CallbackQuery):
        """Downgrade user tier - Admin only"""
        try:
            target_user_id = query.data.replace("downgrade_user_", "")
            
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == target_user_id).first()
            
            if user and user.tier != UserTier.ADMIN:
                old_tier = user.tier
                user.tier = UserTier.FREE
                user.subscription_expires_at = None
                session.commit()
                
                message = f"✅ User @{user.telegram_username} downgraded!\n\n"
                message += f"From: {old_tier.value}\n"
                message += f"To: FREE"
                
                keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f"view_user_{target_user_id}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
                
                # Notify user
                try:
                    await self.bot.send_message(
                        chat_id=target_user_id,
                        text="ℹ️ Your account has been downgraded to FREE tier.",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except:
                    pass
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error downgrading user: {e}")
    
    async def admin_ban_user(self, query: CallbackQuery):
        """Ban a user - Admin only"""
        try:
            target_user_id = query.data.replace("ban_user_", "")
            
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == target_user_id).first()
            
            if user and user.tier != UserTier.ADMIN:
                user.is_banned = True
                user.ban_reason = "Admin action"
                session.commit()
                
                message = f"🚫 User @{user.telegram_username} has been banned!"
                
                keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f"view_user_{target_user_id}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error banning user: {e}")
    
    async def admin_unban_user(self, query: CallbackQuery):
        """Unban a user - Admin only"""
        try:
            target_user_id = query.data.replace("unban_user_", "")
            
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == target_user_id).first()
            
            if user:
                user.is_banned = False
                user.ban_reason = None
                session.commit()
                
                message = f"✅ User @{user.telegram_username} has been unbanned!"
                
                keyboard = [[InlineKeyboardButton("🔙 Back", callback_data=f"view_user_{target_user_id}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error unbanning user: {e}")
    
    async def view_user_details(self, query: CallbackQuery):
        """View detailed user information - Admin only"""
        try:
            target_user_id = query.data.replace("view_user_", "")
            
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == target_user_id).first()
            
            if not user:
                await query.edit_message_text("❌ User not found")
                session.close()
                return
            
            tier_emoji = {
                UserTier.ADMIN: "👑",
                UserTier.PAID: "💎",
                UserTier.AWOOF: "🎁",
                UserTier.FREE: "🆓"
            }.get(user.tier, "")
            
            message = f"{tier_emoji} **User Details**\n\n"
            message += f"**Username:** @{user.telegram_username}\n"
            message += f"**User ID:** `{user.telegram_id}`\n"
            message += f"**Tier:** {user.tier.value}\n"
            message += f"**Exchange:** {user.exchange.value if user.exchange else 'Not set'}\n"
            message += f"**API:** {'✅ Configured' if user.api_key else '❌ Not configured'}\n"
            message += f"**Status:** {'🚫 BANNED' if user.is_banned else '✅ Active'}\n"
            
            if user.ban_reason:
                message += f"**Ban Reason:** {user.ban_reason}\n"
            
            if user.subscription_expires_at:
                message += f"**Subscription Expires:** {user.subscription_expires_at.strftime('%Y-%m-%d')}\n"
            
            message += f"\n**Statistics:**\n"
            message += f"• Total Trades: {user.total_trades}\n"
            message += f"• Successful: {user.successful_trades}\n"
            message += f"• Total P&L: ${user.total_pnl:,.2f}\n"
            message += f"• Joined: {user.created_at.strftime('%Y-%m-%d')}\n"
            
            # Build action buttons
            keyboard = []
            
            if user.tier == UserTier.FREE:
                keyboard.append([InlineKeyboardButton("⬆️ Upgrade to AWOOF", callback_data=f"upgrade_user_{target_user_id}")])
            elif user.tier in [UserTier.PAID, UserTier.AWOOF]:
                keyboard.append([InlineKeyboardButton("⬇️ Downgrade to FREE", callback_data=f"downgrade_user_{target_user_id}")])
            
            if user.is_banned:
                keyboard.append([InlineKeyboardButton("✅ Unban User", callback_data=f"unban_user_{target_user_id}")])
            elif user.tier != UserTier.ADMIN:
                keyboard.append([InlineKeyboardButton("🚫 Ban User", callback_data=f"ban_user_{target_user_id}")])
            
            keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="manage_users")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error viewing user details: {e}")
    
    async def paginate_users(self, query: CallbackQuery):
        """Paginate through users list - Admin only"""
        try:
            # Parse callback data
            parts = query.data.split("_")
            filter_type = parts[2] if len(parts) > 3 else "all"
            page = int(parts[-1])
            
            session = self.db_manager.get_session()
            
            # Build query based on filter
            user_query = session.query(User)
            
            if filter_type == "paid":
                user_query = user_query.filter(User.tier == UserTier.PAID)
            elif filter_type == "awoof":
                user_query = user_query.filter(User.tier == UserTier.AWOOF)
            elif filter_type == "banned":
                user_query = user_query.filter(User.is_banned == True)
            elif filter_type == "free":
                user_query = user_query.filter(User.tier == UserTier.FREE)
            
            # Get total count
            total = user_query.count()
            total_pages = (total + self.users_per_page - 1) // self.users_per_page
            
            # Get users for current page
            users = user_query.offset(page * self.users_per_page).limit(self.users_per_page).all()
            
            message = f"👥 **Users ({filter_type.capitalize()})** - Page {page + 1}/{max(total_pages, 1)}\n\n"
            
            for user in users:
                emoji = {
                    UserTier.ADMIN: "👑",
                    UserTier.PAID: "💎",
                    UserTier.AWOOF: "🎁",
                    UserTier.FREE: "🆓"
                }.get(user.tier, "")
                
                message += f"{emoji} @{user.telegram_username}\n"
                message += f"   ID: `{user.telegram_id}`\n"
                if user.is_banned:
                    message += f"   🚫 BANNED\n"
                message += "\n"
            
            # Build pagination keyboard
            keyboard = []
            
            # User action buttons
            for user in users:
                keyboard.append([
                    InlineKeyboardButton(
                        f"View @{user.telegram_username}",
                        callback_data=f"view_user_{user.telegram_id}"
                    )
                ])
            
            # Navigation buttons
            nav_buttons = []
            if page > 0:
                nav_buttons.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"page_users_{filter_type}_{page-1}"))
            if page < total_pages - 1:
                nav_buttons.append(InlineKeyboardButton("➡️ Next", callback_data=f"page_users_{filter_type}_{page+1}"))
            
            if nav_buttons:
                keyboard.append(nav_buttons)
            
            keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="manage_users")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error paginating users: {e}")
    
    async def paginate_payments(self, query: CallbackQuery, page: int = 0):
        """Paginate through payments - Admin only"""
        try:
            # Parse page from callback data if provided
            if isinstance(query, CallbackQuery) and query.data.startswith("page_payments_"):
                page = int(query.data.split("_")[-1])
            
            session = self.db_manager.get_session()
            
            # Get total count
            total = session.query(Subscription).count()
            total_pages = (total + self.payments_per_page - 1) // self.payments_per_page
            
            # Get payments for current page
            payments = session.query(Subscription).order_by(
                Subscription.created_at.desc()
            ).offset(page * self.payments_per_page).limit(self.payments_per_page).all()
            
            message = f"💰 **Payment History** - Page {page + 1}/{max(total_pages, 1)}\n\n"
            
            for payment in payments:
                user = session.query(User).filter(
                    User.telegram_id == payment.telegram_id
                ).first()
                
                username = user.telegram_username if user else "Unknown"
                
                message += f"**User:** @{username}\n"
                message += f"**Amount:** ${payment.amount:.2f} USDT\n"
                message += f"**Network:** {payment.network}\n"
                message += f"**Date:** {payment.payment_date.strftime('%Y-%m-%d %H:%M')}\n"
                message += f"**Hash:** `{payment.blockchain_hash[:20]}...`\n"
                message += "─────────────\n"
            
            if not payments:
                message += "No payments found."
            
            # Build pagination keyboard
            keyboard = []
            
            nav_buttons = []
            if page > 0:
                nav_buttons.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"page_payments_{page-1}"))
            if page < total_pages - 1:
                nav_buttons.append(InlineKeyboardButton("➡️ Next", callback_data=f"page_payments_{page+1}"))
            
            if nav_buttons:
                keyboard.append(nav_buttons)
            
            keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="main_menu")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error paginating payments: {e}")
    
    async def show_admin_settings(self, query: CallbackQuery):
        """Show admin settings panel - Admin only"""
        try:
            session = self.db_manager.get_session()
            config = session.query(SystemConfig).filter(
                SystemConfig.config_name == 'default'
            ).first()
            
            message = "⚙️ **Admin Settings**\n\n"
            message += f"**Subscription Fee:** ${config.subscription_fee if config else 100}\n"
            message += f"**Sandbox Mode:** {'✅ Enabled' if config and config.sandbox_mode else '❌ Disabled'}\n"
            message += f"**Auto Trading:** {'✅ Enabled' if config and config.enable_auto_trading else '❌ Disabled'}\n\n"
            
            message += "**Deposit Addresses:**\n"
            if config and config.deposit_addresses:
                for network, addr in config.deposit_addresses.items():
                    if addr:
                        message += f"• {network}: `{addr[:20]}...`\n"
            else:
                message += "Not configured\n"
            
            keyboard = [
                [InlineKeyboardButton("🔑 Update API", callback_data="config_admin_api")],
                [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error showing admin settings: {e}")
    
    async def start_broadcast_message(self, query: CallbackQuery):
        """Start broadcast message to users - Admin only"""
        try:
            user_id = str(query.from_user.id)
            
            self.broadcast_states[user_id] = {'step': 'message'}
            
            message = "📢 **Broadcast Message**\n\n"
            message += "Send the message you want to broadcast to all active users.\n\n"
            message += "Markdown formatting is supported.\n\n"
            message += "Send your message now:"
            
            keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_broadcast")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error starting broadcast: {e}")
    
    async def start_user_search(self, query: CallbackQuery):
        """Start user search - Admin only"""
        try:
            user_id = str(query.from_user.id)
            
            self.user_states[user_id] = {'action': 'search_user'}
            
            message = "🔍 **Search User**\n\n"
            message += "Send username or user ID to search:"
            
            keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="manage_users")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error starting user search: {e}")
    
    # User Functions
    async def show_upgrade_options(self, query: CallbackQuery):
        """Show upgrade payment options - For all users"""
        try:
            session = self.db_manager.get_session()
            config = session.query(SystemConfig).filter(
                SystemConfig.config_name == 'default'
            ).first()
            
            if not config or not config.deposit_addresses:
                await query.edit_message_text("❌ System not configured. Contact admin.")
                session.close()
                return
            
            message = "💳 **Upgrade to PAID Tier**\n\n"
            message += f"**Price:** ${config.subscription_fee} USDT/month\n\n"
            message += "**Benefits:**\n"
            message += "✅ 5 automated trades per day\n"
            message += "✅ Full API integration\n"
            message += "✅ Priority notifications\n"
            message += "✅ Advanced analytics\n\n"
            message += "**Send payment to any address:**\n\n"
            
            addresses = config.deposit_addresses
            
            if addresses.get('USDT_TRC20'):
                message += "**TRC20 (Recommended - Low fees):**\n"
                message += f"`{addresses['USDT_TRC20']}`\n\n"
            
            if addresses.get('USDT_ERC20'):
                message += "**ERC20 (Ethereum):**\n"
                message += f"`{addresses['USDT_ERC20']}`\n\n"
            
            if addresses.get('USDT_BEP20'):
                message += "**BEP20 (BSC):**\n"
                message += f"`{addresses['USDT_BEP20']}`\n\n"
            
            message += "After payment, click 'Verify Payment' below."
            
            keyboard = [
                [InlineKeyboardButton("✅ Verify Payment", callback_data="verify_payment")],
                [InlineKeyboardButton("🔙 Back", callback_data="main_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error showing upgrade options: {e}")
    
    async def start_payment_verification(self, query: CallbackQuery):
        """Start payment verification process - For all users"""
        try:
            user_id = str(query.from_user.id)
            
            self.pending_payments[user_id] = {
                'step': 'tx_hash',
                'timestamp': datetime.utcnow()
            }
            
            message = "🔍 **Payment Verification**\n\n"
            message += "Please send your transaction hash.\n\n"
            message += "**Examples:**\n"
            message += "TRC20: `abc123def456...` (64 chars)\n"
            message += "ERC20/BEP20: `0x123abc456def...` (66 chars)\n\n"
            message += "The system will automatically:\n"
            message += "✅ Verify transaction exists\n"
            message += "✅ Check correct address\n"
            message += "✅ Confirm amount ≥ $100\n"
            message += "✅ Ensure not used before\n\n"
            message += "Send your transaction hash now:"
            
            keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_payment")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error starting payment verification: {e}")
    
    async def show_user_status(self, query: CallbackQuery):
        """Show user status - For all users"""
        try:
            user_id = str(query.from_user.id)
            
            session = self.db_manager.get_session()
            db_user = session.query(User).filter(User.telegram_id == user_id).first()
            
            if db_user:
                await self.show_user_status_message(query.message, db_user)
            else:
                await query.edit_message_text("❌ User not found. Please use /start to register.")
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error showing user status: {e}")
    
    async def show_user_status_message(self, message, db_user):
        """Display user status message"""
        try:
            tier_info = {
                UserTier.FREE: ("🆓", "FREE"),
                UserTier.PAID: ("💎", "PAID"),
                UserTier.AWOOF: ("🎁", "AWOOF"),
                UserTier.ADMIN: ("👑", "ADMIN")
            }
            
            emoji, tier_name = tier_info.get(db_user.tier, ("", "Unknown"))
            
            message_text = f"{emoji} **Your Account Status**\n\n"
            message_text += f"**Username:** @{db_user.telegram_username}\n"
            message_text += f"**User ID:** `{db_user.telegram_id}`\n"
            message_text += f"**Tier:** {tier_name}\n"
            message_text += f"**Exchange:** {db_user.exchange.value if db_user.exchange else 'Not set'}\n"
            message_text += f"**API:** {'✅ Configured' if db_user.api_key else '❌ Not configured'}\n\n"
            
            if db_user.tier in [UserTier.PAID, UserTier.AWOOF]:
                if db_user.subscription_expires_at:
                    days_left = (db_user.subscription_expires_at - datetime.utcnow()).days
                    message_text += f"**Subscription:** {days_left} days left\n"
                    message_text += f"**Expires:** {db_user.subscription_expires_at.strftime('%Y-%m-%d')}\n\n"
            
            message_text += "**Trading Stats:**\n"
            message_text += f"• Daily Limit: {db_user.calculated_max_daily_trades}\n"
            message_text += f"• Total Trades: {db_user.total_trades}\n"
            message_text += f"• Successful: {db_user.successful_trades}\n"
            
            if db_user.total_trades > 0:
                success_rate = (db_user.successful_trades / db_user.total_trades * 100)
                message_text += f"• Success Rate: {success_rate:.1f}%\n"
            
            if db_user.total_pnl != 0:
                message_text += f"• Total P&L: ${db_user.total_pnl:,.2f}\n"
            
            message_text += f"\n**Account Created:** {db_user.created_at.strftime('%Y-%m-%d')}"
            
            keyboard = []
            
            if db_user.tier == UserTier.FREE:
                keyboard.append([InlineKeyboardButton("💳 Upgrade", callback_data="upgrade_account")])
            
            if not db_user.api_key:
                keyboard.append([InlineKeyboardButton("⚙️ Configure API", callback_data="configure_api")])
            
            keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="main_menu")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if hasattr(message, 'edit_text'):
                await message.edit_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            else:
                await message.reply_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
                
        except Exception as e:
            self.logger.error(f"Error showing user status: {e}")
    
    async def start_user_api_config(self, query: CallbackQuery):
        """Start user API configuration - For paid users only"""
        try:
            user_id = str(query.from_user.id)
            
            # Check if user is eligible
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == user_id).first()
            
            if not user:
                await query.edit_message_text("❌ User not found")
                session.close()
                return
            
            if user.tier == UserTier.FREE:
                await query.edit_message_text(
                    "❌ API configuration is only available for PAID users.\n\n"
                    "Please upgrade your account first.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("💳 Upgrade", callback_data="upgrade_account")
                    ]])
                )
                session.close()
                return
            
            session.close()
            
            # Start configuration
            self.pending_configurations[user_id] = {
                'step': 'exchange',
                'data': {},
                'type': 'user'
            }
            
            message = "⚙️ **API Configuration**\n\n"
            message += "Select your exchange:"
            
            keyboard = [
                [InlineKeyboardButton("Bybit", callback_data="select_exchange_bybit")],
                [InlineKeyboardButton("Binance", callback_data="select_exchange_binance")],
                [InlineKeyboardButton("KuCoin", callback_data="select_exchange_kucoin")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error starting user API config: {e}")
    
    async def select_exchange(self, query: CallbackQuery):
        """Handle exchange selection"""
        try:
            user_id = str(query.from_user.id)
            
            if user_id not in self.pending_configurations:
                await query.edit_message_text("❌ No configuration in progress")
                return
            
            exchange = query.data.replace("select_exchange_", "")
            self.pending_configurations[user_id]['data']['exchange'] = exchange
            self.pending_configurations[user_id]['step'] = 'api_key'
            
            message = f"✅ **{exchange.capitalize()} Selected**\n\n"
            message += "Now send your API Key:"
            
            keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error selecting exchange: {e}")
    
    async def show_trading_history(self, query: CallbackQuery):
        """Show user trading history - For paid users"""
        try:
            user_id = str(query.from_user.id)
            
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == user_id).first()
            
            if not user:
                await query.edit_message_text("❌ User not found")
                session.close()
                return
            
            message = "📈 **Your Trading History**\n\n"
            message += f"**Total Trades:** {user.total_trades}\n"
            message += f"**Successful:** {user.successful_trades}\n"
            
            if user.total_trades > 0:
                success_rate = (user.successful_trades / user.total_trades * 100)
                message += f"**Success Rate:** {success_rate:.1f}%\n"
            
            message += f"**Total P&L:** ${user.total_pnl:,.2f}\n\n"
            
            message += "Detailed trade history will be available soon."
            
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error showing trading history: {e}")
    
    # Helper Functions
    async def confirm_deposit_addresses(self, query: CallbackQuery):
        """Confirm and save deposit addresses - Admin only"""
        try:
            user_id = str(query.from_user.id)
            
            if user_id not in self.pending_configurations:
                await query.edit_message_text("❌ No pending configuration")
                return
            
            config = self.pending_configurations[user_id]
            
            session = self.db_manager.get_session()
            
            # Update system config
            system_config = session.query(SystemConfig).filter(
                SystemConfig.config_name == 'default'
            ).first()
            
            if not system_config:
                system_config = SystemConfig(config_name='default', subscription_fee=100.0)
                session.add(system_config)
            
            system_config.deposit_addresses = config['data']['addresses']
            
            # Update admin user
            admin = session.query(User).filter(
                User.telegram_id == self.ADMIN_ID
            ).first()
            
            if admin:
                admin.api_key = self.encryption_manager.encrypt_secret(config['data']['api_key'])
                admin.api_secret = self.encryption_manager.encrypt_secret(config['data']['api_secret'])
            
            session.commit()
            session.close()
            
            del self.pending_configurations[user_id]
            
            message = "🎉 **Configuration Complete!**\n\n"
            message += "✅ API credentials saved\n"
            message += "✅ Deposit addresses configured\n"
            message += "✅ System ready\n\n"
            
            if self.bootstrap_mode:
                message += "Bootstrap mode will now exit."
                self.bootstrap_mode = False
            
            keyboard = [
                [InlineKeyboardButton("👥 Manage Users", callback_data="manage_users")],
                [InlineKeyboardButton("📊 System Stats", callback_data="system_stats")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error confirming addresses: {e}")
    
    async def cancel_configuration(self, query: CallbackQuery):
        """Cancel current configuration"""
        try:
            user_id = str(query.from_user.id)
            
            if user_id in self.pending_configurations:
                del self.pending_configurations[user_id]
            
            await query.edit_message_text(
                "❌ Configuration cancelled",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]])
            )
            
        except Exception as e:
            self.logger.error(f"Error cancelling configuration: {e}")
    
    async def cancel_payment(self, query: CallbackQuery):
        """Cancel payment verification"""
        try:
            user_id = str(query.from_user.id)
            
            if user_id in self.pending_payments:
                del self.pending_payments[user_id]
            
            await query.edit_message_text(
                "❌ Payment verification cancelled",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]])
            )
            
        except Exception as e:
            self.logger.error(f"Error cancelling payment: {e}")
    
    async def cancel_broadcast(self, query: CallbackQuery):
        """Cancel broadcast message - Admin only"""
        try:
            user_id = str(query.from_user.id)
            
            if user_id in self.broadcast_states:
                del self.broadcast_states[user_id]
            
            await query.edit_message_text(
                "❌ Broadcast cancelled",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]])
            )
            
        except Exception as e:
            self.logger.error(f"Error cancelling broadcast: {e}")
    
    async def show_bootstrap_status(self, query: CallbackQuery):
        """Show bootstrap/system status"""
        try:
            session = self.db_manager.get_session()
            config = session.query(SystemConfig).filter(
                SystemConfig.config_name == 'default'
            ).first()
            
            message = "📊 **System Status**\n\n"
            
            if config:
                message += "✅ Database configured\n"
                
                if config.deposit_addresses and config.deposit_addresses.get('USDT_TRC20'):
                    message += "✅ Deposit addresses set\n"
                else:
                    message += "❌ Deposit addresses missing\n"
                
                message += f"✅ Subscription fee: ${config.subscription_fee}\n"
            else:
                message += "❌ System config missing\n"
            
            admin = session.query(User).filter(User.tier == UserTier.ADMIN).first()
            
            if admin:
                message += "✅ Admin user exists\n"
                if admin.api_key:
                    message += "✅ Admin API configured\n"
                else:
                    message += "❌ Admin API missing\n"
            else:
                message += "❌ Admin user missing\n"
            
            session.close()
            
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error showing status: {e}")
    
    async def show_help(self, query: CallbackQuery):
        """Show help information - For all users"""
        try:
            user_id = str(query.from_user.id)
            
            message = "❓ **Help & Information**\n\n"
            message += "**Smart Money Trader Bot**\n\n"
            
            message += "**User Tiers:**\n"
            message += "🆓 FREE - View signals only\n"
            message += "💎 PAID - 5 trades/day + API\n"
            message += "🎁 AWOOF - Admin-granted premium\n"
            
            if user_id == self.ADMIN_ID:
                message += "👑 ADMIN - Full control\n"
            
            message += "\n**How to Use:**\n"
            message += "1. Send /start to access menu\n"
            message += "2. Use buttons to navigate\n"
            message += "3. Upgrade for trading features\n\n"
            
            message += "**Support:** @SmartMoneyTraderAdmin"
            
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            self.logger.error(f"Error showing help: {e}")
    
    async def save_user_api_credentials(self, update, user_id: str, data: Dict):
        """Save user API credentials"""
        try:
            session = self.db_manager.get_session()
            user = session.query(User).filter(User.telegram_id == user_id).first()
            
            if user:
                # Update exchange if provided
                if 'exchange' in data:
                    try:
                        user.exchange = ExchangeType[data['exchange'].upper()]
                    except:
                        pass
                
                # Encrypt and save credentials
                user.api_key = self.encryption_manager.encrypt_secret(data['api_key'])
                user.api_secret = self.encryption_manager.encrypt_secret(data['api_secret'])
                session.commit()
                
                message = "✅ **API Configuration Saved!**\n\n"
                message += "Your API credentials have been encrypted and stored securely.\n\n"
                message += "You can now use the auto-trading features."
                
                keyboard = [
                    [InlineKeyboardButton("📊 View Status", callback_data="view_status")],
                    [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(message, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            
            session.close()
            
            if user_id in self.pending_configurations:
                del self.pending_configurations[user_id]
            
        except Exception as e:
            self.logger.error(f"Error saving API credentials: {e}")
            await update.message.reply_text("❌ Error saving credentials. Please try again.")
    
    async def back_to_main_menu(self, query: CallbackQuery):
        """Return to main menu - Shows different menu based on user"""
        try:
            user_id = str(query.from_user.id)
            
            # Simulate /start command behavior
            if user_id == self.ADMIN_ID:
                # Show admin menu
                await self.show_admin_panel(query.message)
            else:
                # Show user menu
                session = self.db_manager.get_session()
                db_user = session.query(User).filter(User.telegram_id == user_id).first()
                await self.show_user_menu(query.message, db_user)
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error returning to menu: {e}")
    
    async def show_admin_panel(self, message):
        """Show admin control panel - Only called for admin"""
        try:
            keyboard = [
                [InlineKeyboardButton("👥 Manage Users", callback_data="manage_users")],
                [InlineKeyboardButton("💰 View Payments", callback_data="view_payments")],
                [InlineKeyboardButton("📊 System Stats", callback_data="system_stats")],
                [InlineKeyboardButton("⚙️ Admin Settings", callback_data="admin_settings")],
                [InlineKeyboardButton("🔑 Update API", callback_data="config_admin_api")],
                [InlineKeyboardButton("📢 Broadcast", callback_data="broadcast_message")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message_text = "👑 **Admin Control Panel**\n\nSelect an option:"
            
            if hasattr(message, 'edit_text'):
                await message.edit_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            else:
                await message.reply_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
                
        except Exception as e:
            self.logger.error(f"Error showing admin panel: {e}")
    
    async def show_user_menu(self, message, db_user):
        """Show regular user menu - Only for non-admin users"""
        try:
            if not db_user:
                keyboard = [
                    [InlineKeyboardButton("❓ Help", callback_data="show_help")]
                ]
                message_text = "Welcome! Please use /start to register."
            else:
                keyboard = [
                    [InlineKeyboardButton("📊 My Status", callback_data="view_status")],
                    [InlineKeyboardButton("⚙️ Configure API", callback_data="configure_api")]
                ]
                
                if db_user.tier == UserTier.FREE:
                    keyboard.insert(0, [InlineKeyboardButton("💳 Upgrade", callback_data="upgrade_account")])
                else:
                    keyboard.append([InlineKeyboardButton("📈 Trading History", callback_data="trading_history")])
                
                keyboard.append([InlineKeyboardButton("❓ Help", callback_data="show_help")])
                
                message_text = f"Welcome back, {db_user.telegram_username}!\n\nSelect an option:"
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if hasattr(message, 'edit_text'):
                await message.edit_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
            else:
                await message.reply_text(message_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
                
        except Exception as e:
            self.logger.error(f"Error showing user menu: {e}")
    
    # [Continue with all remaining methods following the same pattern]
    # All methods remain exactly the same, just ensuring admin-only functions are protected
    
    async def stop_bot(self):
        """Stop the bot gracefully"""
        try:
            if self.application:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            self.logger.info("Bot stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")


# Export functions for backward compatibility
async def run_bootstrap_mode(config: EnhancedSystemConfig) -> bool:
    """Run bootstrap mode if needed"""
    try:
        bootstrap_manager = TelegramBootstrapManager(config)
        
        if not bootstrap_manager.should_enter_bootstrap_mode():
            print("✅ System already configured. No bootstrap needed.")
            return True
        
        print("🔄 Starting Bootstrap Mode...")
        print("📱 Check your Telegram for configuration instructions")
        
        success = await bootstrap_manager.start_bootstrap_mode()
        
        if success:
            while bootstrap_manager.bootstrap_mode:
                await asyncio.sleep(5)
            
            print("✅ Bootstrap mode completed successfully!")
            await bootstrap_manager.stop_bot()
            return True
        else:
            print("❌ Bootstrap mode failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Bootstrap mode error: {e}")
        return False


def check_bootstrap_needed(config: EnhancedSystemConfig) -> bool:
    """Check if bootstrap mode is needed"""
    bootstrap_manager = TelegramBootstrapManager(config)
    return bootstrap_manager.should_enter_bootstrap_mode()


async def send_trading_notification(
    config: EnhancedSystemConfig,
    message: str,
    keyboard: List[List[InlineKeyboardButton]] = None,
    image_path: str = None
):
    """Send trading notification to appropriate users"""
    try:
        bot = Bot(token=config.telegram_bot_token)
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        
        db_manager = DatabaseManager(config.db_config.get_database_url())
        users = db_manager.get_active_trading_users()
        
        for user in users:
            try:
                if image_path:
                    with open(image_path, "rb") as img_file:
                        await bot.send_photo(
                            chat_id=user.telegram_id,
                            photo=img_file,
                            caption=message,
                            reply_markup=reply_markup,
                            parse_mode=ParseMode.MARKDOWN
                        )
                else:
                    await bot.send_message(
                        chat_id=user.telegram_id,
                        text=message,
                        reply_markup=reply_markup,
                        parse_mode=ParseMode.MARKDOWN
                    )
            except Exception as e:
                logging.error(f"Failed to send to {user.telegram_id}: {e}")
        
    except Exception as e:
        logging.error(f"Failed to send notifications: {e}")