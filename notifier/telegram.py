"""
Telegram Bot for Enhanced Bybit Trading System.
Handles bootstrap mode, API key configuration, and trading notifications.
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from config.config import EnhancedSystemConfig, DatabaseConfig
from utils.encryption import SecretManager
from database.models import DatabaseManager


class TelegramBootstrapManager:
    """Handles Telegram bootstrap mode and API key configuration"""
    
    def __init__(self, config: EnhancedSystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.secret_manager = SecretManager.from_config(config)
        self.db_manager = DatabaseManager(config.db_config.get_database_url())
        
        # Bootstrap state tracking
        self.bootstrap_mode = False
        self.pending_configurations = {}  # user_id -> config_state
        
        # Initialize Telegram bot
        self.bot = None
        self.application = None
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize Telegram bot application"""
        try:
            if not self.config.telegram_bot_token:
                self.logger.error("❌ Telegram bot token not configured")
                return
            
            # Create bot application
            self.application = Application.builder().token(self.config.telegram_bot_token).build()
            self.bot = self.application.bot
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.handle_start))
            self.application.add_handler(CommandHandler("status", self.handle_status))
            self.application.add_handler(CommandHandler("bootstrap", self.handle_bootstrap))
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            self.logger.debug("✅ Telegram bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
    
    def check_api_credentials(self) -> Dict[str, bool]:
        """Check which API credentials are missing"""
        credentials_status = {
            'api_key': bool(self.config.api_key),
            'api_secret': bool(self.config.api_secret),
            'demo_api_key': bool(self.config.demo_api_key),
            'demo_api_secret': bool(self.config.demo_api_secret)
        }
        
        all_configured = all(credentials_status.values())
        return {
            'all_configured': all_configured,
            'missing_credentials': [key for key, configured in credentials_status.items() if not configured],
            'status': credentials_status
        }
    
    def should_enter_bootstrap_mode(self) -> bool:
        """Determine if system should enter bootstrap mode"""
        cred_status = self.check_api_credentials()
        return not cred_status['all_configured']
    
    async def start_bootstrap_mode(self) -> bool:
        """Start bootstrap mode and notify via Telegram"""
        try:
            self.bootstrap_mode = True
            self.logger.info("🔄 Entering Bootstrap Mode - API credentials needed")
            
            # Send bootstrap notification
            await self.send_bootstrap_notification()
            
            # Start bot polling
            await self.start_bot()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start bootstrap mode: {e}")
            return False
    
    async def send_bootstrap_notification(self):
        """Send bootstrap mode notification to Telegram"""
        try:
            user_id = self.config.telegram_id
            
            cred_status = self.check_api_credentials()
            missing_creds = cred_status['missing_credentials']
            
            message = "🤖 **BYBIT AUTO-TRADING SYSTEM**\n"
            message += "🔄 **BOOTSTRAP MODE ACTIVATED**\n\n"
            message += "⚠️ **API Credentials Required**\n"
            message += f"Missing credentials: {', '.join(missing_creds)}\n\n"
            message += "📝 **Configuration Status:**\n"
            
            for cred, status in cred_status['status'].items():
                status_emoji = "✅" if status else "❌"
                message += f"   {status_emoji} {cred.replace('_', ' ').title()}\n"
            
            message += "\n🔧 **Next Steps:**\n"
            message += "1. Click 'Configure API Keys' below\n"
            message += "2. Follow the prompts to enter your credentials\n"
            message += "3. System will automatically start trading once configured\n\n"
            message += "🔒 **Security:** All keys are encrypted before storage"
            
            # Create inline keyboard
            keyboard = [
                [InlineKeyboardButton("🔧 Configure API Keys", callback_data="start_config")],
                [InlineKeyboardButton("📊 Check Status", callback_data="check_status")],
                [InlineKeyboardButton("❓ Help", callback_data="show_help")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.bot.send_message(
                chat_id=user_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
            self.logger.info(f"📱 Bootstrap notification sent to Telegram user: {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send bootstrap notification: {e}")
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        try:
            user_id = str(update.effective_user.id)
            
            if user_id != self.config.telegram_id:
                await update.message.reply_text("❌ Unauthorized user")
                return
            
            message = "🤖 **Bybit Auto-Trading System**\n\n"
            
            if self.bootstrap_mode:
                message += "🔄 **Bootstrap Mode Active**\n"
                message += "API credentials need to be configured.\n\n"
                
                keyboard = [
                    [InlineKeyboardButton("🔧 Configure API Keys", callback_data="start_config")],
                    [InlineKeyboardButton("📊 Check Status", callback_data="check_status")]
                ]
            else:
                message += "✅ **System Operational**\n"
                message += "Auto-trading is active and running.\n\n"
                
                keyboard = [
                    [InlineKeyboardButton("📊 System Status", callback_data="check_status")],
                    [InlineKeyboardButton("📈 Trading Stats", callback_data="trading_stats")],
                    [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
                ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error in handle_start: {e}")
    
    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            user_id = str(update.effective_user.id)
            
            if user_id != self.config.telegram_id:
                await update.message.reply_text("❌ Unauthorized user")
                return
            
            cred_status = self.check_api_credentials()
            
            message = "📊 **System Status Report**\n\n"
            message += f"🔄 Bootstrap Mode: {'Active' if self.bootstrap_mode else 'Inactive'}\n"
            message += f"⚙️ API Config: {'Complete' if cred_status['all_configured'] else 'Incomplete'}\n\n"
            
            message += "🔑 **API Credentials:**\n"
            for cred, status in cred_status['status'].items():
                status_emoji = "✅" if status else "❌"
                message += f"   {status_emoji} {cred.replace('_', ' ').title()}\n"
            
            if not cred_status['all_configured']:
                message += f"\n⚠️ Missing: {', '.join(cred_status['missing_credentials'])}"
            
            message += f"\n🕒 Status checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in handle_status: {e}")
    
    async def handle_bootstrap(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /bootstrap command"""
        try:
            user_id = str(update.effective_user.id)
            
            if user_id != self.config.telegram_id:
                await update.message.reply_text("❌ Unauthorized user")
                return
            
            if self.should_enter_bootstrap_mode():
                await self.start_bootstrap_mode()
                await update.message.reply_text("🔄 Bootstrap mode activated!")
            else:
                await update.message.reply_text("✅ All API credentials are configured. No bootstrap needed.")
            
        except Exception as e:
            self.logger.error(f"Error in handle_bootstrap: {e}")
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        try:
            query = update.callback_query
            await query.answer()
            
            user_id = str(query.from_user.id)
            
            if user_id != self.config.telegram_id:
                await query.edit_message_text("❌ Unauthorized user")
                return
            
            callback_data = query.data
            
            if callback_data == "start_config":
                await self.start_api_configuration(query)
            elif callback_data == "check_status":
                await self.show_status(query)
            elif callback_data == "show_help":
                await self.show_help(query)
            elif callback_data.startswith("config_"):
                await self.handle_config_step(query, callback_data)
            elif callback_data == "cancel_config":
                await self.cancel_configuration(query)
            elif callback_data == "confirm_config":
                await self.confirm_configuration(query)
            else:
                await query.edit_message_text(f"Unknown action: {callback_data}")
            
        except Exception as e:
            self.logger.error(f"Error in handle_callback: {e}")
    
    async def start_api_configuration(self, query):
        """Start API key configuration process"""
        try:
            user_id = str(query.from_user.id)
            
            # Initialize configuration state
            self.pending_configurations[user_id] = {
                'step': 'api_key',
                'data': {},
                'started_at': datetime.now()
            }
            
            message = "🔧 **API Key Configuration**\n\n"
            message += "📝 **Step 1/4: Production API Key**\n"
            message += "Please enter your Bybit **API Key** for production trading.\n\n"
            message += "⚠️ **Important:**\n"
            message += "• Use API keys with trading permissions\n"
            message += "• Keys will be encrypted before storage\n"
            message += "• Never share your API keys with anyone\n\n"
            message += "💬 **Send your API Key as a message below:**"
            
            keyboard = [
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error starting API configuration: {e}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages (API key inputs)"""
        try:
            user_id = str(update.effective_user.id)
            
            if user_id != self.config.telegram_id:
                await update.message.reply_text("❌ Unauthorized user")
                return
            
            # Check if user is in configuration process
            if user_id not in self.pending_configurations:
                await update.message.reply_text(
                    "ℹ️ No active configuration. Use /start to begin."
                )
                return
            
            config_state = self.pending_configurations[user_id]
            user_input = update.message.text.strip()
            
            # Delete the user's message for security
            try:
                await update.message.delete()
            except:
                pass
            
            await self.process_config_input(user_id, config_state, user_input)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    async def process_config_input(self, user_id: str, config_state: Dict, user_input: str):
        """Process configuration input for current step"""
        try:
            step = config_state['step']
            
            # Store the input
            config_state['data'][step] = user_input
            
            # Determine next step
            step_sequence = ['api_key', 'api_secret', 'demo_api_key', 'demo_api_secret']
            current_index = step_sequence.index(step)
            
            if current_index < len(step_sequence) - 1:
                # Move to next step
                next_step = step_sequence[current_index + 1]
                config_state['step'] = next_step
                await self.show_next_config_step(user_id, next_step)
            else:
                # Configuration complete, show confirmation
                await self.show_configuration_confirmation(user_id)
            
        except Exception as e:
            self.logger.error(f"Error processing config input: {e}")
    
    async def show_next_config_step(self, user_id: str, step: str):
        """Show the next configuration step"""
        try:
            step_info = {
                'api_key': {
                    'title': 'Production API Key',
                    'number': '1/4',
                    'description': 'Enter your Bybit API Key for production trading'
                },
                'api_secret': {
                    'title': 'Production API Secret',
                    'number': '2/4',
                    'description': 'Enter your Bybit API Secret for production trading'
                },
                'demo_api_key': {
                    'title': 'Demo API Key',
                    'number': '3/4',
                    'description': 'Enter your Bybit Demo API Key for testing'
                },
                'demo_api_secret': {
                    'title': 'Demo API Secret',
                    'number': '4/4',
                    'description': 'Enter your Bybit Demo API Secret for testing'
                }
            }
            
            info = step_info[step]
            
            message = "🔧 **API Key Configuration**\n\n"
            message += f"📝 **Step {info['number']}: {info['title']}**\n"
            message += f"{info['description']}\n\n"
            message += "💬 **Send your key as a message below:**"
            
            keyboard = [
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send new message instead of editing (for security)
            await self.bot.send_message(
                chat_id=user_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error showing next config step: {e}")
    
    async def show_configuration_confirmation(self, user_id: str):
        """Show configuration confirmation"""
        try:
            config_state = self.pending_configurations[user_id]
            
            message = "✅ **Configuration Complete**\n\n"
            message += "🔑 **API Keys Collected:**\n"
            message += "   ✅ Production API Key\n"
            message += "   ✅ Production API Secret\n"
            message += "   ✅ Demo API Key\n"
            message += "   ✅ Demo API Secret\n\n"
            message += "🔒 **Security:** All keys will be encrypted before storage.\n\n"
            message += "⚠️ **Confirm to save and start trading system?**"
            
            keyboard = [
                [InlineKeyboardButton("✅ Confirm & Save", callback_data="confirm_config")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel_config")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await self.bot.send_message(
                chat_id=user_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error showing configuration confirmation: {e}")
    
    async def confirm_configuration(self, query):
        """Confirm and save API configuration"""
        try:
            user_id = str(query.from_user.id)
            
            if user_id not in self.pending_configurations:
                await query.edit_message_text("❌ No pending configuration found")
                return
            
            config_data = self.pending_configurations[user_id]['data']
            
            # Encrypt and save API keys
            success = await self.save_encrypted_api_keys(config_data)
            
            if success:
                # Clear pending configuration
                del self.pending_configurations[user_id]
                
                # Exit bootstrap mode
                self.bootstrap_mode = False
                
                message = "🎉 **Configuration Saved Successfully!**\n\n"
                message += "✅ All API keys encrypted and stored\n"
                message += "🔄 Bootstrap mode deactivated\n"
                message += "🚀 Auto-trading system ready to start\n\n"
                message += "📊 **Next Steps:**\n"
                message += "• System will now validate API connections\n"
                message += "• Auto-trading will begin on next scheduled scan\n"
                message += "• You'll receive notifications for all trades\n\n"
                message += f"🕒 Configured at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                await query.edit_message_text(message, parse_mode='Markdown')
                
                self.logger.info("✅ API configuration completed and saved")
                
                # Signal that bootstrap is complete
                return True
                
            else:
                message = "❌ **Configuration Failed**\n\n"
                message += "Failed to save API keys. Please try again.\n"
                message += "Check logs for error details."
                
                await query.edit_message_text(message, parse_mode='Markdown')
                
                return False
            
        except Exception as e:
            self.logger.error(f"Error confirming configuration: {e}")
            await query.edit_message_text(f"❌ Error saving configuration: {e}")
            return False
    
    async def save_encrypted_api_keys(self, config_data: Dict) -> bool:
        """Encrypt and save API keys to database"""
        try:
            # Encrypt all API keys
            encrypted_data = {}
            
            for key, value in config_data.items():
                if value:  # Only encrypt non-empty values
                    encrypted_value = self.secret_manager.encrypt_secret(value)
                    encrypted_data[key] = encrypted_value
                    self.logger.debug(f"Encrypted {key} (length: {len(encrypted_value)})")
            
            # Update configuration in database
            success = self.config.update_config(**encrypted_data)
            
            if success:
                self.logger.info("✅ API keys encrypted and saved to database")
                return True
            else:
                self.logger.error("❌ Failed to save encrypted API keys to database")
                return False
            
        except Exception as e:
            self.logger.error(f"Error saving encrypted API keys: {e}")
            return False
    
    async def cancel_configuration(self, query):
        """Cancel configuration process"""
        try:
            user_id = str(query.from_user.id)
            
            if user_id in self.pending_configurations:
                del self.pending_configurations[user_id]
            
            message = "❌ **Configuration Cancelled**\n\n"
            message += "🔄 Bootstrap mode remains active\n"
            message += "💡 Use /start to begin configuration again"
            
            await query.edit_message_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error cancelling configuration: {e}")
    
    async def show_status(self, query):
        """Show system status"""
        try:
            cred_status = self.check_api_credentials()
            
            message = "📊 **System Status**\n\n"
            message += f"🔄 Bootstrap: {'Active' if self.bootstrap_mode else 'Inactive'}\n"
            message += f"⚙️ API Config: {'Complete' if cred_status['all_configured'] else 'Incomplete'}\n\n"
            
            message += "🔑 **Credentials Status:**\n"
            for cred, status in cred_status['status'].items():
                status_emoji = "✅" if status else "❌"
                message += f"   {status_emoji} {cred.replace('_', ' ').title()}\n"
            
            if not cred_status['all_configured']:
                message += f"\n⚠️ Missing: {', '.join(cred_status['missing_credentials'])}"
                
                keyboard = [
                    [InlineKeyboardButton("🔧 Configure Now", callback_data="start_config")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
            else:
                keyboard = [
                    [InlineKeyboardButton("🔄 Refresh", callback_data="check_status")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error showing status: {e}")
    
    async def show_help(self, query):
        """Show help information"""
        try:
            message = "❓ **Help & Information**\n\n"
            message += "🤖 **Bootstrap Mode:**\n"
            message += "• Activated when API keys are missing\n"
            message += "• Guides you through secure key configuration\n"
            message += "• Automatically exits when all keys are set\n\n"
            
            message += "🔑 **API Keys Required:**\n"
            message += "• Production API Key & Secret (for live trading)\n"
            message += "• Demo API Key & Secret (for testing)\n\n"
            
            message += "🔒 **Security:**\n"
            message += "• All keys are encrypted before storage\n"
            message += "• Keys are decrypted only when needed\n"
            message += "• Messages with keys are auto-deleted\n\n"
            
            message += "📱 **Commands:**\n"
            message += "• /start - Main menu\n"
            message += "• /status - System status\n"
            message += "• /bootstrap - Force bootstrap mode\n\n"
            
            message += "🆘 **Support:**\n"
            message += "Check system logs for detailed error information"
            
            keyboard = [
                [InlineKeyboardButton("🔙 Back", callback_data="check_status")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error showing help: {e}")
    
    async def start_bot(self):
        """Start the Telegram bot polling"""
        try:
            if self.application:
                self.logger.info("🤖 Starting Telegram bot polling...")
                await self.application.initialize()
                await self.application.start()
                await self.application.updater.start_polling()
                self.logger.info("✅ Telegram bot is now running")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to start bot polling: {e}")
            return False
    
    async def stop_bot(self):
        """Stop the Telegram bot"""
        try:
            if self.application:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                self.logger.info("🛑 Telegram bot stopped")
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    def is_bootstrap_complete(self) -> bool:
        """Check if bootstrap process is complete"""
        return not self.bootstrap_mode and self.check_api_credentials()['all_configured']


# Convenience functions for integration with main system
async def run_bootstrap_mode(config: EnhancedSystemConfig) -> bool:
    """Run bootstrap mode until API keys are configured"""
    try:
        bootstrap_manager = TelegramBootstrapManager(config)
        
        if not bootstrap_manager.should_enter_bootstrap_mode():
            print("✅ All API credentials are configured. No bootstrap needed.")
            return True
        
        print("🔄 Starting Bootstrap Mode...")
        print("📱 Check your Telegram for configuration instructions")
        
        success = await bootstrap_manager.start_bootstrap_mode()
        
        if success:
            # Wait for configuration to complete
            while bootstrap_manager.bootstrap_mode:
                await asyncio.sleep(5)  # Check every 5 seconds
            
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


# async def send_trading_notification(config: EnhancedSystemConfig, message: str, keyboard: List[List[InlineKeyboardButton]] = None):
#     """Send trading notification to Telegram"""
#     try:
#         bot = Bot(token=config.telegram_bot_token)
        
#         reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        
#         await bot.send_message(
#             chat_id=config.telegram_id,
#             text=message,
#             reply_markup=reply_markup,
#             parse_mode='Markdown'
#         )
        
#     except Exception as e:
#         logging.error(f"Failed to send Telegram notification: {e}")


from telegram.constants import ParseMode

async def send_trading_notification(
    config: EnhancedSystemConfig,
    message: str,
    keyboard: List[List[InlineKeyboardButton]] = None,
    image_path: str = None
):
    """Send trading notification to Telegram, with optional image attachment"""
    try:
        bot = Bot(token=config.telegram_bot_token)
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

        if image_path:
            # Send as photo with caption
            with open(image_path, "rb") as img_file:
                await bot.send_photo(
                    chat_id=config.telegram_id,
                    photo=img_file,
                    caption=message,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.MARKDOWN
                )
        else:
            # Send as text message
            await bot.send_message(
                chat_id=config.telegram_id,
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

    except Exception as e:
        logging.error(f"Failed to send Telegram notification: {e}")


# Main execution for standalone testing
if __name__ == "__main__":
    async def test_bootstrap():
        """Test bootstrap functionality"""
        try:
            # Load configuration
            from config.config import DatabaseConfig, EnhancedSystemConfig
            
            db_config = DatabaseConfig()
            config = EnhancedSystemConfig.from_database(db_config, 'default')
            
            # Run bootstrap
            success = await run_bootstrap_mode(config)
            print(f"Bootstrap result: {'Success' if success else 'Failed'}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    asyncio.run(test_bootstrap())