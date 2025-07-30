#!/usr/bin/env python3
"""
Bootstrap Setup Script for Enhanced Bybit Trading System.
Prepares system for Telegram bootstrap mode and validates configuration.
COMPLETE: All functionality from project knowledge included.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('ccxt', 'Cryptocurrency exchange library'),
        ('pandas', 'Data manipulation and analysis'),
        ('numpy', 'Numerical computing'),
        ('ta', 'Technical analysis indicators'),
        ('sqlalchemy', 'SQL toolkit and ORM'),
        ('pymysql', 'Pure Python MySQL client'),
        ('telegram', 'Telegram Bot API wrapper'),
        ('cryptography', 'Cryptographic recipes'),
        ('plotly', 'Interactive plotting library'),
        ('yaml', 'YAML parser and emitter'),
        ('psutil', 'System and process utilities')
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package, description in required_packages:
        try:
            if package == 'telegram':
                # Special check for python-telegram-bot
                import telegram
                # Check if it's the right telegram package
                if hasattr(telegram, 'Bot'):
                    installed_packages.append((package, description))
                else:
                    missing_packages.append((package, description, 'python-telegram-bot'))
            elif package == 'yaml':
                import yaml
                installed_packages.append((package, description))
            else:
                __import__(package)
                installed_packages.append((package, description))
        except ImportError:
            if package == 'telegram':
                missing_packages.append((package, description, 'python-telegram-bot'))
            else:
                missing_packages.append((package, description, package))
    
    print(f"✅ Installed packages: {len(installed_packages)}")
    for pkg, desc in installed_packages:
        print(f"   ✓ {pkg} - {desc}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {len(missing_packages)}")
        for pkg, desc, install_name in missing_packages:
            print(f"   ✗ {pkg} - {desc}")
        
        print("\n📦 Install missing packages with:")
        print("   pip install " + " ".join([pkg[2] for pkg in missing_packages]))
        print("\nOr install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed")
    return True


def validate_telegram_config():
    """Validate Telegram configuration"""
    try:
        from config.config import DatabaseConfig, EnhancedSystemConfig
        
        # Load database config from YAML, then system config from database
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        print(f"📄 Loading configuration from {config_path}...")
        db_config = DatabaseConfig.from_yaml_file(config_path)
        
        print(f"📊 Database config loaded:")
        print(f"   Host: {db_config.host}:{db_config.port}")
        print(f"   Database: {db_config.database}")
        print(f"   Username: {db_config.username}")
        print(f"   Password: {'***' if db_config.password else 'NOT SET'}")
        
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        if not config.telegram_bot_token:
            print("❌ Telegram bot token not configured")
            print("   Please set telegram_bot_token in database configuration")
            return False
        
        if not config.telegram_id:
            print("❌ Telegram user ID not configured") 
            print("   Please set telegram_id in database configuration")
            return False
        
        print("✅ Telegram configuration validated")
        print(f"   Bot Token: {'***' + config.telegram_bot_token[-10:] if len(config.telegram_bot_token) > 10 else 'Present'}")
        print(f"   User ID: {config.telegram_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating Telegram config: {e}")
        return False


def test_telegram_connection():
    """Test Telegram bot connection"""
    try:
        from config.config import DatabaseConfig, EnhancedSystemConfig
        
        # Load config from YAML and database
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        db_config = DatabaseConfig.from_yaml_file(config_path)
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        async def test_bot():
            try:
                # Import here to avoid conflicts
                from telegram import Bot
                
                bot = Bot(token=config.telegram_bot_token)
                bot_info = await bot.get_me()
                print(f"✅ Telegram bot connected: @{bot_info.username}")
                print(f"   Bot ID: {bot_info.id}")
                print(f"   Bot Name: {bot_info.first_name}")
                
                # Test sending message to configured user
                test_message = (
                    "🧪 **Bootstrap Setup Test**\n\n"
                    "✅ Telegram connection successful!\n"
                    "🤖 The auto-trading system is ready for bootstrap mode.\n\n"
                    "🔧 **Next Steps:**\n"
                    "1. Run the main auto-trading script\n"
                    "2. Follow bootstrap instructions for API setup\n"
                    "3. Start automated trading\n\n"
                    f"🕒 Test completed: {os.popen('date').read().strip()}"
                )
                
                await bot.send_message(
                    chat_id=config.telegram_id,
                    text=test_message,
                    parse_mode='Markdown'
                )
                print(f"✅ Test message sent to user ID: {config.telegram_id}")
                return True
                
            except Exception as e:
                print(f"❌ Telegram bot test failed: {e}")
                if "Unauthorized" in str(e):
                    print("   Check your bot token")
                elif "Bad Request" in str(e):
                    print("   Check your user ID")
                elif "Forbidden" in str(e):
                    print("   Make sure you've started a conversation with your bot")
                return False
        
        # Run async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(test_bot())
        finally:
            loop.close()
            
    except Exception as e:
        print(f"❌ Error testing Telegram connection: {e}")
        return False


def setup_telegram_bot_instructions():
    """Display instructions for setting up Telegram bot"""
    print("\n🤖 TELEGRAM BOT SETUP INSTRUCTIONS")
    print("=" * 50)
    print("If you haven't set up your Telegram bot yet, follow these steps:")
    print("")
    print("1️⃣ **Create Telegram Bot:**")
    print("   • Open Telegram and search for @BotFather")
    print("   • Send /newbot command")
    print("   • Choose a name and username for your bot")
    print("   • Copy the bot token (format: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)")
    print("")
    print("2️⃣ **Get Your User ID:**")
    print("   • Search for @userinfobot in Telegram")
    print("   • Send /start to get your user ID")
    print("   • Copy your numeric user ID (e.g., 123456789)")
    print("")
    print("3️⃣ **Start Conversation with Your Bot:**")
    print("   • Search for your bot username in Telegram")
    print("   • Send /start to begin conversation")
    print("   • This is required for the bot to message you")
    print("")
    print("4️⃣ **Update Database Configuration:**")
    print("   • Update telegram_bot_token in database")
    print("   • Update telegram_id in database")
    print("   • Or set them via SQL commands:")
    print("")
    print("   UPDATE system_config SET")
    print("   telegram_bot_token = 'YOUR_BOT_TOKEN',")
    print("   telegram_id = 'YOUR_USER_ID'")
    print("   WHERE config_name = 'default';")
    print("")
    print("5️⃣ **Test Configuration:**")
    print("   • Run this script again to test connection")
    print("   • Start the auto-trading system")
    print("")


def validate_database_connection():
    """Validate database connection and tables"""
    try:
        from config.config import DatabaseConfig
        from database.models import DatabaseManager
        
        # Load database config from YAML file
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            print("   Run the main script first to create default configuration")
            return False
            
        db_config = DatabaseConfig.from_yaml_file(config_path)
        print(f"📊 Database config loaded from YAML:")
        print(f"   Host: {db_config.host}:{db_config.port}")
        print(f"   Database: {db_config.database}")
        print(f"   Username: {db_config.username}")
        print(f"   Password: {'***' if db_config.password else 'NOT SET'}")
        
        if not db_config.password:
            print("❌ Database password not set in enhanced_config.yaml")
            print("   Please update the 'password' field with your MySQL root password")
            return False
        
        db_manager = DatabaseManager(db_config.get_database_url())
        
        if not db_manager.test_connection():
            print("❌ Database connection failed")
            print(f"   Host: {db_config.host}:{db_config.port}")
            print(f"   Database: {db_config.database}")
            print(f"   Username: {db_config.username}")
            print("")
            print("🔧 Troubleshooting:")
            print("   • Ensure MySQL is running")
            print("   • Check username and password")
            print("   • Verify database exists")
            print("   • Check firewall settings")
            return False
        
        print("✅ Database connection successful")
        
        # Test table creation
        try:
            db_manager.create_tables()
            print("✅ Database tables verified/created")
            
            # Test basic operations
            session = db_manager.get_session()
            
            # Test system config table
            from database.models import SystemConfig
            config_count = session.query(SystemConfig).count()
            print(f"✅ System configurations found: {config_count}")
            
            session.close()
            
        except Exception as e:
            print(f"❌ Error with database tables: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database validation error: {e}")
        return False


def check_api_credentials_status():
    """Check current API credentials status"""
    try:
        from config.config import DatabaseConfig, EnhancedSystemConfig
        from telegram_bot_and_notification.bootstrap_manager import TelegramBootstrapManager
        
        # Load config from YAML and database
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        db_config = DatabaseConfig.from_yaml_file(config_path)
        config = EnhancedSystemConfig.from_database(db_config, 'default')
        
        bootstrap_manager = TelegramBootstrapManager(config)
        cred_status = bootstrap_manager.check_api_credentials()
        
        print("\n🔑 API CREDENTIALS STATUS:")
        print("=" * 30)
        
        for cred, status in cred_status['status'].items():
            status_emoji = "✅" if status else "❌"
            cred_display = cred.replace('_', ' ').title()
            print(f"   {status_emoji} {cred_display}")
        
        if cred_status['all_configured']:
            print("\n✅ All API credentials are configured")
            print("🚀 System ready for auto-trading")
            return True
        else:
            print(f"\n⚠️  Missing: {', '.join(cred_status['missing_credentials'])}")
            print("🔄 Bootstrap mode will be required")
            print("")
            print("💡 The missing credentials will be configured via Telegram bootstrap mode:")
            print("   1. Start the main auto-trading script")
            print("   2. Follow Telegram instructions to enter API keys")
            print("   3. Keys will be encrypted and stored securely")
            return False
            
    except Exception as e:
        print(f"❌ Error checking API credentials: {e}")
        return False


def create_systemd_service():
    """Create systemd service file for auto-trading"""
    try:
        service_content = f"""[Unit]
Description=Enhanced Bybit Auto-Trading System
After=network.target mysql.service
Wants=mysql.service

[Service]
Type=simple
User={os.getenv('USER', 'trading')}
WorkingDirectory={project_root}
ExecStart={sys.executable} main.py
Restart=always
RestartSec=30
Environment=PYTHONPATH={project_root}
Environment=PYTHONUNBUFFERED=1

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bybit-autotrader

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={project_root}

# Resource limits
LimitNOFILE=65536
MemoryAccounting=true
MemoryMax=2G

[Install]
WantedBy=multi-user.target
"""
        
        service_file = project_root / "bybit-autotrader.service"
        
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"✅ Systemd service file created: {service_file}")
        print("\n📋 To install as system service:")
        print(f"   sudo cp {service_file} /etc/systemd/system/")
        print("   sudo systemctl daemon-reload")
        print("   sudo systemctl enable bybit-autotrader")
        print("   sudo systemctl start bybit-autotrader")
        print("")
        print("📊 Monitor with:")
        print("   sudo systemctl status bybit-autotrader")
        print("   sudo journalctl -u bybit-autotrader -f")
        print("   sudo journalctl -u bybit-autotrader --since '1 hour ago'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating systemd service: {e}")
        return False


def check_system_requirements():
    """Check system requirements and configuration"""
    try:
        import platform
        import psutil
        
        print("\n💻 SYSTEM REQUIREMENTS CHECK:")
        print("=" * 35)
        
        # Operating system
        os_info = platform.system()
        os_version = platform.release()
        print(f"   OS: {os_info} {os_version}")
        
        # Python version
        python_version = platform.python_version()
        print(f"   Python: {python_version}")
        
        if python_version < "3.8":
            print("   ⚠️  Python 3.8+ recommended")
        else:
            print("   ✅ Python version OK")
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"   RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 2:
            print("   ⚠️  2GB+ RAM recommended")
        else:
            print("   ✅ Memory OK")
        
        # Disk space
        disk = psutil.disk_usage(project_root)
        disk_free_gb = disk.free / (1024**3)
        print(f"   Free Disk: {disk_free_gb:.1f} GB")
        
        if disk_free_gb < 1:
            print("   ⚠️  1GB+ free space recommended")
        else:
            print("   ✅ Disk space OK")
        
        # CPU cores
        cpu_cores = psutil.cpu_count()
        print(f"   CPU Cores: {cpu_cores}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking system requirements: {e}")
        return False


def validate_project_structure():
    """Validate project directory structure"""
    try:
        print("\n📁 PROJECT STRUCTURE VALIDATION:")
        print("=" * 35)
        
        required_dirs = [
            'config',
            'core', 
            'analysis',
            'signals',
            'visualization',
            'database',
            'utils',
            'notifier',
            'bybit'
        ]
        
        required_files = [
            'main.py',
            'enhanced_config.yaml',
            'requirements.txt',
            'config/config.py',
            'notifier/telegram.py',
            'bybit/autotrader.py',
            'database/models.py'
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"   ✅ {dir_name}/")
            else:
                print(f"   ❌ {dir_name}/")
                missing_dirs.append(dir_name)
        
        # Check files
        for file_name in required_files:
            file_path = project_root / file_name
            if file_path.exists() and file_path.is_file():
                print(f"   ✅ {file_name}")
            else:
                print(f"   ❌ {file_name}")
                missing_files.append(file_name)
        
        if missing_dirs or missing_files:
            print(f"\n⚠️  Missing components:")
            for item in missing_dirs + missing_files:
                print(f"   • {item}")
            return False
        
        print("\n✅ Project structure complete")
        return True
        
    except Exception as e:
        print(f"❌ Error validating project structure: {e}")
        return False


def create_example_config():
    """Create example configuration files"""
    try:
        print("\n📄 CREATING EXAMPLE CONFIGURATION:")
        print("=" * 35)
        
        # Create enhanced_config.yaml if it doesn't exist
        config_path = project_root / 'enhanced_config.yaml'
        if not config_path.exists():
            config_content = """# Enhanced Bybit Trading System - Database Configuration
# All trading system configurations are now stored in the MySQL database.
# This file only contains database connection settings.

database:
  # Database type: mysql only
  type: "mysql"
  
  # MySQL settings - UPDATE THESE VALUES
  host: "localhost"
  port: 3306
  database: "bybit_trading_system"
  username: "root"
  password: "YOUR_MYSQL_ROOT_PASSWORD_HERE"  # ← UPDATE THIS
  
  # Connection pool settings
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
  echo_sql: false

# Configuration Management
config:
  # Which configuration to load from database
  config_name: "default"
  
  # Auto-create default config if not found
  auto_create_default: true
  
  # Backup configuration to YAML on changes
  backup_on_update: true
  backup_directory: "config_backups"

# System Settings
system:
  # Enable database logging
  enable_db_logging: true
  
  # Automatic cleanup settings
  auto_cleanup_enabled: true
  cleanup_days_to_keep: 30
  cleanup_schedule: "daily"  # daily, weekly, monthly
  
  # Database maintenance
  auto_vacuum: true
  backup_on_startup: false

# Auto-Trading Settings
auto_trading:
  enabled: true
  bootstrap_mode: true
  note: "Auto-trading parameters are stored in database, not this file"
"""
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            print(f"   ✅ Created {config_path}")
            print("   ⚠️  Please update the MySQL password!")
        else:
            print(f"   ✅ {config_path} already exists")
        
        # Create .env example file
        env_example_path = project_root / '.env.example'
        if not env_example_path.exists():
            env_content = """# Enhanced Bybit Trading System - Environment Variables
# Copy this file to .env and update the values

# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_NAME=bybit_trading_system
DB_USER=root
DB_PASSWORD=your_mysql_password_here

# Encryption Key for API secrets
SECRET_PASSWORD=your_secure_encryption_password_here

# Optional: Override default settings
# LOG_LEVEL=INFO
# PYTHONUNBUFFERED=1
"""
            
            with open(env_example_path, 'w') as f:
                f.write(env_content)
            
            print(f"   ✅ Created {env_example_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating example configuration: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive system test"""
    try:
        print("\n🧪 COMPREHENSIVE SYSTEM TEST:")
        print("=" * 32)
        
        # Test 1: Import all modules
        print("1️⃣ Testing module imports...")
        try:
            from config.config import DatabaseConfig, EnhancedSystemConfig
            from database.models import DatabaseManager
            from telegram_bot_and_notification.bootstrap_manager import TelegramBootstrapManager
            from bybit.autotrader import AutoTrader
            print("   ✅ All modules import successfully")
        except Exception as e:
            print(f"   ❌ Module import failed: {e}")
            return False
        
        # Test 2: Configuration loading
        print("2️⃣ Testing configuration loading...")
        try:
            config_path = 'enhanced_config.yaml'
            if os.path.exists(config_path):
                db_config = DatabaseConfig.from_yaml_file(config_path)
                print(f"   ✅ Database config loaded from YAML")
                print(f"       Host: {db_config.host}:{db_config.port}")
                print(f"       Database: {db_config.database}")
            else:
                print("   ⚠️  Configuration file not found")
        except Exception as e:
            print(f"   ❌ Configuration loading failed: {e}")
        
        # Test 3: Database connection (if password is set)
        print("3️⃣ Testing database connection...")
        try:
            if 'db_config' in locals() and db_config.password and db_config.password != "YOUR_MYSQL_ROOT_PASSWORD_HERE":
                db_manager = DatabaseManager(db_config.get_database_url())
                if db_manager.test_connection():
                    print("   ✅ Database connection successful")
                else:
                    print("   ❌ Database connection failed")
            else:
                print("   ⚠️  Database password not configured, skipping test")
        except Exception as e:
            print(f"   ❌ Database test failed: {e}")
        
        # Test 4: Telegram configuration
        print("4️⃣ Testing Telegram configuration...")
        try:
            if 'db_config' in locals() and db_config.password and db_config.password != "YOUR_MYSQL_ROOT_PASSWORD_HERE":
                config = EnhancedSystemConfig.from_database(db_config, 'default')
                if config.telegram_bot_token and config.telegram_id:
                    print("   ✅ Telegram configuration present")
                else:
                    print("   ⚠️  Telegram configuration incomplete")
            else:
                print("   ⚠️  Database not configured, skipping test")
        except Exception as e:
            print(f"   ⚠️  Telegram test failed: {e}")
        
        print("\n✅ Comprehensive test completed")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        return False


def main():
    """Main bootstrap setup function"""
    print("🚀 ENHANCED BYBIT AUTO-TRADING SYSTEM")
    print("🔧 Bootstrap Setup & Validation")
    print("=" * 60)
    print("")
    
    all_checks_passed = True
    
    # Check 0: System requirements
    print("0️⃣ Checking system requirements...")
    if not check_system_requirements():
        all_checks_passed = False
    print("")
    
    # Check 1: Dependencies
    print("1️⃣ Checking Python dependencies...")
    if not check_dependencies():
        all_checks_passed = False
    print("")
    
    # Check 2: Project structure
    print("2️⃣ Validating project structure...")
    if not validate_project_structure():
        all_checks_passed = False
    print("")
    
    # Check 3: Create example config
    print("3️⃣ Setting up configuration files...")
    if not create_example_config():
        all_checks_passed = False
    print("")
    
    # Check 4: Database
    print("4️⃣ Validating database connection...")
    database_ok = validate_database_connection()
    if not database_ok:
        all_checks_passed = False
    print("")
    
    # Check 5: Telegram Configuration
    print("5️⃣ Validating Telegram configuration...")
    telegram_config_valid = validate_telegram_config()
    if not telegram_config_valid:
        all_checks_passed = False
        setup_telegram_bot_instructions()
    print("")
    
    # Check 6: Telegram Connection (only if config is valid and database is OK)
    if telegram_config_valid and database_ok:
        print("6️⃣ Testing Telegram bot connection...")
        if not test_telegram_connection():
            all_checks_passed = False
        print("")
    
    # Check 7: API Credentials Status
    if database_ok:
        print("7️⃣ Checking API credentials status...")
        api_configured = check_api_credentials_status()
        print("")
    else:
        api_configured = False
    
    # Check 8: Create systemd service
    print("8️⃣ Creating systemd service file...")
    create_systemd_service()
    print("")
    
    # Check 9: Comprehensive test
    print("9️⃣ Running comprehensive system test...")
    run_comprehensive_test()
    print("")
    
    # Summary
    print("📋 SETUP SUMMARY:")
    print("=" * 20)
    
    if all_checks_passed:
        print("✅ All system checks passed!")
        
        if api_configured:
            print("🚀 System is ready for auto-trading")
            print("\n🎯 Next steps:")
            print("   • Run: python main.py")
            print("   • Or install as system service")
            print("   • Monitor trades via Telegram notifications")
        else:
            print("🔄 Bootstrap mode will configure API credentials")
            print("\n🎯 Next steps:")
            print("   • Run: python main.py")
            print("   • Follow Telegram instructions to set API keys")
            print("   • System will start auto-trading once configured")
        
    else:
        print("❌ Some system checks failed")
        print("\n🔧 Please fix the issues above and run setup again")
        print("\n💡 Common issues and solutions:")
        print("   • Missing dependencies: pip install -r requirements.txt")
        print("   • Database connection: Check MySQL password in enhanced_config.yaml")
        print("   • Telegram config: Follow the bot setup instructions")
    
    print("\n📱 Telegram Features:")
    print("   • Secure API key configuration via bootstrap mode")
    print("   • Real-time trade notifications")
    print("   • Position management commands")
    print("   • System status monitoring")
    
    print("\n📊 Auto-Trading Features:")
    print("   • Multi-timeframe signal confirmation (30m→1h/4h/6h)")
    print("   • Automated position management with leverage")
    print("   • Risk-based position sizing")
    print("   • Profit-target based auto-closing")
    print("   • Comprehensive database logging")
    
    print(f"\n🕒 Setup completed: {os.popen('date').read().strip()}")
    
    # Additional help
    if not all_checks_passed:
        print("\n🆘 Need help? Check:")
        print("   • README.md for detailed setup instructions")
        print("   • requirements.txt for dependency versions")
        print("   • enhanced_config.yaml for database settings")
        print("   • MySQL service: sudo systemctl status mysql")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()