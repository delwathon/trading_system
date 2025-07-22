#!/usr/bin/env python3
"""
Database Schema Update Script
Adds the encryption_password column to existing system_config table
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import DatabaseConfig
from database.models import DatabaseManager
import pymysql


def update_database_schema():
    """Add encryption_password column to system_config table"""
    try:
        print("🔄 UPDATING DATABASE SCHEMA")
        print("=" * 40)
        
        # Load database config
        config_path = 'enhanced_config.yaml'
        if not os.path.exists(config_path):
            print("❌ Configuration file not found")
            return False
        
        db_config = DatabaseConfig.from_yaml_file(config_path)
        print(f"📊 Connecting to database: {db_config.database}")
        
        # Connect to database
        connection = pymysql.connect(
            host=db_config.host,
            port=db_config.port,
            user=db_config.username,
            password=db_config.password,
            database=db_config.database,
            charset='utf8mb4'
        )
        
        cursor = connection.cursor()
        
        # Check if encryption_password column already exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_schema = %s 
            AND table_name = 'system_config' 
            AND column_name = 'encryption_password'
        """, (db_config.database,))
        
        column_exists = cursor.fetchone()[0] > 0
        
        if column_exists:
            print("✅ encryption_password column already exists")
        else:
            print("➕ Adding encryption_password column...")
            
            # Add the column
            cursor.execute("""
                ALTER TABLE system_config 
                ADD COLUMN encryption_password VARCHAR(255) 
                DEFAULT 'bybit_trading_system_secure_key_2024'
            """)
            
            print("✅ encryption_password column added successfully")
        
        # Update existing records to have the default encryption password
        cursor.execute("""
            UPDATE system_config 
            SET encryption_password = 'bybit_trading_system_secure_key_2024' 
            WHERE encryption_password IS NULL OR encryption_password = ''
        """)
        
        affected_rows = cursor.rowcount
        if affected_rows > 0:
            print(f"✅ Updated {affected_rows} records with default encryption password")
        
        connection.commit()
        
        # Verify the update
        cursor.execute("SELECT config_name, encryption_password FROM system_config")
        results = cursor.fetchall()
        
        print("\n📋 Current configurations:")
        for config_name, enc_password in results:
            print(f"   {config_name}: {enc_password}")
        
        cursor.close()
        connection.close()
        
        print("\n✅ Database schema update completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database schema update failed: {e}")
        return False


def recreate_tables():
    """Recreate all tables (alternative approach)"""
    try:
        print("\n🔄 RECREATING DATABASE TABLES")
        print("=" * 40)
        
        # Load database config
        config_path = 'enhanced_config.yaml'
        db_config = DatabaseConfig.from_yaml_file(config_path)
        
        # Use DatabaseManager to recreate tables
        db_manager = DatabaseManager(db_config.get_database_url())
        
        if db_manager.test_connection():
            print("✅ Database connection successful")
            
            # This will create all tables including the new encryption_password column
            db_manager.create_tables()
            
            print("✅ All tables recreated with updated schema")
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Table recreation failed: {e}")
        return False


def main():
    """Main function"""
    print("🛠️ DATABASE SCHEMA UPDATE TOOL")
    print("=" * 50)
    print("This tool adds the encryption_password field to your database")
    print("")
    
    try:
        # Method 1: Add column to existing table
        if update_database_schema():
            print("\n🎉 Schema update completed!")
            print("The encryption password is now stored in the database.")
            print("You can modify it via SQL or through the system configuration.")
        else:
            print("\n⚠️ Schema update failed. Trying alternative method...")
            
            # Method 2: Recreate tables (if adding column failed)
            if recreate_tables():
                print("✅ Tables recreated successfully!")
            else:
                print("❌ Both update methods failed.")
                print("Please manually add the column:")
                print("ALTER TABLE system_config ADD COLUMN encryption_password VARCHAR(255) DEFAULT 'bybit_trading_system_secure_key_2024';")
    
    except Exception as e:
        print(f"❌ Update tool failed: {e}")


if __name__ == "__main__":
    main()