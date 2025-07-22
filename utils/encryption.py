"""
API Secret Encryption/Decryption Utilities
Handles encrypted API secrets for secure storage
FIXED: Improved encryption/decryption with better error handling
"""

import os
import base64
import secrets
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging


class SecretManager:
    """Handles encryption and decryption of API secrets using database-stored password"""
    
    def __init__(self, config=None, password: str = None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Priority order for encryption password:
        # 1. Explicit password parameter
        # 2. Database config encryption_password
        # 3. Environment variable
        # 4. Default fallback
        if password:
            self.password = password
        elif config and hasattr(config, 'encryption_password') and config.encryption_password:
            self.password = config.encryption_password
            self.logger.debug("Using encryption password from database configuration")
        else:
            self.password = os.getenv('SECRET_PASSWORD', 'bybit_trading_system_secure_key_2024')
            if not os.getenv('SECRET_PASSWORD'):
                self.logger.warning("Using default encryption password. Consider setting encryption_password in database config.")
    
    @classmethod
    def from_config(cls, config):
        """Create SecretManager instance from system configuration"""
        return cls(config=config)
    
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from password and salt"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            return kdf.derive(self.password.encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Key derivation failed: {e}")
            raise
    
    def encrypt_secret(self, plain_secret: str, salt: bytes = None) -> str:
        """Encrypt a plain API secret (for storing new secrets)"""
        try:
            if not plain_secret:
                return None
            
            # Generate salt if not provided
            if salt is None:
                salt = secrets.token_bytes(16)
            
            # Derive key
            key = self._derive_key(salt)
            
            # Add PKCS7 padding
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plain_secret.encode('utf-8'))
            padded_data += padder.finalize()
            
            # Encrypt using AES-256-CBC
            iv = secrets.token_bytes(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine salt + iv + ciphertext, then base64 encode
            encrypted_data = salt + iv + ciphertext
            encrypted_secret = base64.b64encode(encrypted_data).decode('utf-8')
            
            self.logger.debug("Secret encrypted successfully")
            return encrypted_secret
            
        except Exception as e:
            self.logger.error(f"Secret encryption failed: {e}")
            return None
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt an encrypted API secret"""
        try:
            if not encrypted_secret or encrypted_secret == 'None' or encrypted_secret == '':
                self.logger.warning("Empty or None encrypted secret provided")
                return None
            
            # Check if the secret looks like it might not be encrypted (plain text)
            if len(encrypted_secret) < 50 and not encrypted_secret.startswith('==') and '==' not in encrypted_secret:
                self.logger.warning("Secret appears to be plain text, returning as-is")
                return encrypted_secret
            
            try:
                # Decode base64
                encrypted_data = base64.b64decode(encrypted_secret.encode('utf-8'))
            except Exception as e:
                self.logger.warning(f"Base64 decode failed, assuming plain text: {e}")
                return encrypted_secret
            
            # Check minimum length (salt + iv + some ciphertext)
            if len(encrypted_data) < 32:
                self.logger.warning("Encrypted data too short, assuming plain text")
                return encrypted_secret
            
            # Extract salt (first 16 bytes), iv (next 16 bytes), and ciphertext
            salt = encrypted_data[:16]
            iv = encrypted_data[16:32]
            ciphertext = encrypted_data[32:]
            
            # Derive key
            key = self._derive_key(salt)
            
            # Decrypt using AES-256-CBC
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove PKCS7 padding
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext)
            plaintext += unpadder.finalize()
            
            decrypted_secret = plaintext.decode('utf-8')
            
            self.logger.debug("Secret decrypted successfully")
            return decrypted_secret
            
        except Exception as e:
            self.logger.error(f"Secret decryption failed: {e}")
            self.logger.error(f"Encrypted secret length: {len(encrypted_secret) if encrypted_secret else 0}")
            
            # If decryption fails, check if it might be plain text
            if encrypted_secret and len(encrypted_secret) > 10 and len(encrypted_secret) < 200:
                self.logger.warning("Decryption failed, returning as plain text")
                return encrypted_secret
            
            return None
    
    def is_encrypted(self, secret: str) -> bool:
        """Check if a secret appears to be encrypted"""
        if not secret:
            return False
        
        try:
            # Try to decode as base64
            base64.b64decode(secret)
            # If successful and long enough, likely encrypted
            return len(secret) > 50
        except:
            # If base64 decode fails, likely plain text
            return False
    
    def test_encryption_decryption(self) -> bool:
        """Test encryption and decryption functionality"""
        try:
            test_secret = "test_api_key_123456789"
            
            # Encrypt
            encrypted = self.encrypt_secret(test_secret)
            if not encrypted:
                self.logger.error("Encryption test failed - no encrypted result")
                return False
            
            # Decrypt
            decrypted = self.decrypt_secret(encrypted)
            if not decrypted:
                self.logger.error("Decryption test failed - no decrypted result")
                return False
            
            # Compare
            if test_secret == decrypted:
                self.logger.debug("✅ Encryption/decryption test passed")
                return True
            else:
                self.logger.error(f"Encryption/decryption test failed - mismatch: '{test_secret}' != '{decrypted}'")
                return False
                
        except Exception as e:
            self.logger.error(f"Encryption/decryption test failed: {e}")
            return False