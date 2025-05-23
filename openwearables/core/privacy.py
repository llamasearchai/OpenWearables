import os
import json
import hashlib
import logging
import base64
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Conditional imports for encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

logger = logging.getLogger("OpenWearables.Privacy")

class PrivacyManager:
    """
    Manages privacy and security features for health data.
    
    This class handles data anonymization, encryption, and 
    compliance with health data privacy regulations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the privacy manager.
        
        Args:
            config: Configuration dictionary for privacy settings
        """
        self.config = config or {}
        self.encryption_enabled = self.config.get("encryption", True) and HAS_CRYPTO
        self.anonymization_enabled = self.config.get("anonymization", True)
        self.data_retention_days = self.config.get("data_retention", 90)
        
        # Initialize encryption key if enabled
        self.encryption_key = None
        if self.encryption_enabled:
            self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption system."""
        if not HAS_CRYPTO:
            logger.warning("Cryptography package not available. Encryption disabled.")
            self.encryption_enabled = False
            return
        
        try:
            # In a production system, this would use a securely stored key
            # For demo purposes, we'll generate a key from a password
            password = self.config.get("encryption_password", "OpenWearables_demo_key")
            salt = b'OpenWearables_salt'  # In production, this would be randomly generated and stored
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self.encryption_key = Fernet(key)
            logger.info("Encryption initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing encryption: {str(e)}")
            self.encryption_enabled = False
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data as bytes
        """
        if not self.encryption_enabled or not self.encryption_key:
            if isinstance(data, str):
                return data.encode()
            return data
        
        try:
            if isinstance(data, str):
                data = data.encode()
            
            return self.encryption_key.encrypt(data)
        
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            if isinstance(data, str):
                return data.encode()
            return data
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data as bytes
        """
        if not self.encryption_enabled or not self.encryption_key:
            return encrypted_data
        
        try:
            return self.encryption_key.decrypt(encrypted_data)
        
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            return encrypted_data
    
    def anonymize_identifier(self, identifier: str) -> str:
        """
        Anonymize an identifier using one-way hashing.
        
        Args:
            identifier: Personal identifier to anonymize
            
        Returns:
            Anonymized identifier
        """
        if not self.anonymization_enabled:
            return identifier
        
        try:
            # Add a salt for security
            salt = "OpenWearables_anonymization_salt"
            salted = (identifier + salt).encode()
            
            # Create SHA-256 hash
            hashed = hashlib.sha256(salted).hexdigest()
            return hashed
        
        except Exception as e:
            logger.error(f"Error anonymizing identifier: {str(e)}")
            return identifier
    
    def is_data_expired(self, timestamp: float) -> bool:
        """
        Check if data has exceeded retention period.
        
        Args:
            timestamp: Unix timestamp of data creation
            
        Returns:
            True if data should be deleted, False otherwise
        """
        if self.data_retention_days <= 0:
            return False  # No expiration
        
        creation_date = datetime.fromtimestamp(timestamp)
        expiration_date = creation_date + timedelta(days=self.data_retention_days)
        
        return datetime.now() > expiration_date
    
    def sanitize_output(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize analysis results for privacy.
        
        Args:
            analysis_results: Raw analysis results
            
        Returns:
            Sanitized results with any sensitive information removed
        """
        if not analysis_results:
            return {}
        
        # Create a deep copy to avoid modifying the original
        sanitized = json.loads(json.dumps(analysis_results))
        
        # Remove any potentially sensitive data
        sensitive_keys = ["user_id", "device_id", "location", "raw_data"]
        
        def remove_sensitive_keys(data: Dict[str, Any]) -> None:
            """Recursively remove sensitive keys from nested dictionaries."""
            if not isinstance(data, dict):
                return
            
            for key in list(data.keys()):
                if key in sensitive_keys:
                    data.pop(key)
                elif isinstance(data[key], dict):
                    remove_sensitive_keys(data[key])
                elif isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            remove_sensitive_keys(item)
        
        remove_sensitive_keys(sanitized)
        return sanitized
    
    def generate_data_access_log(self, user_id: str, data_type: str, purpose: str) -> Dict[str, Any]:
        """
        Generate an audit log entry for data access.
        
        Args:
            user_id: ID of user accessing the data
            data_type: Type of data being accessed
            purpose: Purpose of access
            
        Returns:
            Audit log entry
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self.anonymize_identifier(user_id) if self.anonymization_enabled else user_id,
            "data_type": data_type,
            "purpose": purpose,
            "access_id": hashlib.md5(os.urandom(16)).hexdigest()
        }
        
        return log_entry


class EncryptedData:
    """
    Container class for encrypted health data with metadata.
    
    This class provides a secure way to store and manage encrypted
    health data while maintaining necessary metadata for processing.
    """
    
    def __init__(self, data: Any, privacy_manager: PrivacyManager = None):
        """
        Initialize EncryptedData container.
        
        Args:
            data: Data to encrypt and store
            privacy_manager: PrivacyManager instance for encryption
        """
        self.privacy_manager = privacy_manager or PrivacyManager()
        self.created_at = datetime.now()
        self.data_type = type(data).__name__
        self.is_encrypted = self.privacy_manager.encryption_enabled
        
        # Encrypt the data
        if isinstance(data, dict):
            data_str = json.dumps(data, default=str)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(list(data), default=str)
        else:
            data_str = str(data)
        
        self.encrypted_content = self.privacy_manager.encrypt_data(data_str)
        self.content_hash = hashlib.sha256(data_str.encode()).hexdigest()
    
    def decrypt(self) -> Any:
        """
        Decrypt and return the stored data.
        
        Returns:
            Decrypted data in its original format
        """
        try:
            # Decrypt the data
            decrypted_bytes = self.privacy_manager.decrypt_data(self.encrypted_content)
            decrypted_str = decrypted_bytes.decode('utf-8')
            
            # Try to parse as JSON if it was originally a dict or list
            if self.data_type in ['dict', 'list', 'tuple']:
                return json.loads(decrypted_str)
            
            # Return as string for other types
            return decrypted_str
            
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            return None
    
    def verify_integrity(self) -> bool:
        """
        Verify data integrity using hash comparison.
        
        Returns:
            True if data integrity is verified, False otherwise
        """
        try:
            # Decrypt and re-hash the data
            decrypted_data = self.decrypt()
            if decrypted_data is None:
                return False
            
            # Convert back to string for hashing
            if isinstance(decrypted_data, dict):
                data_str = json.dumps(decrypted_data, default=str)
            elif isinstance(decrypted_data, (list, tuple)):
                data_str = json.dumps(list(decrypted_data), default=str)
            else:
                data_str = str(decrypted_data)
            
            current_hash = hashlib.sha256(data_str.encode()).hexdigest()
            return current_hash == self.content_hash
            
        except Exception as e:
            logger.error(f"Error verifying data integrity: {str(e)}")
            return False
    
    def is_expired(self) -> bool:
        """
        Check if the encrypted data has expired based on retention policy.
        
        Returns:
            True if data should be deleted, False otherwise
        """
        return self.privacy_manager.is_data_expired(self.created_at.timestamp())
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the encrypted data.
        
        Returns:
            Dictionary containing metadata
        """
        return {
            "created_at": self.created_at.isoformat(),
            "data_type": self.data_type,
            "is_encrypted": self.is_encrypted,
            "content_hash": self.content_hash,
            "size_bytes": len(self.encrypted_content),
            "is_expired": self.is_expired(),
            "integrity_verified": self.verify_integrity()
        }
    
    def export_for_analysis(self, purpose: str = "health_analysis") -> Dict[str, Any]:
        """
        Export data for analysis with privacy safeguards.
        
        Args:
            purpose: Purpose of the data export for audit logging
            
        Returns:
            Sanitized data ready for analysis
        """
        # Log the data access
        access_log = self.privacy_manager.generate_data_access_log(
            user_id="system",
            data_type=self.data_type,
            purpose=purpose
        )
        
        # Decrypt the data
        decrypted_data = self.decrypt()
        if decrypted_data is None:
            return {"error": "Failed to decrypt data"}
        
        # Sanitize the data for privacy
        if isinstance(decrypted_data, dict):
            sanitized_data = self.privacy_manager.sanitize_output(decrypted_data)
        else:
            sanitized_data = {"data": decrypted_data}
            sanitized_data = self.privacy_manager.sanitize_output(sanitized_data)
        
        return {
            "data": sanitized_data,
            "access_log": access_log,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "purpose": purpose,
                "data_type": self.data_type
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert EncryptedData to dictionary format.
        
        Returns:
            Dictionary representation (without decrypted content)
        """
        return {
            "metadata": self.get_metadata(),
            "encrypted_content_size": len(self.encrypted_content),
            "is_accessible": not self.is_expired()
        }
    
    def __repr__(self) -> str:
        """String representation of EncryptedData."""
        return f"EncryptedData(type={self.data_type}, size={len(self.encrypted_content)} bytes, encrypted={self.is_encrypted})"
    
    def __len__(self) -> int:
        """Return size of encrypted content in bytes."""
        return len(self.encrypted_content)
    
    @classmethod
    def from_file(cls, filepath: str, privacy_manager: PrivacyManager = None) -> 'EncryptedData':
        """
        Create EncryptedData from a file.
        
        Args:
            filepath: Path to file containing data
            privacy_manager: PrivacyManager instance
            
        Returns:
            EncryptedData instance
        """
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                else:
                    data = f.read()
            
            return cls(data, privacy_manager)
            
        except Exception as e:
            logger.error(f"Error loading data from file {filepath}: {str(e)}")
            return cls({}, privacy_manager)
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save encrypted data to file.
        
        Args:
            filepath: Path to output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save metadata and encrypted content
            save_data = {
                "metadata": self.get_metadata(),
                "encrypted_content": base64.b64encode(self.encrypted_content).decode('utf-8')
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving encrypted data to {filepath}: {str(e)}")
            return False