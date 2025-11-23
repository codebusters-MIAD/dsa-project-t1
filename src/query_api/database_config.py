import logging
from typing import Dict
from urllib.parse import urlparse

from .aws_utils import AWSSecretsManager

logger = logging.getLogger(__name__)


class DatabaseConfigParser:
    """Parse database configuration from multiple sources."""
    
    @staticmethod
    def from_secret_manager(
        secret_arn: str,
        region: str,
        use_credentials: bool,
        access_key_id: str = "",
        secret_access_key: str = "",
        fallback_config: Dict[str, any] = None
    ) -> Dict[str, any]:
        """
        Get database credentials from AWS Secrets Manager.
        """
        if fallback_config is None:
            fallback_config = {}
        
        if use_credentials:
            credentials = AWSSecretsManager.get_secret(
                secret_arn=secret_arn,
                region=region,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key
            )
        else:
            credentials = AWSSecretsManager.get_secret(
                secret_arn=secret_arn,
                region=region
            )
        
        return {
            'host': credentials.get('host', fallback_config.get('host', 'localhost')),
            'port': credentials.get('port', fallback_config.get('port', 5432)),
            'database': credentials.get('dbname', fallback_config.get('database', '')),
            'user': credentials.get('username', fallback_config.get('user', '')),
            'password': credentials.get('password', fallback_config.get('password', ''))
        }
    
    @staticmethod
    def from_url(database_url: str, fallback_config: Dict[str, any] = None) -> Dict[str, any]:
        """
        Parse database credentials from connection URL.
        """
        if fallback_config is None:
            fallback_config = {}
        
        parsed = urlparse(database_url)
        
        return {
            'host': parsed.hostname or fallback_config.get('host', 'localhost'),
            'port': parsed.port or fallback_config.get('port', 5432),
            'database': parsed.path.lstrip('/') or fallback_config.get('database', ''),
            'user': parsed.username or fallback_config.get('user', ''),
            'password': parsed.password or fallback_config.get('password', '')
        }
    
    @staticmethod
    def from_components(host: str, port: int, database: str, user: str, password: str) -> Dict[str, any]:
        """
        Create database config from individual components.
        """
        return {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
