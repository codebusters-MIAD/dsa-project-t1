import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AWSSecretsManager:
    """Handle AWS Secrets Manager operations."""
    
    @staticmethod
    def get_secret(
        secret_arn: str,
        region: str,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None
    ) -> dict:
        """
        Retrieve secret from AWS Secrets Manager.
        """
        try:
            import boto3
            
            session_kwargs = {'region_name': region}
            
            if access_key_id and secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': access_key_id,
                    'aws_secret_access_key': secret_access_key
                })
            
            session = boto3.session.Session(**session_kwargs)
            client = session.client('secretsmanager')
            
            response = client.get_secret_value(SecretId=secret_arn)
            secret = json.loads(response['SecretString'])
            
            logger.info(f"Successfully retrieved secret from AWS Secrets Manager")
            return secret
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret from AWS: {e}")
            raise
