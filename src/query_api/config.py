import logging
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # API metadata
    app_name: str = "FilmLens Query API"
    version: str = "0.1.0"
    description: str = "Query service for movie triggers database"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    
    # Environment
    environment: str = "development"
    
    # AWS Configuration
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    rds_secret_manager_arn: str = ""
    aws_use_iam_role: bool = True
    
    # Database
    database_url: str = ""
    db_host: str = "database"
    db_port: int = 5432
    db_name: str = "filmlens"
    db_user: str = "filmlens_user"
    db_password: str = "filmlens_dev_2025"
    
    # CORS
    allowed_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    def get_database_config(self) -> dict:
        """
        Get database configuration with priority: RDS_SECRET_MANAGER_ARN > DATABASE_URL > defaults.
        """
        from .database_config import DatabaseConfigParser
        
        fallback = {
            'host': self.db_host,
            'port': self.db_port,
            'database': self.db_name,
            'user': self.db_user,
            'password': self.db_password
        }
        
        if self.rds_secret_manager_arn:
            logger.info("Using AWS Secret Manager for database credentials")
            try:
                return DatabaseConfigParser.from_secret_manager(
                    secret_arn=self.rds_secret_manager_arn,
                    region=self.aws_default_region,
                    use_credentials=not self.aws_use_iam_role,
                    access_key_id=self.aws_access_key_id,
                    secret_access_key=self.aws_secret_access_key,
                    fallback_config=fallback
                )
            except Exception as e:
                logger.error(f"Failed to retrieve from Secret Manager: {e}")
                logger.info("Attempting fallback to DATABASE_URL or defaults")
        
        if self.database_url:
            logger.info("Using DATABASE_URL from environment")
            return DatabaseConfigParser.from_url(self.database_url, fallback_config=fallback)
        
        logger.info("Using default database credentials")
        return DatabaseConfigParser.from_components(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
