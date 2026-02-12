"""Secure secrets management using AWS Secrets Manager."""

import os
import json
import boto3
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
from src.utils.logger import logger


class SecretsManager:
    """Manages secrets using AWS Secrets Manager."""
    
    def __init__(self, region: str = None):
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("secretsmanager", region_name=self.region)
    
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """
        Retrieve secret from AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Secret dictionary
        """
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_string = response.get("SecretString", "")
            return json.loads(secret_string)
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                logger.warning(f"Secret {secret_name} not found, falling back to environment variables")
                return {}
            else:
                logger.error(f"Error retrieving secret: {e}")
                raise
    
    def get_aws_credentials(self, secret_name: str = "customer-chatbot/secrets") -> Dict[str, str]:
        """
        Get AWS credentials from Secrets Manager.
        
        Args:
            secret_name: Secret name containing credentials
            
        Returns:
            Dictionary with AWS credentials
        """
        secret = self.get_secret(secret_name)
        
        return {
            "aws_access_key_id": secret.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": secret.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region": secret.get("aws_region") or os.getenv("AWS_REGION", "us-east-1")
        }
    
    def create_secret(
        self,
        secret_name: str,
        secret_value: Dict[str, Any],
        description: str = ""
    ) -> str:
        """
        Create a new secret in Secrets Manager.
        
        Args:
            secret_name: Name for the secret
            secret_value: Secret data
            description: Optional description
            
        Returns:
            Secret ARN
        """
        try:
            response = self.client.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value),
                Description=description
            )
            logger.info(f"Secret {secret_name} created")
            return response["ARN"]
            
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceExistsException":
                # Update existing secret
                response = self.client.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps(secret_value)
                )
                logger.info(f"Secret {secret_name} updated")
                return response["ARN"]
            else:
                logger.error(f"Error creating secret: {e}")
                raise


# Global secrets manager instance
secrets_manager = SecretsManager()

# Helper function to get credentials securely
def get_secure_credentials(use_secrets_manager: bool = True) -> Dict[str, str]:
    """
    Get AWS credentials securely.
    
    Args:
        use_secrets_manager: Whether to use Secrets Manager (True) or env vars (False)
        
    Returns:
        Credentials dictionary
    """
    if use_secrets_manager:
        try:
            return secrets_manager.get_aws_credentials()
        except Exception as e:
            logger.warning(f"Failed to get secrets from Secrets Manager: {e}, using environment variables")
    
    # Fallback to environment variables
    return {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1")
    }

