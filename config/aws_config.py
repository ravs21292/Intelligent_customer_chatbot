"""AWS service configuration and client initialization."""

import os
import boto3
from typing import Optional
from botocore.config import Config


class AWSConfig:
    """Manages AWS service configurations and clients."""
    
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.config = Config(
            region_name=self.region,
            retries={"max_attempts": 3, "mode": "standard"}
        )
        
    def get_s3_client(self):
        """Get S3 client."""
        return boto3.client("s3", config=self.config)
    
    def get_kinesis_client(self):
        """Get Kinesis client."""
        return boto3.client("kinesis", config=self.config)
    
    def get_sagemaker_client(self):
        """Get SageMaker client."""
        return boto3.client("sagemaker", config=self.config)
    
    def get_bedrock_client(self):
        """Get Bedrock client."""
        return boto3.client("bedrock-runtime", config=self.config)
    
    def get_opensearch_client(self):
        """Get OpenSearch client."""
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.region,
            "es",
            session_token=credentials.token
        )
        
        endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        return OpenSearch(
            hosts=[{"host": endpoint.replace("https://", ""), "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    
    def get_cloudwatch_client(self):
        """Get CloudWatch client."""
        return boto3.client("cloudwatch", config=self.config)
    
    def get_lambda_client(self):
        """Get Lambda client."""
        return boto3.client("lambda", config=self.config)
    
    def get_eventbridge_client(self):
        """Get EventBridge client."""
        return boto3.client("events", config=self.config)


# Global AWS configuration instance
aws_config = AWSConfig()

