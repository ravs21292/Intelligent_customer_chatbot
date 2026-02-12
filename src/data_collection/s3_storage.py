"""S3 storage operations for data management."""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from config.aws_config import aws_config
from config.pipeline_config import pipeline_config
from src.utils.logger import logger


class S3Storage:
    """Manages S3 storage operations for data and models."""
    
    def __init__(self):
        self.s3_client = aws_config.get_s3_client()
        self.data_bucket = pipeline_config.S3_BUCKET_DATA
        self.model_bucket = pipeline_config.S3_BUCKET_MODELS
        self.logs_bucket = pipeline_config.S3_BUCKET_LOGS
    
    def upload_data(
        self,
        data: Any,
        key: str,
        bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload data to S3.
        
        Args:
            data: Data to upload (dict, list, or string)
            key: S3 object key
            bucket: S3 bucket name (defaults to data bucket)
            metadata: Optional metadata tags
            
        Returns:
            True if successful
        """
        try:
            bucket = bucket or self.data_bucket
            
            if isinstance(data, (dict, list)):
                body = json.dumps(data, indent=2)
                content_type = "application/json"
            else:
                body = str(data)
                content_type = "text/plain"
            
            put_params = {
                "Bucket": bucket,
                "Key": key,
                "Body": body,
                "ContentType": content_type
            }
            
            if metadata:
                put_params["Metadata"] = metadata
            
            self.s3_client.put_object(**put_params)
            logger.info(f"Uploaded data to s3://{bucket}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            return False
    
    def download_data(self, key: str, bucket: Optional[str] = None) -> Optional[Any]:
        """
        Download data from S3.
        
        Args:
            key: S3 object key
            bucket: S3 bucket name
            
        Returns:
            Downloaded data or None if error
        """
        try:
            bucket = bucket or self.data_bucket
            
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            
            # Try to parse as JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
                
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"Key not found: s3://{bucket}/{key}")
            else:
                logger.error(f"Error downloading from S3: {e}")
            return None
    
    def list_objects(
        self,
        prefix: str,
        bucket: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[str]:
        """
        List objects in S3 with given prefix.
        
        Args:
            prefix: Key prefix to filter
            bucket: S3 bucket name
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object keys
        """
        try:
            bucket = bucket or self.data_bucket
            keys = []
            
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            for page in pages:
                if "Contents" in page:
                    keys.extend([obj["Key"] for obj in page["Contents"]])
            
            return keys
            
        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
            return []
    
    def upload_training_dataset(
        self,
        dataset: List[Dict[str, Any]],
        dataset_name: str,
        version: str
    ) -> str:
        """
        Upload training dataset to S3 with versioning.
        
        Args:
            dataset: Training dataset
            dataset_name: Name of the dataset
            version: Dataset version
            
        Returns:
            S3 key of uploaded dataset
        """
        key = f"{pipeline_config.S3_TRAINING_PATH}/{dataset_name}/v{version}/data.json"
        
        metadata = {
            "dataset_name": dataset_name,
            "version": version,
            "record_count": str(len(dataset)),
            "upload_date": datetime.utcnow().isoformat()
        }
        
        if self.upload_data(dataset, key, metadata=metadata):
            return key
        return ""
    
    def upload_model(
        self,
        model_path: str,
        model_name: str,
        version: str
    ) -> str:
        """
        Upload model artifacts to S3.
        
        Args:
            model_path: Local path to model files
            model_name: Name of the model
            version: Model version
            
        Returns:
            S3 key prefix for uploaded model
        """
        try:
            s3_prefix = f"{pipeline_config.S3_MODEL_PATH}/{model_name}/v{version}/"
            
            # Upload all files in model directory
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, model_path)
                    s3_key = f"{s3_prefix}{relative_path}"
                    
                    self.s3_client.upload_file(
                        local_path,
                        self.model_bucket,
                        s3_key
                    )
            
            logger.info(f"Uploaded model to s3://{self.model_bucket}/{s3_prefix}")
            return s3_prefix
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return ""
    
    def create_bucket_if_not_exists(self, bucket_name: str):
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.info(f"Creating bucket {bucket_name}")
                self.s3_client.create_bucket(Bucket=bucket_name)
                logger.info(f"Bucket {bucket_name} created")
            else:
                raise


# Global S3 storage instance
s3_storage = S3Storage()

