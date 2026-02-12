"""Pipeline and MLOps configuration."""

import os
from typing import Dict


class PipelineConfig:
    """Configuration for training and deployment pipelines."""
    
    # SageMaker Configuration
    SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
    SAGEMAKER_INSTANCE_TYPE = "ml.m5.xlarge"
    SAGEMAKER_TRAINING_INSTANCE_TYPE = "ml.g4dn.xlarge"
    
    # S3 Paths
    S3_BUCKET_DATA = os.getenv("S3_BUCKET_DATA", "customer-chatbot-data")
    S3_BUCKET_MODELS = os.getenv("S3_BUCKET_MODELS", "customer-chatbot-models")
    S3_BUCKET_LOGS = os.getenv("S3_BUCKET_LOGS", "customer-chatbot-logs")
    
    # Data Paths
    S3_DATA_PATH = f"s3://{S3_BUCKET_DATA}/data"
    S3_MODEL_PATH = f"s3://{S3_BUCKET_MODELS}/models"
    S3_TRAINING_PATH = f"s3://{S3_BUCKET_DATA}/training"
    
    # Retraining Configuration
    RETRAINING_SCHEDULE = "cron(0 2 ? * SUN)"  # Weekly on Sunday 2 AM
    MIN_NEW_SAMPLES_FOR_RETRAIN = 1000
    PERFORMANCE_DEGRADATION_THRESHOLD = 0.05  # 5% drop in accuracy
    
    # Model Monitoring
    ENABLE_MODEL_MONITOR = os.getenv("ENABLE_MONITORING", "True").lower() == "true"
    DRIFT_DETECTION_THRESHOLD = 0.1
    MONITORING_SCHEDULE = "cron(0 * * * ? *)"  # Hourly
    
    # CI/CD Configuration
    GITHUB_REPO = os.getenv("GITHUB_REPO", "")
    DEPLOYMENT_STAGES = ["dev", "staging", "prod"]
    
    # A/B Testing
    ENABLE_AB_TESTING = True
    TRAFFIC_SPLIT = {
        "baseline": 0.5,
        "new_model": 0.5
    }


class DataCollectionConfig:
    """Configuration for data collection."""
    
    KINESIS_STREAM_NAME = os.getenv("KINESIS_STREAM_NAME", "customer-chat-stream")
    KINESIS_SHARD_COUNT = 2
    BATCH_SIZE = 100
    FLUSH_INTERVAL = 60  # seconds
    
    # Data Retention
    S3_RETENTION_DAYS = 90
    RAW_DATA_PREFIX = "raw/chat-logs"
    PROCESSED_DATA_PREFIX = "processed/datasets"


# Global pipeline configuration instances
pipeline_config = PipelineConfig()
data_collection_config = DataCollectionConfig()

