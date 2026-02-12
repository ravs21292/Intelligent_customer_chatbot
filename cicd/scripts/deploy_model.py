#!/usr/bin/env python3
"""
Model deployment script with automatic approval logic.
Usage: python deploy_model.py --model-uri s3://... --compare-with current
"""

import argparse
import boto3
import json
import time
from datetime import datetime
from src.utils.logger import logger


def get_model_metrics(model_package_arn):
    """Get metrics from model package."""
    sagemaker = boto3.client("sagemaker")
    response = sagemaker.describe_model_package(ModelPackageName=model_package_arn)
    return response.get("ModelMetrics", {})


def get_current_production_model():
    """Get current production model metrics."""
    sagemaker = boto3.client("sagemaker")
    
    # Get model package group
    response = sagemaker.list_model_packages(
        ModelPackageGroupName="intent-classifier",
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending"
    )
    
    if response["ModelPackageSummaryList"]:
        latest = response["ModelPackageSummaryList"][0]
        return {
            "model_package_arn": latest["ModelPackageArn"],
            "metrics": get_model_metrics(latest["ModelPackageArn"])
        }
    return None


def compare_models(new_metrics, current_metrics):
    """Compare model metrics."""
    if not current_metrics:
        return True, "No current model"
    
    new_accuracy = new_metrics.get("ModelQuality", {}).get("Statistics", {}).get("Accuracy", {}).get("Value", 0)
    current_accuracy = current_metrics.get("ModelQuality", {}).get("Statistics", {}).get("Accuracy", {}).get("Value", 0)
    
    improvement = new_accuracy - current_accuracy
    
    if improvement > 0.01:  # 1% improvement
        return True, f"Model improved by {improvement:.2%}"
    elif improvement > -0.01:  # Within 1% (acceptable)
        return True, f"Model performance similar ({improvement:.2%})"
    else:
        return False, f"Model degraded by {abs(improvement):.2%}"


def deploy_model(model_uri, endpoint_name, auto_approve=False):
    """Deploy model to SageMaker endpoint."""
    from sagemaker.huggingface import HuggingFaceModel
    from sagemaker import Session
    
    session = Session()
    
    # Create model
    model = HuggingFaceModel(
        model_data=model_uri,
        role="arn:aws:iam::account:role/SageMakerExecutionRole",
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39"
    )
    
    # Deploy endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name
    )
    
    logger.info(f"Model deployed to endpoint: {endpoint_name}")
    return endpoint_name


def main():
    parser = argparse.ArgumentParser(description="Deploy model with automatic approval")
    parser.add_argument("--model-uri", required=True, help="S3 URI of model artifacts")
    parser.add_argument("--endpoint-name", default="intent-classifier-endpoint", help="Endpoint name")
    parser.add_argument("--compare-with", choices=["current", "none"], default="current")
    parser.add_argument("--auto-approve-if-better", action="store_true", help="Auto-approve if better")
    
    args = parser.parse_args()
    
    # Get new model metrics (would be from training job output)
    # For demo, using placeholder
    new_metrics = {
        "ModelQuality": {
            "Statistics": {
                "Accuracy": {"Value": 0.91},
                "F1Score": {"Value": 0.89}
            }
        }
    }
    
    # Compare with current if requested
    if args.compare_with == "current":
        current_model = get_current_production_model()
        if current_model:
            is_better, reason = compare_models(new_metrics, current_model["metrics"])
            logger.info(f"Model comparison: {reason}")
            
            if not is_better and not args.auto_approve_if_better:
                logger.warning("Model not better than current. Deployment cancelled.")
                return
        else:
            logger.info("No current model found. Proceeding with deployment.")
    
    # Deploy model
    endpoint_name = deploy_model(args.model_uri, args.endpoint_name)
    
    logger.info(f" Model deployed successfully to {endpoint_name}")


if __name__ == "__main__":
    main()

