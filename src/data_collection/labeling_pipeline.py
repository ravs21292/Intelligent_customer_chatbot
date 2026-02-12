"""Data labeling pipeline using SageMaker Ground Truth."""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import boto3
from config.aws_config import aws_config
from config.pipeline_config import pipeline_config
from src.data_collection.s3_storage import s3_storage
from src.utils.logger import logger


class LabelingPipeline:
    """Manages data labeling using SageMaker Ground Truth."""
    
    def __init__(self):
        self.sagemaker = aws_config.get_sagemaker_client()
        self.role_arn = pipeline_config.SAGEMAKER_ROLE_ARN
        self.workteam_arn = None  # Set if using private workforce
    
    def create_labeling_job(
        self,
        job_name: str,
        input_manifest_s3_uri: str,
        output_s3_uri: str,
        label_category_config: Dict[str, Any],
        task_type: str = "text-classification"
    ) -> str:
        """
        Create a SageMaker Ground Truth labeling job.
        
        Args:
            job_name: Unique name for the labeling job
            input_manifest_s3_uri: S3 URI of input manifest file
            output_s3_uri: S3 URI for output
            label_category_config: Label category configuration
            task_type: Type of labeling task
            
        Returns:
            Labeling job ARN
        """
        try:
            # Create labeling job request
            request = {
                "LabelingJobName": job_name,
                "LabelAttributeName": "category",
                "InputConfig": {
                    "DataSource": {
                        "S3DataSource": {
                            "ManifestS3Uri": input_manifest_s3_uri
                        }
                    }
                },
                "OutputConfig": {
                    "S3OutputPath": output_s3_uri
                },
                "RoleArn": self.role_arn,
                "HumanTaskConfig": {
                    "WorkteamArn": self.workteam_arn or f"arn:aws:sagemaker:{aws_config.region}:394669845002:workteam/public-crowd/default",
                    "UiConfig": {
                        "UiTemplateS3Uri": self._get_ui_template_uri(task_type)
                    },
                    "PreHumanTaskLambdaArn": f"arn:aws:lambda:{aws_config.region}:081040173940:function:PRE-{task_type}",
                    "TaskKeywords": ["text", "classification"],
                    "TaskTitle": "Classify Customer Support Messages",
                    "TaskDescription": "Classify the intent of customer support messages",
                    "NumberOfHumanWorkersPerDataObject": 1,
                    "TaskTimeLimitInSeconds": 300,
                    "AnnotationConsolidationConfig": {
                        "AnnotationConsolidationLambdaArn": f"arn:aws:lambda:{aws_config.region}:081040173940:function:ACS-{task_type}"
                    }
                },
                "LabelCategoryConfigS3Uri": self._create_label_config(label_category_config, job_name)
            }
            
            response = self.sagemaker.create_labeling_job(**request)
            job_arn = response["LabelingJobArn"]
            
            logger.info(f"Labeling job created: {job_arn}")
            return job_arn
            
        except Exception as e:
            logger.error(f"Error creating labeling job: {e}")
            raise
    
    def _get_ui_template_uri(self, task_type: str) -> str:
        """Get UI template S3 URI for the task type."""
        # In production, upload custom UI templates to S3
        templates = {
            "text-classification": f"s3://aws-ml-blog-samples/{task_type}/template.liquid",
            "named-entity-recognition": f"s3://aws-ml-blog-samples/{task_type}/template.liquid"
        }
        return templates.get(task_type, templates["text-classification"])
    
    def _create_label_config(self, label_config: Dict[str, Any], job_name: str) -> str:
        """Create and upload label category configuration."""
        config_key = f"labeling-jobs/{job_name}/label-config.json"
        s3_storage.upload_data(label_config, config_key)
        return f"s3://{pipeline_config.S3_BUCKET_DATA}/{config_key}"
    
    def create_manifest_file(
        self,
        data_records: List[Dict[str, Any]],
        manifest_name: str
    ) -> str:
        """
        Create manifest file for labeling job.
        
        Args:
            data_records: List of data records to label
            manifest_name: Name for the manifest file
            
        Returns:
            S3 URI of manifest file
        """
        manifest = []
        for record in data_records:
            manifest_entry = {
                "source": record.get("text", record.get("message", "")),
                "metadata": {
                    "user_id": record.get("user_id", ""),
                    "timestamp": record.get("timestamp", datetime.utcnow().isoformat())
                }
            }
            manifest.append(manifest_entry)
        
        manifest_key = f"labeling-jobs/{manifest_name}/manifest.jsonl"
        manifest_content = "\n".join([json.dumps(entry) for entry in manifest])
        
        s3_storage.upload_data(manifest_content, manifest_key)
        manifest_uri = f"s3://{pipeline_config.S3_BUCKET_DATA}/{manifest_key}"
        
        logger.info(f"Manifest file created: {manifest_uri}")
        return manifest_uri
    
    def get_labeling_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of a labeling job."""
        try:
            response = self.sagemaker.describe_labeling_job(LabelingJobName=job_name)
            return {
                "status": response["LabelingJobStatus"],
                "creation_time": response["CreationTime"].isoformat(),
                "label_count": response.get("LabelCounters", {}),
                "output_location": response.get("LabelingJobOutput", {}).get("OutputDatasetS3Uri", "")
            }
        except Exception as e:
            logger.error(f"Error getting labeling job status: {e}")
            return {}
    
    def wait_for_job_completion(self, job_name: str, timeout: int = 3600) -> bool:
        """Wait for labeling job to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_labeling_job_status(job_name)
            job_status = status.get("status", "Unknown")
            
            if job_status == "Completed":
                logger.info(f"Labeling job {job_name} completed")
                return True
            elif job_status in ["Failed", "Stopped"]:
                logger.error(f"Labeling job {job_name} failed with status: {job_status}")
                return False
            
            time.sleep(30)  # Check every 30 seconds
        
        logger.warning(f"Labeling job {job_name} timed out")
        return False
    
    def download_labeled_data(self, output_s3_uri: str) -> List[Dict[str, Any]]:
        """Download and parse labeled data from S3."""
        try:
            # Parse S3 URI
            s3_path = output_s3_uri.replace("s3://", "").split("/", 1)
            bucket = s3_path[0]
            prefix = s3_path[1] if len(s3_path) > 1 else ""
            
            # List output files
            keys = s3_storage.list_objects(prefix, bucket)
            
            labeled_data = []
            for key in keys:
                if key.endswith(".json"):
                    data = s3_storage.download_data(key, bucket)
                    if data:
                        labeled_data.append(data)
            
            logger.info(f"Downloaded {len(labeled_data)} labeled records")
            return labeled_data
            
        except Exception as e:
            logger.error(f"Error downloading labeled data: {e}")
            return []


# Global labeling pipeline instance
labeling_pipeline = LabelingPipeline()

