"""Model monitoring using SageMaker Model Monitor."""

from typing import Dict, Any, Optional
from config.aws_config import aws_config
from config.pipeline_config import pipeline_config
from src.utils.logger import logger


class ModelMonitor:
    """Monitors model performance and data quality."""
    
    def __init__(self):
        self.sagemaker = aws_config.get_sagemaker_client()
    
    def create_monitoring_schedule(
        self,
        endpoint_name: str,
        baseline_s3_uri: str,
        output_s3_uri: str,
        schedule_name: str = None
    ) -> str:
        """
        Create SageMaker Model Monitor schedule.
        
        Args:
            endpoint_name: SageMaker endpoint name
            baseline_s3_uri: S3 URI of baseline statistics
            output_s3_uri: S3 URI for monitoring output
            schedule_name: Optional schedule name
            
        Returns:
            Monitoring schedule name
        """
        schedule_name = schedule_name or f"{endpoint_name}-monitor"
        
        try:
            from sagemaker.model_monitor import DataQualityMonitor
            
            # Create data quality monitor
            monitor = DataQualityMonitor(
                role=pipeline_config.SAGEMAKER_ROLE_ARN,
                instance_count=1,
                instance_type="ml.t3.medium",
                volume_size_in_gb=20,
                max_runtime_in_seconds=3600,
                sagemaker_session=None  # Would create session
            )
            
            # Create schedule
            monitor.create_monitoring_schedule(
                monitor_schedule_name=schedule_name,
                endpoint_input=endpoint_name,
                output_s3_uri=output_s3_uri,
                statistics=baseline_s3_uri,
                constraints=None,
                schedule_cron_expression=pipeline_config.MONITORING_SCHEDULE
            )
            
            logger.info(f"Monitoring schedule created: {schedule_name}")
            return schedule_name
            
        except Exception as e:
            logger.error(f"Error creating monitoring schedule: {e}")
            raise
    
    def check_violations(
        self,
        schedule_name: str
    ) -> Dict[str, Any]:
        """
        Check for monitoring violations.
        
        Args:
            schedule_name: Monitoring schedule name
            
        Returns:
            Violation information
        """
        try:
            violations = self.sagemaker.list_monitoring_executions(
                MonitoringScheduleName=schedule_name
            )
            
            return {
                "schedule_name": schedule_name,
                "executions": violations.get("MonitoringExecutionSummaries", [])
            }
            
        except Exception as e:
            logger.error(f"Error checking violations: {e}")
            return {}


# Global model monitor instance
model_monitor = ModelMonitor()

