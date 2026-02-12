"""Automated retraining triggers using EventBridge and Lambda."""

import json
from typing import Dict, Any
from datetime import datetime
from config.aws_config import aws_config
from config.pipeline_config import pipeline_config
from src.training.incremental_learning import incremental_learning
from src.utils.logger import logger


class RetrainingTrigger:
    """Manages automated retraining triggers."""
    
    def __init__(self):
        self.eventbridge = aws_config.get_eventbridge_client()
        self.lambda_client = aws_config.get_lambda_client()
    
    def create_scheduled_retraining(
        self,
        schedule_expression: str = None,
        lambda_function_name: str = "retraining-trigger"
    ) -> str:
        """
        Create scheduled retraining rule.
        
        Args:
            schedule_expression: Cron expression (defaults to weekly)
            lambda_function_name: Lambda function to trigger
            
        Returns:
            Rule ARN
        """
        schedule_expression = schedule_expression or pipeline_config.RETRAINING_SCHEDULE
        
        rule_name = "customer-chatbot-retraining-schedule"
        
        try:
            # Create or update rule
            response = self.eventbridge.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule_expression,
                State="ENABLED",
                Description="Scheduled retraining for customer chatbot models"
            )
            
            rule_arn = response["RuleArn"]
            
            # Add Lambda target
            self.eventbridge.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        "Id": "1",
                        "Arn": f"arn:aws:lambda:{aws_config.region}:*:function:{lambda_function_name}"
                    }
                ]
            )
            
            logger.info(f"Scheduled retraining rule created: {rule_arn}")
            return rule_arn
            
        except Exception as e:
            logger.error(f"Error creating scheduled retraining: {e}")
            raise
    
    def trigger_on_data_threshold(
        self,
        threshold: int = None
    ):
        """
        Trigger retraining when data threshold is met.
        
        Args:
            threshold: Number of new samples required
        """
        threshold = threshold or pipeline_config.MIN_NEW_SAMPLES_FOR_RETRAIN
        
        if incremental_learning.check_retraining_conditions():
            incremental_learning.trigger_retraining()
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Lambda handler for retraining trigger.
        
        Args:
            event: Lambda event
            context: Lambda context
            
        Returns:
            Response dictionary
        """
        try:
            logger.info("Retraining trigger invoked")
            
            # Check conditions
            if incremental_learning.check_retraining_conditions():
                # Trigger retraining
                model_path = incremental_learning.trigger_retraining()
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "message": "Retraining triggered",
                        "model_path": model_path
                    })
                }
            else:
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "message": "Retraining conditions not met"
                    })
                }
                
        except Exception as e:
            logger.error(f"Error in retraining trigger: {e}")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)})
            }


# Global retraining trigger instance
retraining_trigger = RetrainingTrigger()

