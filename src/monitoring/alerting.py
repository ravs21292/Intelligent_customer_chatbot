"""Alerting system for model and system issues."""

from typing import Dict, Any, List
from config.aws_config import aws_config
from src.utils.logger import logger


class AlertManager:
    """Manages alerts for model and system issues."""
    
    def __init__(self):
        self.sns = boto3.client("sns", region_name=aws_config.region)
        self.alert_topic_arn = None  # Would be configured
    
    def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "medium"
    ):
        """
        Send alert notification.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (low, medium, high, critical)
        """
        try:
            alert = {
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if self.alert_topic_arn:
                self.sns.publish(
                    TopicArn=self.alert_topic_arn,
                    Message=json.dumps(alert),
                    Subject=f"Chatbot Alert: {alert_type}"
                )
            
            logger.warning(f"Alert sent: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def check_model_health(
        self,
        model_name: str,
        metrics: Dict[str, float]
    ):
        """Check model health and send alerts if needed."""
        # Check accuracy
        if metrics.get("accuracy", 1.0) < 0.7:
            self.send_alert(
                "model_performance",
                f"Model {model_name} accuracy below threshold: {metrics['accuracy']}",
                severity="high"
            )
        
        # Check latency
        if metrics.get("latency", 0.0) > 5.0:
            self.send_alert(
                "model_latency",
                f"Model {model_name} latency high: {metrics['latency']}s",
                severity="medium"
            )


# Import required modules
import boto3
import json
from datetime import datetime

# Global alert manager
alert_manager = AlertManager()

