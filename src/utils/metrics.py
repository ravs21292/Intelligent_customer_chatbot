"""Metrics collection and tracking utilities."""

import time
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime
import boto3
from config.aws_config import aws_config
from src.utils.logger import logger


class MetricsCollector:
    """Collects and sends metrics to CloudWatch."""
    
    def __init__(self):
        self.cloudwatch = aws_config.get_cloudwatch_client()
        self.namespace = "CustomerChatbot"
        self.metrics_buffer = []
    
    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "Count",
        dimensions: Optional[Dict[str, str]] = None
    ):
        """
        Put a metric to CloudWatch.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            dimensions: Optional dimensions for the metric
        """
        try:
            metric_data = {
                "MetricName": metric_name,
                "Value": value,
                "Unit": unit,
                "Timestamp": datetime.utcnow()
            }
            
            if dimensions:
                metric_data["Dimensions"] = [
                    {"Name": k, "Value": v} for k, v in dimensions.items()
                ]
            
            self.metrics_buffer.append(metric_data)
            
            # Flush buffer if it reaches 20 items (CloudWatch limit)
            if len(self.metrics_buffer) >= 20:
                self.flush_metrics()
                
        except Exception as e:
            logger.error(f"Error putting metric: {e}")
    
    def flush_metrics(self):
        """Flush buffered metrics to CloudWatch."""
        if not self.metrics_buffer:
            return
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=self.metrics_buffer
            )
            self.metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
    
    def track_latency(self, operation: str):
        """Decorator to track operation latency."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    latency = time.time() - start_time
                    self.put_metric(
                        f"{operation}_latency",
                        latency,
                        unit="Seconds",
                        dimensions={"operation": operation}
                    )
                    return result
                except Exception as e:
                    self.put_metric(
                        f"{operation}_error",
                        1,
                        dimensions={"operation": operation, "error_type": type(e).__name__}
                    )
                    raise
            return wrapper
        return decorator
    
    def track_model_performance(
        self,
        model_name: str,
        accuracy: float,
        latency: float,
        cost: Optional[float] = None
    ):
        """Track model performance metrics."""
        self.put_metric(
            "model_accuracy",
            accuracy,
            unit="Percent",
            dimensions={"model": model_name}
        )
        self.put_metric(
            "model_latency",
            latency,
            unit="Seconds",
            dimensions={"model": model_name}
        )
        if cost:
            self.put_metric(
                "model_cost",
                cost,
                unit="None",
                dimensions={"model": model_name}
            )


# Global metrics collector
metrics_collector = MetricsCollector()

