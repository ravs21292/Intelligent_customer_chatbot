"""Incremental learning and model updates."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from config.pipeline_config import pipeline_config
from src.data_collection.s3_storage import s3_storage
from src.training.training_pipeline import training_pipeline
from src.utils.logger import logger


class IncrementalLearning:
    """Manages incremental learning from new data."""
    
    def collect_feedback_data(
        self,
        conversation_id: str,
        user_feedback: str,
        correct_intent: Optional[str] = None
    ) -> bool:
        """
        Collect user feedback for incremental learning.
        
        Args:
            conversation_id: Conversation identifier
            user_feedback: User feedback (thumbs up/down)
            correct_intent: Optional correct intent if misclassified
            
        Returns:
            True if successful
        """
        try:
            feedback_record = {
                "conversation_id": conversation_id,
                "feedback": user_feedback,
                "correct_intent": correct_intent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store feedback
            feedback_key = f"feedback/{datetime.utcnow().strftime('%Y/%m/%d')}/{conversation_id}.json"
            s3_storage.upload_data(feedback_record, feedback_key)
            
            logger.info(f"Feedback collected for conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return False
    
    def check_retraining_conditions(self) -> bool:
        """
        Check if retraining conditions are met.
        
        Returns:
            True if retraining should be triggered
        """
        # Check for new labeled data
        new_data_count = self._count_new_labeled_data()
        if new_data_count >= pipeline_config.MIN_NEW_SAMPLES_FOR_RETRAIN:
            logger.info(f"Retraining condition met: {new_data_count} new samples")
            return True
        
        # Check model performance degradation
        if self._check_performance_degradation():
            logger.info("Retraining condition met: performance degradation detected")
            return True
        
        return False
    
    def _count_new_labeled_data(self) -> int:
        """Count new labeled data samples."""
        # List new labeled data files
        labeled_prefix = "labeled/datasets/"
        keys = s3_storage.list_objects(labeled_prefix)
        return len(keys)
    
    def _check_performance_degradation(self) -> bool:
        """Check if model performance has degraded."""
        # Placeholder - would check actual metrics
        return False
    
    def trigger_retraining(
        self,
        model_type: str = "intent_classifier"
    ) -> str:
        """
        Trigger model retraining.
        
        Args:
            model_type: Type of model to retrain
            
        Returns:
            Training job ID or model path
        """
        logger.info(f"Triggering retraining for {model_type}")
        
        # Get latest training data
        data_path = self._get_latest_training_data()
        
        if model_type == "intent_classifier":
            model_path = training_pipeline.run_intent_classification_training(
                data_path,
                use_sagemaker=True
            )
        else:
            model_path = training_pipeline.run_fine_tuning_pipeline(
                model_type,
                data_path
            )
        
        logger.info(f"Retraining completed: {model_path}")
        return model_path
    
    def _get_latest_training_data(self) -> str:
        """Get path to latest training data."""
        # List training datasets
        training_prefix = pipeline_config.S3_TRAINING_PATH
        keys = s3_storage.list_objects(training_prefix)
        
        if keys:
            # Return most recent
            return f"s3://{pipeline_config.S3_BUCKET_DATA}/{sorted(keys)[-1]}"
        else:
            raise ValueError("No training data found")


# Global incremental learning instance
incremental_learning = IncrementalLearning()

