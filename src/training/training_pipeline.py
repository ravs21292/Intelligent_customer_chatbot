"""Main training pipeline orchestrator."""

import os
from typing import Dict, Any, Optional
from config.pipeline_config import pipeline_config
from src.intent_classification.model_training import intent_trainer
from src.models.fine_tuning.lora_trainer import lora_trainer
from src.data_collection.s3_storage import s3_storage
from src.utils.logger import logger


class TrainingPipeline:
    """Orchestrates model training pipelines."""
    
    def run_intent_classification_training(
        self,
        data_path: str,
        output_dir: str = "models/intent_classifier",
        use_sagemaker: bool = False
    ) -> str:
        """
        Run intent classification training.
        
        Args:
            data_path: Path to training data
            output_dir: Output directory
            use_sagemaker: Whether to use SageMaker
            
        Returns:
            Path to trained model
        """
        logger.info("Starting intent classification training pipeline...")
        
        # Prepare data
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            intent_trainer.prepare_data(data_path)
        
        if use_sagemaker:
            # Upload data to S3
            train_s3_uri = s3_storage.upload_training_dataset(
                [{"text": t, "label": model_config.INTENT_CLASSES[l]} 
                 for t, l in zip(train_texts, train_labels)],
                "intent_train",
                "1"
            )
            val_s3_uri = s3_storage.upload_training_dataset(
                [{"text": t, "label": model_config.INTENT_CLASSES[l]} 
                 for t, l in zip(val_texts, val_labels)],
                "intent_val",
                "1"
            )
            
            # Train on SageMaker
            model_uri = intent_trainer.train_sagemaker(
                train_s3_uri,
                val_s3_uri,
                f"intent-classifier-{int(time.time())}"
            )
            return model_uri
        else:
            # Train locally
            model_path = intent_trainer.train_local(
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                output_dir
            )
            
            # Evaluate
            metrics = intent_trainer.evaluate(model_path, test_texts, test_labels)
            logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.4f}")
            
            return model_path
    
    def run_fine_tuning_pipeline(
        self,
        domain: str,
        data_path: str,
        output_dir: str = None
    ) -> str:
        """
        Run fine-tuning pipeline for a domain.
        
        Args:
            domain: Domain name
            data_path: Path to training data
            output_dir: Output directory
            
        Returns:
            Path to fine-tuned model
        """
        logger.info(f"Starting fine-tuning pipeline for domain: {domain}")
        
        from src.models.fine_tuning.data_preparation import data_preparation
        
        # Prepare data
        training_data = data_preparation.prepare_domain_specific_data(domain, data_path)
        
        # Create dataset
        from datasets import Dataset
        dataset = lora_trainer.prepare_dataset(training_data)
        
        # Split
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        
        # Train
        output_dir = output_dir or f"models/fine_tuned/{domain}"
        model_path = lora_trainer.train(
            train_dataset,
            val_dataset,
            output_dir
        )
        
        logger.info(f"Fine-tuning completed for {domain}")
        return model_path


# Import time for timestamps
import time
from config.model_config import model_config

# Global training pipeline instance
training_pipeline = TrainingPipeline()

