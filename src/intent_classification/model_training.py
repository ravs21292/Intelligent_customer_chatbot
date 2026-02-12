"""Intent classification model training using BERT and SageMaker."""

import os
import json
import pickle
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sagemaker.huggingface import HuggingFace
from config.model_config import model_config
from config.pipeline_config import pipeline_config
from src.data_collection.s3_storage import s3_storage
from src.utils.logger import logger


class IntentDataset(Dataset):
    """PyTorch dataset for intent classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class IntentModelTrainer:
    """Trains intent classification models."""
    
    def __init__(self):
        self.model_name = model_config.INTENT_MODEL_NAME
        self.num_labels = len(model_config.INTENT_CLASSES)
        self.max_length = model_config.MAX_SEQUENCE_LENGTH
        self.batch_size = model_config.BATCH_SIZE
        self.label_to_id = {label: idx for idx, label in enumerate(model_config.INTENT_CLASSES)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
    def prepare_data(
        self,
        data_path: str,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
        """
        Prepare training data from labeled dataset.
        
        Args:
            data_path: Path to labeled data file
            test_size: Proportion of test set
            val_size: Proportion of validation set (from training set)
            
        Returns:
            Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
        """
        # Load labeled data
        if data_path.startswith("s3://"):
            data = s3_storage.download_data(data_path.replace("s3://", "").split("/", 1)[1])
        else:
            with open(data_path, "r") as f:
                data = json.load(f)
        
        texts = []
        labels = []
        
        for record in data:
            text = record.get("text", record.get("message", ""))
            label = record.get("label", record.get("intent", ""))
            
            if text and label in self.label_to_id:
                texts.append(text)
                labels.append(self.label_to_id[label])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_local(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: str = "models/intent_classifier",
        epochs: int = 3
    ) -> str:
        """
        Train model locally.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Output directory for model
            epochs: Number of training epochs
            
        Returns:
            Path to saved model
        """
        logger.info("Starting local model training...")
        
        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Create datasets
        train_dataset = IntentDataset(train_texts, train_labels, tokenizer, self.max_length)
        val_dataset = IntentDataset(val_texts, val_labels, tokenizer, self.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", "w") as f:
            json.dump({
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label
            }, f, indent=2)
        
        logger.info(f"Model trained and saved to {output_dir}")
        return output_dir
    
    def train_sagemaker(
        self,
        train_data_s3_uri: str,
        val_data_s3_uri: str,
        job_name: str,
        instance_type: str = "ml.g4dn.xlarge"
    ) -> str:
        """
        Train model on SageMaker.
        
        Args:
            train_data_s3_uri: S3 URI of training data
            val_data_s3_uri: S3 URI of validation data
            job_name: SageMaker training job name
            instance_type: SageMaker instance type
            
        Returns:
            Model artifact S3 URI
        """
        logger.info("Starting SageMaker training job...")
        
        # Create training script
        training_script = self._create_training_script()
        
        # HuggingFace estimator
        estimator = HuggingFace(
            entry_point="train.py",
            source_dir="src/intent_classification",
            instance_type=instance_type,
            instance_count=1,
            role=pipeline_config.SAGEMAKER_ROLE_ARN,
            transformers_version="4.26",
            pytorch_version="1.13",
            py_version="py39",
            hyperparameters={
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "epochs": 3,
                "batch_size": self.batch_size,
                "max_length": self.max_length
            }
        )
        
        # Start training
        estimator.fit({
            "training": train_data_s3_uri,
            "validation": val_data_s3_uri
        }, job_name=job_name)
        
        model_uri = estimator.model_data
        logger.info(f"Training completed. Model URI: {model_uri}")
        return model_uri
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    def _create_training_script(self) -> str:
        """Create training script for SageMaker."""
        script_content = """
import os
import json
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

def main():
    # Load hyperparameters
    model_name = os.environ.get("SM_HP_MODEL_NAME", "bert-base-uncased")
    num_labels = int(os.environ.get("SM_HP_NUM_LABELS", 8))
    epochs = int(os.environ.get("SM_HP_EPOCHS", 3))
    batch_size = int(os.environ.get("SM_HP_BATCH_SIZE", 32))
    
    # Load data
    train_data = load_dataset("json", data_files=os.environ["SM_CHANNEL_TRAINING"])
    val_data = load_dataset("json", data_files=os.environ["SM_CHANNEL_VALIDATION"])
    
    # Initialize model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Training
    training_args = TrainingArguments(
        output_dir=os.environ["SM_MODEL_DIR"],
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data
    )
    
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
"""
        script_path = "src/intent_classification/train.py"
        with open(script_path, "w") as f:
            f.write(script_content)
        return script_path
    
    def evaluate(
        self,
        model_path: str,
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            model_path: Path to trained model
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")
        
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        predictions = []
        with torch.no_grad():
            for text in test_texts:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                outputs = model(**encoding)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
        
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(
            test_labels,
            predictions,
            target_names=model_config.INTENT_CLASSES,
            output_dict=True
        )
        
        metrics = {
            "accuracy": accuracy,
            "classification_report": report
        }
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        return metrics


# Global trainer instance
intent_trainer = IntentModelTrainer()

