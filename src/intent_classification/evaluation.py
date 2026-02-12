"""Model evaluation utilities."""

import json
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from config.model_config import model_config
from src.intent_classification.intent_classifier import IntentClassifier
from src.utils.logger import logger


class ModelEvaluator:
    """Evaluates intent classification models."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.classifier = IntentClassifier(model_path)
    
    def evaluate_on_dataset(
        self,
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data: List of test examples with 'text' and 'label' keys
            
        Returns:
            Evaluation metrics
        """
        texts = [item["text"] for item in test_data]
        true_labels = [item["label"] for item in test_data]
        
        # Predict
        predictions = []
        for text in texts:
            result = self.classifier.classify(text, return_confidence=False)
            predictions.append(result["intent"])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=model_config.INTENT_CLASSES,
            average=None
        )
        
        # Per-class metrics
        per_class_metrics = {
            label: {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1_score),
                "support": int(supp)
            }
            for label, prec, rec, f1_score, supp in zip(
                model_config.INTENT_CLASSES, precision, recall, f1, support
            )
        }
        
        # Overall metrics
        macro_avg = precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro"
        )
        weighted_avg = precision_recall_fscore_support(
            true_labels,
            predictions,
            average="weighted"
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "per_class": per_class_metrics,
            "macro_avg": {
                "precision": float(macro_avg[0]),
                "recall": float(macro_avg[1]),
                "f1": float(macro_avg[2])
            },
            "weighted_avg": {
                "precision": float(weighted_avg[0]),
                "recall": float(weighted_avg[1]),
                "f1": float(weighted_avg[2])
            },
            "confusion_matrix": confusion_matrix(
                true_labels,
                predictions,
                labels=model_config.INTENT_CLASSES
            ).tolist()
        }
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: List[List[int]],
        output_path: str = "confusion_matrix.png"
    ):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model_config.INTENT_CLASSES,
            yticklabels=model_config.INTENT_CLASSES
        )
        plt.title("Intent Classification Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def compare_models(
        self,
        model_paths: List[str],
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.
        
        Args:
            model_paths: List of model paths to compare
            test_data: Test dataset
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for model_path in model_paths:
            evaluator = ModelEvaluator(model_path)
            metrics = evaluator.evaluate_on_dataset(test_data)
            comparison[model_path] = metrics
        
        # Find best model
        best_model = max(
            comparison.items(),
            key=lambda x: x[1]["accuracy"]
        )
        
        logger.info(f"Best model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
        return {
            "comparison": comparison,
            "best_model": {
                "path": best_model[0],
                "accuracy": best_model[1]["accuracy"]
            }
        }


# Note: Added Optional import for type hints
from typing import Optional

