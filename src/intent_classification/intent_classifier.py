"""Intent classification inference."""

import os
import json
import torch
from typing import Dict, Any, List, Optional
from transformers import BertTokenizer, BertForSequenceClassification
from config.model_config import model_config
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


class IntentClassifier:
    """Performs intent classification on user messages."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/intent_classifier"
        self.max_length = model_config.MAX_SEQUENCE_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
        
        # Load label mappings
        self._load_label_mappings()
    
    def _load_model(self):
        """Load the intent classification model."""
        try:
            if os.path.exists(self.model_path):
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model = BertForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Intent classifier loaded from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}, using default model")
                self.tokenizer = BertTokenizer.from_pretrained(model_config.INTENT_MODEL_NAME)
                self.model = BertForSequenceClassification.from_pretrained(
                    model_config.INTENT_MODEL_NAME,
                    num_labels=len(model_config.INTENT_CLASSES)
                )
                self.model.to(self.device)
                self.model.eval()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_label_mappings(self):
        """Load label to ID mappings."""
        mapping_file = os.path.join(self.model_path, "label_mappings.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                mappings = json.load(f)
                self.label_to_id = mappings["label_to_id"]
                self.id_to_label = mappings["id_to_label"]
        else:
            self.label_to_id = {label: idx for idx, label in enumerate(model_config.INTENT_CLASSES)}
            self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
    
    @metrics_collector.track_latency("intent_classification")
    def classify(
        self,
        text: str,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Classify intent of a text message.
        
        Args:
            text: Input text message
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with intent and optional confidence scores
        """
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_id].item()
            
            intent = self.id_to_label[predicted_id]
            
            result = {
                "intent": intent,
                "confidence": confidence
            }
            
            if return_confidence:
                # Get top 3 intents
                top_probs, top_indices = torch.topk(probabilities[0], k=min(3, len(model_config.INTENT_CLASSES)))
                result["top_intents"] = [
                    {
                        "intent": self.id_to_label[idx.item()],
                        "confidence": prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify intents for multiple texts.
        
        Args:
            texts: List of text messages
            
        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            result = self.classify(text)
            results.append(result)
        return results


# Global classifier instance
intent_classifier = IntentClassifier()

