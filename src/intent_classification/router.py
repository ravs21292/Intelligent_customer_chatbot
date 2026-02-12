"""Routing logic based on intent classification."""

from typing import Dict, Any, Optional
from enum import Enum
from config.model_config import model_config, domain_config
from src.intent_classification.intent_classifier import intent_classifier
from src.utils.logger import logger


class ModelStrategy(Enum):
    """Available model strategies."""
    PRE_TRAINED = "pre_trained"  # AWS Bedrock
    FINE_TUNED = "fine_tuned"    # Domain-specific fine-tuned model
    RAG = "rag"                  # RAG with knowledge base


class ModelRouter:
    """Routes queries to appropriate model strategy based on intent."""
    
    def __init__(self):
        self.thresholds = model_config.ROUTING_THRESHOLDS
    
    def route(
        self,
        message: str,
        intent_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a message to the appropriate model strategy.
        
        Args:
            message: User message
            intent_result: Optional pre-computed intent classification result
            
        Returns:
            Routing decision with strategy and metadata
        """
        # Classify intent if not provided
        if intent_result is None:
            intent_result = intent_classifier.classify(message)
        
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        
        # Get domain configuration
        domain_info = domain_config.DOMAINS.get(intent, {})
        
        # Routing logic
        routing_decision = {
            "intent": intent,
            "confidence": confidence,
            "strategy": None,
            "model_name": None,
            "use_rag": False,
            "reasoning": ""
        }
        
        # Simple queries with high confidence -> Pre-trained model
        if confidence >= self.thresholds["simple_query_confidence"]:
            if intent in ["general_inquiry", "product_inquiry"]:
                routing_decision["strategy"] = ModelStrategy.PRE_TRAINED.value
                routing_decision["model_name"] = "bedrock-claude"
                routing_decision["reasoning"] = "High confidence simple query"
        
        # Domain-specific queries -> Fine-tuned model
        elif domain_info.get("fine_tuned", False):
            routing_decision["strategy"] = ModelStrategy.FINE_TUNED.value
            routing_decision["model_name"] = domain_info["model_name"]
            routing_decision["reasoning"] = f"Domain-specific query for {intent}"
            
            # Check if RAG is also needed
            if domain_info.get("use_rag", False):
                routing_decision["use_rag"] = True
        
        # Complex queries or low confidence -> RAG
        elif confidence < self.thresholds["rag_required_confidence"] or intent in ["technical_support", "complaint"]:
            routing_decision["strategy"] = ModelStrategy.RAG.value
            routing_decision["use_rag"] = True
            routing_decision["reasoning"] = "Complex query requiring knowledge base"
        
        # Default to pre-trained
        else:
            routing_decision["strategy"] = ModelStrategy.PRE_TRAINED.value
            routing_decision["model_name"] = "bedrock-claude"
            routing_decision["reasoning"] = "Default routing to pre-trained model"
        
        logger.debug(f"Routing decision: {routing_decision}")
        return routing_decision
    
    def should_escalate(
        self,
        intent: str,
        confidence: float,
        sentiment: Optional[str] = None
    ) -> bool:
        """
        Determine if query should be escalated to human agent.
        
        Args:
            intent: Classified intent
            confidence: Classification confidence
            sentiment: Optional sentiment analysis result
            
        Returns:
            True if should escalate
        """
        # Low confidence
        if confidence < 0.5:
            return True
        
        # Explicit escalation intent
        if intent == "escalation":
            return True
        
        # Negative sentiment with complaint
        if sentiment == "negative" and intent == "complaint":
            return True
        
        # Billing issues with refund intent
        if intent == "refund" and confidence > 0.8:
            return True
        
        return False


# Global router instance
model_router = ModelRouter()

