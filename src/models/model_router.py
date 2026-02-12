"""Main model router that orchestrates multi-model strategy."""

from typing import Dict, Any, Optional, List
from src.models.bedrock_client import bedrock_client
from src.models.rag.rag_pipeline import rag_pipeline
from src.intent_classification.router import model_router
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


class MultiModelRouter:
    """Routes queries to appropriate model strategy and aggregates responses."""
    
    def __init__(self):
        self.router = model_router
        self.bedrock = bedrock_client
        self.rag = rag_pipeline
        # Fine-tuned models would be loaded here
        self.fine_tuned_models = {}
    
    @metrics_collector.track_latency("model_routing")
    def generate_response(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using appropriate model strategy.
        
        Args:
            message: User message
            conversation_history: Previous conversation messages
            user_id: Optional user identifier
            
        Returns:
            Generated response with metadata
        """
        try:
            # Classify intent and get routing decision
            routing_decision = self.router.route(message)
            
            intent = routing_decision["intent"]
            strategy = routing_decision["strategy"]
            
            # Route to appropriate model
            if strategy == "pre_trained":
                response = self.bedrock.generate_customer_support_response(
                    message,
                    intent,
                    conversation_history
                )
            
            elif strategy == "rag":
                response = self.rag.generate_response(
                    message,
                    intent=intent,
                    conversation_history=conversation_history
                )
            
            elif strategy == "fine_tuned":
                # Use fine-tuned model (placeholder - would load actual model)
                model_name = routing_decision.get("model_name")
                if model_name in self.fine_tuned_models:
                    # Generate using fine-tuned model
                    response = self._generate_with_fine_tuned(
                        model_name,
                        message,
                        intent
                    )
                else:
                    # Fallback to Bedrock
                    logger.warning(f"Fine-tuned model {model_name} not available, using Bedrock")
                    response = self.bedrock.generate_customer_support_response(
                        message,
                        intent,
                        conversation_history
                    )
            
            else:
                # Default fallback
                response = self.bedrock.generate_customer_support_response(
                    message,
                    intent,
                    conversation_history
                )
            
            # Add routing metadata
            response["routing"] = routing_decision
            response["intent"] = intent
            
            # Check if escalation needed
            should_escalate = self.router.should_escalate(
                intent,
                routing_decision["confidence"]
            )
            response["escalate"] = should_escalate
            
            if should_escalate:
                response["escalation_message"] = "I'll connect you with a human agent who can better assist you."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in model routing: {e}")
            # Fallback response
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact support.",
                "error": str(e),
                "strategy": "fallback"
            }
    
    def _generate_with_fine_tuned(
        self,
        model_name: str,
        message: str,
        intent: str
    ) -> Dict[str, Any]:
        """Generate response using fine-tuned model."""
        # Placeholder - would implement actual fine-tuned model inference
        return {
            "response": f"[Fine-tuned model response for {intent}]",
            "model": model_name,
            "strategy": "fine_tuned"
        }
    
    def load_fine_tuned_model(self, model_name: str, model_path: str):
        """Load a fine-tuned model."""
        # Placeholder - would load actual model
        self.fine_tuned_models[model_name] = model_path
        logger.info(f"Loaded fine-tuned model: {model_name}")


# Global multi-model router instance
multi_model_router = MultiModelRouter()

