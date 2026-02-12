"""Model configuration settings."""

import os
from typing import Dict, List


class ModelConfig:
    """Configuration for ML models."""
    
    # Intent Classification
    INTENT_CLASSES = [
        "billing",
        "technical_support",
        "product_inquiry",
        "complaint",
        "refund",
        "general_inquiry",
        "account_management",
        "escalation"
    ]
    
    INTENT_MODEL_NAME = "bert-base-uncased"
    MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    
    # Bedrock Configuration
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-v2")
    BEDROCK_MAX_TOKENS = 2048
    BEDROCK_TEMPERATURE = 0.7
    
    # Fine-tuning Configuration
    FINE_TUNE_BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TRAINING_EPOCHS = 3
    LEARNING_RATE = 2e-4
    
    # RAG Configuration
    OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "customer-support-kb")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIMENSION = 384
    TOP_K_RETRIEVAL = 5
    
    # Model Routing Thresholds
    ROUTING_THRESHOLDS = {
        "simple_query_confidence": 0.8,
        "domain_specific_confidence": 0.7,
        "rag_required_confidence": 0.6
    }
    
    # Response Quality Scoring
    QUALITY_THRESHOLD = 0.7
    MIN_RESPONSE_LENGTH = 20
    MAX_RESPONSE_LENGTH = 500


class DomainConfig:
    """Configuration for domain-specific models."""
    
    DOMAINS = {
        "billing": {
            "model_name": "billing-support-model",
            "fine_tuned": True,
            "use_rag": False
        },
        "technical_support": {
            "model_name": "technical-support-model",
            "fine_tuned": True,
            "use_rag": True
        },
        "product_inquiry": {
            "model_name": "product-inquiry-model",
            "fine_tuned": False,
            "use_rag": True
        },
        "complaint": {
            "model_name": "complaint-handling-model",
            "fine_tuned": True,
            "use_rag": False
        }
    }


# Global model configuration instance
model_config = ModelConfig()
domain_config = DomainConfig()

