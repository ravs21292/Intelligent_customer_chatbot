"""Helper utility functions."""

import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime


def generate_conversation_id(user_id: str, timestamp: Optional[datetime] = None) -> str:
    """Generate a unique conversation ID."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    data = f"{user_id}_{timestamp.isoformat()}"
    return hashlib.md5(data.encode()).hexdigest()


def validate_message(message: str, max_length: int = 2000) -> bool:
    """Validate user message."""
    if not message or not isinstance(message, str):
        return False
    if len(message.strip()) == 0:
        return False
    if len(message) > max_length:
        return False
    return True


def format_response(response: str, sources: Optional[list] = None) -> Dict[str, Any]:
    """Format chatbot response with metadata."""
    formatted = {
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
        "sources": sources or []
    }
    return formatted


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (can be enhanced with embeddings)."""
    # Simple word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON string."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

