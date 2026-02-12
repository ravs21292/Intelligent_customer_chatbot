"""Document retrieval for RAG."""

from typing import List, Dict, Any, Optional
from src.models.rag.vector_store import vector_store
from config.model_config import model_config
from src.utils.logger import logger


class DocumentRetriever:
    """Retrieves relevant documents for RAG."""
    
    def __init__(self):
        self.vector_store = vector_store
        self.top_k = model_config.TOP_K_RETRIEVAL
    
    def retrieve(
        self,
        query: str,
        intent: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            intent: Optional intent classification
            filters: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        # Add intent to filters if provided
        if intent:
            if filters is None:
                filters = {}
            filters["intent"] = intent
        
        # Retrieve from vector store
        documents = self.vector_store.search(
            query,
            top_k=self.top_k,
            filters=filters
        )
        
        logger.debug(f"Retrieved {len(documents)} documents for query")
        return documents
    
    def format_context(
        self,
        documents: List[Dict[str, Any]],
        max_length: int = 1000
    ) -> str:
        """
        Format retrieved documents as context.
        
        Args:
            documents: Retrieved documents
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for doc in documents:
            text = doc["text"]
            if current_length + len(text) > max_length:
                break
            
            context_parts.append(f"Document: {text}")
            current_length += len(text)
        
        return "\n\n".join(context_parts)


# Global retriever instance
document_retriever = DocumentRetriever()

