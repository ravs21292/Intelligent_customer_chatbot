"""Complete RAG pipeline combining retrieval and generation."""

from typing import Dict, Any, Optional, List
from src.models.rag.retriever import document_retriever
from src.models.bedrock_client import bedrock_client
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


class RAGPipeline:
    """Complete RAG pipeline for knowledge-augmented generation."""
    
    def __init__(self):
        self.retriever = document_retriever
        self.llm = bedrock_client
    
    @metrics_collector.track_latency("rag_inference")
    def generate_response(
        self,
        query: str,
        intent: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using RAG.
        
        Args:
            query: User query
            intent: Optional intent classification
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response with sources
        """
        try:
            # Retrieve relevant documents
            documents = self.retriever.retrieve(query, intent=intent)
            
            if not documents:
                logger.warning("No relevant documents found for RAG")
                # Fallback to standard generation
                return self.llm.generate_customer_support_response(
                    query,
                    intent or "general_inquiry",
                    conversation_history
                )
            
            # Format context
            context = self.retriever.format_context(documents)
            
            # Build prompt with context
            prompt = f"""Based on the following knowledge base documents, answer the user's question.

Knowledge Base:
{context}

User Question: {query}

Provide a helpful answer based on the knowledge base. If the answer is not in the knowledge base, say so clearly."""

            # Generate response
            response = self.llm.generate_response(prompt)
            
            # Add sources
            response["sources"] = [
                {
                    "text": doc["text"][:200],  # Truncate for display
                    "metadata": doc.get("metadata", {}),
                    "relevance_score": doc.get("score", 0.0)
                }
                for doc in documents
            ]
            response["strategy"] = "rag"
            
            metrics_collector.put_metric(
                "rag_requests",
                1,
                dimensions={"intent": intent or "unknown"}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            # Fallback
            return self.llm.generate_customer_support_response(
                query,
                intent or "general_inquiry",
                conversation_history
            )


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()

