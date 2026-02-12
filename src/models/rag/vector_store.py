"""Vector store management for RAG using OpenSearch."""

import json
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from config.aws_config import aws_config
from config.model_config import model_config
from src.utils.logger import logger


class VectorStore:
    """Manages vector store for RAG using OpenSearch."""
    
    def __init__(self):
        self.opensearch = aws_config.get_opensearch_client()
        self.index_name = model_config.OPENSEARCH_INDEX_NAME
        self.embedding_model = SentenceTransformer(model_config.EMBEDDING_MODEL)
        self.vector_dimension = model_config.VECTOR_DIMENSION
    
    def create_index(self):
        """Create OpenSearch index for vector storage."""
        try:
            # Check if index exists
            if self.opensearch.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return
            
            # Index mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "metadata": {"type": "object"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.vector_dimension
                        }
                    }
                }
            }
            
            self.opensearch.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Index {self.index_name} created")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ):
        """
        Add documents to vector store.
        
        Args:
            documents: List of documents with 'text' and optional 'metadata'
        """
        try:
            for i, doc in enumerate(documents):
                text = doc.get("text", doc.get("content", ""))
                metadata = doc.get("metadata", {})
                
                # Generate embedding
                embedding = self.embedding_model.encode(text).tolist()
                
                # Index document
                self.opensearch.index(
                    index=self.index_name,
                    id=i,
                    body={
                        "text": text,
                        "metadata": metadata,
                        "embedding": embedding
                    }
                )
            
            # Refresh index
            self.opensearch.indices.refresh(index=self.index_name)
            logger.info(f"Indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of relevant documents
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build search query
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                }
            }
            
            # Add filters if provided
            if filters:
                search_body["query"] = {
                    "bool": {
                        "must": [search_body["query"]],
                        "filter": [
                            {"term": {f"metadata.{k}": v}}
                            for k, v in filters.items()
                        ]
                    }
                }
            
            # Search
            response = self.opensearch.search(
                index=self.index_name,
                body=search_body
            )
            
            # Format results
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "text": hit["_source"]["text"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []


# Global vector store instance
vector_store = VectorStore()

