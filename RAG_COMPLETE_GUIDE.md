# Complete RAG System Guide

## Overview: What is RAG and Why It's Used

**RAG (Retrieval-Augmented Generation)** combines information retrieval with language model generation to provide accurate, source-cited responses based on a knowledge base.

### Why RAG in This Project?

1. **Pre-trained models don't know YOUR company's information** - They have general knowledge but not your specific products, policies, or documentation
2. **Fine-tuned models can't cite sources** - They learn patterns but can't reference specific documents
3. **Need for up-to-date information** - Knowledge bases change frequently (new products, policy updates)
4. **Source attribution** - Users need to verify information comes from trusted sources

---

## RAG Architecture and Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG System Components                      │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Vector Store    │      │    Retriever     │      │   RAG Pipeline   │
│  (OpenSearch)    │◄─────│  (Document       │◄─────│  (Orchestrator)  │
│                  │      │   Retrieval)     │      │                  │
│ - Embeddings     │      │                  │      │ - Retrieval      │
│ - Metadata       │      │ - Query          │      │ - Augmentation   │
│ - Text           │      │ - Filtering      │      │ - Generation     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         ▲                         ▲                          │
         │                         │                          │
         │                         │                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Embedding Model (Sentence Transformers)         │
│              - Converts text to vectors (384 dimensions)     │
└─────────────────────────────────────────────────────────────┘
         │                         │                          │
         │                         │                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Generation Model (AWS Bedrock/Claude)            │
│              - Generates response from retrieved context     │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. **Vector Store** (`src/models/rag/vector_store.py`)
- **Purpose**: Stores document embeddings in OpenSearch
- **Technology**: OpenSearch with KNN vector search
- **Storage Format**: 
  - `text`: Original document text
  - `metadata`: Document metadata (source, domain, date, etc.)
  - `embedding`: 384-dimensional vector representation

#### 2. **Document Retriever** (`src/models/rag/retriever.py`)
- **Purpose**: Retrieves relevant documents for a query
- **Features**:
  - Semantic similarity search
  - Metadata filtering (by intent, domain, etc.)
  - Top-K retrieval (default: 5 documents)

#### 3. **RAG Pipeline** (`src/models/rag/rag_pipeline.py`)
- **Purpose**: Orchestrates retrieval and generation
- **Steps**:
  1. Retrieve relevant documents
  2. Format context from documents
  3. Build augmented prompt
  4. Generate response with LLM
  5. Attach source citations

---

## Data Management in RAG

### Data Structure

Each document in the vector store has this structure:

```python
{
    "text": "To reset your password, go to Settings > Security > Reset Password...",
    "metadata": {
        "source": "user_guide.pdf",
        "page": 12,
        "domain": "technical_support",
        "intent": "technical_support",
        "last_updated": "2024-01-15T10:30:00Z",
        "version": "1.2"
    },
    "embedding": [0.123, -0.456, 0.789, ...]  # 384-dimensional vector
}
```

### Data Storage Details

**File**: `src/models/rag/vector_store.py`

```python
class VectorStore:
    def __init__(self):
        self.opensearch = aws_config.get_opensearch_client()
        self.index_name = "customer-support-kb"  # Configurable
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_dimension = 384
```

**Index Mapping** (OpenSearch schema):
```json
{
  "mappings": {
    "properties": {
      "text": {"type": "text"},           // Full text for retrieval
      "metadata": {"type": "object"},     // Flexible metadata structure
      "embedding": {
        "type": "knn_vector",             // Vector for similarity search
        "dimension": 384                  // Embedding dimension
      }
    }
  }
}
```

---

## Data Ingestion Pipeline

### Step 1: Document Preparation

Documents come from various sources:
- Product documentation (PDFs, markdown files)
- FAQ databases
- Support ticket history
- Company policies
- Knowledge base articles

**Preprocessing Steps**:
1. **Text Extraction**: Extract text from various formats (PDF, HTML, markdown)
2. **Chunking**: Split large documents into smaller chunks (200-500 tokens)
   - Preserves context with overlap between chunks
   - Each chunk becomes a separate vector
3. **Metadata Extraction**: Extract source, domain, date, version info

### Step 2: Embedding Generation

**File**: `src/models/rag/vector_store.py` (lines 50-85)

```python
def add_documents(self, documents: List[Dict[str, Any]]):
    """Add documents to vector store."""
    for i, doc in enumerate(documents):
        text = doc.get("text", doc.get("content", ""))
        metadata = doc.get("metadata", {})
        
        # Generate embedding using Sentence Transformers
        embedding = self.embedding_model.encode(text).tolist()
        # Returns: [0.123, -0.456, 0.789, ...] (384 values)
        
        # Index document in OpenSearch
        self.opensearch.index(
            index=self.index_name,
            id=i,  # Document ID
            body={
                "text": text,              # Original text
                "metadata": metadata,       # Metadata
                "embedding": embedding     # Vector embedding
            }
        )
    
    # Refresh index to make documents searchable
    self.opensearch.indices.refresh(index=self.index_name)
```

**Embedding Model Details**:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Characteristics**:
  - Fast inference (~100ms per document)
  - Good semantic understanding
  - Handles up to 512 tokens per text
  - Automatically handles: lowercasing, tokenization, special characters

### Step 3: Index Creation

**File**: `src/models/rag/vector_store.py` (lines 21-48)

```python
def create_index(self):
    """Create OpenSearch index for vector storage."""
    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "metadata": {"type": "object"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384
                }
            }
        }
    }
    
    self.opensearch.indices.create(index=self.index_name, body=mapping)
```

---

## Data Retrieval Process

### Step-by-Step Retrieval Flow

**File**: `src/models/rag/retriever.py`

#### Step 1: Query Processing

```python
def retrieve(self, query: str, intent: Optional[str] = None, 
             filters: Optional[Dict[str, Any]] = None):
    """Retrieve relevant documents."""
    
    # Add intent to filters if provided
    if intent:
        if filters is None:
            filters = {}
        filters["intent"] = intent  # Filter by domain
```

#### Step 2: Query Embedding

**File**: `src/models/rag/vector_store.py` (lines 104-106)

```python
# Convert query to embedding (same model as documents)
query_embedding = self.embedding_model.encode(query).tolist()
# Example: "How do I reset password?" → [0.234, -0.567, 0.890, ...]
```

#### Step 3: Vector Similarity Search

**File**: `src/models/rag/vector_store.py` (lines 108-137)

```python
# Build KNN search query
search_body = {
    "size": top_k,  # Number of results (default: 5)
    "query": {
        "knn": {
            "embedding": {
                "vector": query_embedding,  # Query vector
                "k": top_k                  # Top K similar vectors
            }
        }
    }
}

# Add metadata filters (if provided)
if filters:
    search_body["query"] = {
        "bool": {
            "must": [search_body["query"]],  # KNN search
            "filter": [                       # Metadata filters
                {"term": {f"metadata.{k}": v}}
                for k, v in filters.items()
            ]
        }
    }

# Execute search
response = self.opensearch.search(
    index=self.index_name,
    body=search_body
)
```

**How KNN Search Works**:
1. OpenSearch compares query embedding with all document embeddings
2. Calculates cosine similarity (or Euclidean distance)
3. Returns top K most similar documents
4. Results are ranked by similarity score

#### Step 4: Result Formatting

```python
# Format results
results = []
for hit in response["hits"]["hits"]:
    results.append({
        "text": hit["_source"]["text"],           # Document text
        "metadata": hit["_source"].get("metadata", {}),  # Metadata
        "score": hit["_score"]                    # Similarity score (0-1)
    })

return results  # Sorted by relevance (highest score first)
```

### Retrieval Characteristics

1. **Semantic Search**: Finds documents by meaning, not just keywords
   - Example: "password reset" matches "How to change your login credentials"
   
2. **Metadata Filtering**: Can filter by:
   - Intent (billing, technical_support, etc.)
   - Domain
   - Source document
   - Date range
   - Version

3. **Hybrid Approach**: Combines:
   - Vector similarity (semantic matching)
   - Metadata filtering (domain-specific)

4. **Top-K Retrieval**: Returns top 5 most relevant documents (configurable)

---

## Data Update Pipeline

### Current Implementation

The current codebase has the foundation for updates, but here's how it should work:

### Method 1: Add New Documents

**File**: `src/models/rag/vector_store.py`

```python
# Simply add new documents - they get new IDs
vector_store.add_documents([
    {
        "text": "Updated refund policy: 30-day window...",
        "metadata": {
            "source": "policy_v2.pdf",
            "last_updated": "2024-01-20T10:00:00Z",
            "version": "2.0",
            "domain": "billing"
        }
    }
])
```

### Method 2: Update Existing Documents

**To implement document updates**, you would:

```python
def update_document(self, doc_id: str, new_text: str, 
                    updated_metadata: Dict[str, Any]):
    """Update an existing document."""
    # Generate new embedding for updated text
    new_embedding = self.embedding_model.encode(new_text).tolist()
    
    # Update document in OpenSearch
    self.opensearch.update(
        index=self.index_name,
        id=doc_id,
        body={
            "doc": {
                "text": new_text,
                "metadata": updated_metadata,
                "embedding": new_embedding
            }
        }
    )
    
    # Refresh index
    self.opensearch.indices.refresh(index=self.index_name)
```

### Method 3: Delete Outdated Documents

```python
def delete_document(self, doc_id: str):
    """Delete a document from the index."""
    self.opensearch.delete(
        index=self.index_name,
        id=doc_id
    )
```

### Recommended Update Pipeline

Here's a complete update pipeline you could implement:

```python
# src/models/rag/document_updater.py

class DocumentUpdater:
    """Manages document updates in RAG system."""
    
    def __init__(self):
        self.vector_store = vector_store
    
    def update_knowledge_base(
        self,
        updates: List[Dict[str, Any]],
        update_strategy: str = "replace"
    ):
        """
        Update knowledge base with new information.
        
        Args:
            updates: List of updates with:
                - doc_id: Document ID to update (if exists)
                - text: New/updated text
                - metadata: Updated metadata
            update_strategy: "replace", "append", or "version"
        """
        for update in updates:
            doc_id = update.get("doc_id")
            new_text = update.get("text")
            metadata = update.get("metadata", {})
            
            if doc_id and self.vector_store.opensearch.exists(
                index=self.vector_store.index_name,
                id=doc_id
            ):
                # Update existing document
                if update_strategy == "replace":
                    self._replace_document(doc_id, new_text, metadata)
                elif update_strategy == "version":
                    self._version_document(doc_id, new_text, metadata)
            else:
                # Add new document
                self.vector_store.add_documents([{
                    "text": new_text,
                    "metadata": metadata
                }])
    
    def _replace_document(self, doc_id: str, text: str, metadata: Dict):
        """Replace document completely."""
        embedding = self.vector_store.embedding_model.encode(text).tolist()
        
        self.vector_store.opensearch.index(
            index=self.vector_store.index_name,
            id=doc_id,
            body={
                "text": text,
                "metadata": {**metadata, "last_updated": datetime.utcnow().isoformat()},
                "embedding": embedding
            }
        )
    
    def _version_document(self, doc_id: str, text: str, metadata: Dict):
        """Create new version, keep old one."""
        # Get old document
        old_doc = self.vector_store.opensearch.get(
            index=self.vector_store.index_name,
            id=doc_id
        )
        
        # Archive old version
        old_metadata = old_doc["_source"]["metadata"]
        old_metadata["status"] = "archived"
        old_metadata["archived_at"] = datetime.utcnow().isoformat()
        
        # Create new version
        new_doc_id = f"{doc_id}_v{old_metadata.get('version', 1) + 1}"
        self.vector_store.add_documents([{
            "text": text,
            "metadata": {
                **metadata,
                "version": old_metadata.get("version", 1) + 1,
                "original_id": doc_id
            }
        }])
```

### Automated Update Pipeline

For continuous updates, you could set up:

```python
# Pipeline: S3 → Process → Update RAG

def sync_knowledge_base_from_s3(s3_bucket: str, s3_prefix: str):
    """
    Sync knowledge base from S3 bucket.
    
    Flow:
    1. Monitor S3 for new/updated documents
    2. Process documents (extract, chunk)
    3. Generate embeddings
    4. Update OpenSearch index
    """
    s3_client = aws_config.get_s3_client()
    
    # List objects in S3
    objects = s3_client.list_objects_v2(
        Bucket=s3_bucket,
        Prefix=s3_prefix
    )
    
    for obj in objects.get("Contents", []):
        # Download document
        content = s3_client.get_object(
            Bucket=s3_bucket,
            Key=obj["Key"]
        )["Body"].read()
        
        # Extract text (handle PDF, HTML, etc.)
        text = extract_text_from_content(content, obj["Key"])
        
        # Chunk text
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        # Add to vector store
        documents = [
            {
                "text": chunk,
                "metadata": {
                    "source": obj["Key"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "chunk_id": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        
        vector_store.add_documents(documents)
```

---

## Complete RAG Flow: Query to Response

### End-to-End Process

**File**: `src/models/rag/rag_pipeline.py`

```python
def generate_response(self, query: str, intent: Optional[str] = None):
    """
    Complete RAG flow:
    1. Retrieve → 2. Augment → 3. Generate
    """
    
    # STEP 1: RETRIEVAL
    documents = self.retriever.retrieve(query, intent=intent)
    # Returns: List of relevant documents with scores
    
    if not documents:
        # Fallback if no documents found
        return self.llm.generate_customer_support_response(query, intent)
    
    # STEP 2: AUGMENTATION (Format Context)
    context = self.retriever.format_context(documents, max_length=1000)
    # Combines retrieved documents into context string
    
    # STEP 3: PROMPT ENGINEERING
    prompt = f"""Based on the following knowledge base documents, answer the user's question.

Knowledge Base:
{context}

User Question: {query}

Provide a helpful answer based on the knowledge base. 
If the answer is not in the knowledge base, say so clearly."""
    
    # STEP 4: GENERATION
    response = self.llm.generate_response(prompt)
    # Uses AWS Bedrock/Claude to generate response
    
    # STEP 5: SOURCE ATTRIBUTION
    response["sources"] = [
        {
            "text": doc["text"][:200],  # Preview
            "metadata": doc.get("metadata", {}),
            "relevance_score": doc.get("score", 0.0)
        }
        for doc in documents
    ]
    
    return response
```

### Flow Diagram

```
User Query: "How do I reset my password?"
    │
    ▼
┌─────────────────────────────────┐
│ 1. Query Embedding               │
│    "reset password" → [0.23, ...]│
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 2. Vector Similarity Search      │
│    Find top 5 similar documents │
│    Score: 0.89, 0.85, 0.82, ... │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 3. Metadata Filtering           │
│    Filter by intent if provided │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 4. Context Formatting           │
│    Combine documents:           │
│    "Document 1: To reset..."    │
│    "Document 2: Password..."    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 5. Prompt Augmentation          │
│    "Based on these docs:        │
│     [context]                   │
│     Answer: [query]"           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 6. LLM Generation               │
│    Claude generates response    │
│    with citations               │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 7. Response + Sources           │
│    Answer + Source documents    │
└─────────────────────────────────┘
```

---

## RAG Characteristics and Features

### 1. **Semantic Understanding**
- Uses embeddings to understand meaning, not just keywords
- Handles synonyms and paraphrasing
- Example: "password reset" matches "change login credentials"

### 2. **Source Attribution**
- Every response includes source documents
- Users can verify information
- Builds trust and transparency

### 3. **Metadata Filtering**
- Can filter by domain, intent, source, date
- Improves relevance for domain-specific queries
- Example: Billing queries only search billing documents

### 4. **Fallback Mechanisms**
- If no documents found → Falls back to standard generation
- If retrieval fails → Falls back to Bedrock
- Ensures system always responds

### 5. **Configurable Retrieval**
- Top-K configurable (default: 5)
- Similarity threshold can be set
- Context length limit (default: 1000 tokens)

### 6. **Real-time Updates**
- New documents can be added immediately
- Index refresh makes documents searchable instantly
- No model retraining needed for new information

---

## Configuration

**File**: `config/model_config.py`

```python
# RAG Configuration
OPENSEARCH_INDEX_NAME = "customer-support-kb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
TOP_K_RETRIEVAL = 5  # Number of documents to retrieve
```

---

## Best Practices for RAG

### 1. **Document Chunking**
- Optimal chunk size: 200-500 tokens
- Overlap: 50-100 tokens between chunks
- Preserves context across boundaries

### 2. **Metadata Management**
- Always include: source, domain, last_updated
- Use versioning for document updates
- Track document lifecycle (active, archived)

### 3. **Update Strategy**
- **Replace**: For corrections
- **Version**: For policy changes (keep history)
- **Append**: For new information

### 4. **Quality Control**
- Monitor retrieval quality (relevance scores)
- Track which documents are used most
- Update low-quality documents

### 5. **Performance Optimization**
- Batch document additions
- Use async operations for large updates
- Monitor index size and performance

---

## Summary

**RAG Design**:
- Three-layer architecture: Vector Store → Retriever → Pipeline
- OpenSearch for vector storage with KNN search
- Sentence Transformers for embeddings
- AWS Bedrock for generation

**Data Management**:
- Documents stored with text, metadata, and embeddings
- Flexible metadata structure
- Support for versioning and updates

**Retrieval**:
- Semantic similarity search
- Metadata filtering
- Top-K retrieval with relevance scoring

**Updates**:
- Add new documents easily
- Update existing documents (needs implementation)
- Delete outdated documents
- Automated sync from S3 (can be implemented)

**Pipeline**:
- Complete flow: Query → Embed → Search → Retrieve → Augment → Generate
- Fallback mechanisms at each step
- Source attribution in responses

This RAG system provides accurate, source-cited responses that can be updated in real-time without model retraining!

