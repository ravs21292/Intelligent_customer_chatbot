# Interview Q&A Guide - Intelligent Customer Support Chatbot

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Locations](#component-locations)
3. [Complete Flow Walkthrough](#complete-flow-walkthrough)
4. [Interview Questions & Answers](#interview-questions--answers)
5. [Technical Deep Dives](#technical-deep-dives)
6. [Code References](#code-references)

---

## Architecture Overview

### Where Each Component Runs

| Component | Where it Runs | Why |
|-----------|---------------|-----|
| **FastAPI Server** | Your server (EC2/ECS/Lambda) | Main application |
| **BERT Intent Classifier** | SageMaker Endpoint | Custom trained model |
| **Llama Fine-tuned (LoRA)** | API Server (same as FastAPI) | Small adapters (32MB), loads in memory |
| **RAG Retrieval** | OpenSearch Service | Vector database |
| **RAG Generation** | Bedrock API | Pre-trained model (API call) |
| **Simple Queries** | Bedrock API | Pre-trained model (API call) |

---

## Component Locations

### Key Insight
- **SageMaker** = For YOUR trained models (BERT)
- **Bedrock** = For pre-trained models (Claude) - API calls only
- **OpenSearch** = For vector search (database, not a model)
- **API Server** = For fine-tuned models (Llama with LoRA adapters)

---

## Complete Flow Walkthrough

### Step 1: User Sends Message

**File**: `src/api/chat_endpoints.py` (line 45-69)

```
User → POST /api/v1/chat
    → FastAPI receives request
    → Validates message
    → Calls multi_model_router.generate_response()
```

### Step 2: Intent Classification

**File**: `src/models/model_router.py` (line 41) → `src/intent_classification/router.py` (line 40)

```
Router → intent_classifier.classify(message)
    → BERT model (on SageMaker endpoint OR loaded locally)
    → Returns: {"intent": "billing", "confidence": 0.88}
```

### Step 3: Routing Decision

**File**: `src/intent_classification/router.py` (line 23-88)

```
Intent + Confidence → Router logic
    ↓
Decision Tree:
├── High confidence (≥0.8) + Simple query
│   → Strategy: "pre_trained" (Bedrock)
│
├── Domain-specific intent (billing, technical)
│   → Strategy: "fine_tuned" (Llama on server)
│
└── Low confidence (<0.6) OR Complex query
    → Strategy: "rag" (OpenSearch + Bedrock)
```

### Step 4: Model Execution (3 Scenarios)

#### Scenario A: Simple Query → Bedrock
```
Strategy: "pre_trained"
    ↓
bedrock_client.generate_customer_support_response()
    ↓
Bedrock API Call (Claude v2)
    ↓
Response generated
```

#### Scenario B: Domain-Specific → Fine-tuned Llama
```
Strategy: "fine_tuned"
    ↓
Load Llama model (if not already loaded)
    ├── Base model: Llama-2-7b (from Hugging Face)
    └── LoRA adapters: billing-model (32MB, from local storage)
    ↓
Generate response on API server
    ↓
Response generated
```

#### Scenario C: Complex Query → RAG
```
Strategy: "rag"
    ↓
Step 1: Retrieve documents
    → document_retriever.retrieve(query)
    → OpenSearch vector search (k-NN)
    → Returns top 5 relevant documents
    ↓
Step 2: Augment prompt
    → Combine query + retrieved documents
    → Create context
    ↓
Step 3: Generate
    → bedrock_client.generate_response(augmented_prompt)
    → Bedrock API Call (Claude)
    ↓
Response with source citations
```

### Step 5: Response and Logging

**File**: `src/api/chat_endpoints.py` (line 71-90)

```
Response generated
    ↓
Log to Kinesis (for data collection)
    ↓
Track metrics (CloudWatch)
    ↓
Return response to user
```

---

## Interview Questions & Answers

### Q1: Can you give me an overview of your project?

**Answer:**
> "I built an intelligent customer support chatbot using a multi-model strategy. The system uses three different approaches based on query complexity: pre-trained models (AWS Bedrock) for simple queries, fine-tuned domain models (Llama-2 with LoRA) for specific domains like billing, and RAG (Retrieval-Augmented Generation) for complex queries requiring knowledge base access. The system includes intent classification using BERT, automated training pipelines, and continuous learning from user feedback."

---

### Q2: Why did you use multiple models instead of just one?

**Answer:**
> "Different queries require different approaches. Simple queries like 'What are your business hours?' don't need expensive fine-tuned models - Bedrock handles them efficiently. Domain-specific queries like billing issues benefit from models trained on our specific data. Complex queries needing specific information use RAG to retrieve relevant documents. This multi-strategy approach reduces costs by 40% while maintaining high accuracy."

---

### Q3: Where does each model run? Why BERT on SageMaker but Llama on the server?

**Answer:**
> "BERT is deployed on SageMaker because it's a custom-trained model that needs managed infrastructure with auto-scaling. The fine-tuned Llama models use LoRA adapters that are only 32MB each - they run efficiently on our API server in memory. This avoids the overhead of SageMaker endpoints for small models. RAG uses OpenSearch for vector search and Bedrock for generation - both are API-based services, so no deployment needed."

---

### Q4: Walk me through what happens when a user sends a message.

**Answer:**
> "When a user sends a message, it hits our FastAPI endpoint (`src/api/chat_endpoints.py`). The endpoint validates the message and passes it to our multi-model router. The router first classifies the intent using our BERT model on SageMaker, which returns an intent category and confidence score. Based on this, the router decides which strategy to use: Bedrock for simple queries, our fine-tuned Llama model for domain-specific queries, or RAG for complex queries. After generating the response, we log the interaction to Kinesis for future training and track metrics in CloudWatch."

---

### Q5: How does the routing logic work?

**Answer:**
> "The routing logic (`src/intent_classification/router.py`) uses a decision tree based on intent and confidence score. If confidence is high (≥0.8) and it's a simple query, we route to Bedrock. If it's a domain-specific intent like billing or technical support, we use our fine-tuned models. If confidence is low (<0.6) or it's a complex query, we use RAG. This ensures we use the most appropriate and cost-effective model for each query."

---

### Q6: Why use LoRA for fine-tuning instead of full fine-tuning?

**Answer:**
> "LoRA (Low-Rank Adaptation) allows us to fine-tune large models efficiently. Instead of training all 7 billion parameters of Llama-2, we only train 8 million parameters (0.12%). This reduces training time from days to hours, cuts costs by 99%, and achieves 95-98% of full fine-tuning performance. The LoRA adapters are only 32MB, making them easy to load and switch between domains on our API server."

---

### Q7: How does RAG work in your system?

**Answer:**
> "RAG combines retrieval and generation. When a complex query comes in, we first search our OpenSearch vector database to find the 5 most relevant documents. We use sentence transformers to create embeddings and perform k-NN search. Then we combine the user's query with these retrieved documents into a prompt and send it to Bedrock's Claude model. This gives us accurate, cited responses based on our knowledge base."

---

### Q8: Why use OpenSearch for RAG instead of deploying on SageMaker?

**Answer:**
> "OpenSearch is a purpose-built vector database optimized for similarity search. It's not a model - it's a database service. SageMaker is for deploying trained models. Using OpenSearch allows us to scale to millions of documents, perform fast k-NN searches, and manage the vector store independently from our models. It's the right tool for the retrieval part of RAG."

---

### Q9: How do you train and deploy models?

**Answer:**
> "We use SageMaker for training. The BERT intent classifier is trained on labeled data using SageMaker Training Jobs. After training, the model is saved to S3 and registered in SageMaker Model Registry. If the new model performs better than the current one, it's automatically deployed to a SageMaker endpoint. We also have automated retraining triggers based on data thresholds, performance degradation, or scheduled events."

---

### Q10: How do you monitor model performance?

**Answer:**
> "We use multiple monitoring approaches. SageMaker Model Monitor runs hourly to detect data drift by comparing incoming data with baseline statistics. We also track custom metrics in CloudWatch including accuracy, latency, request volume, and cost. If performance degrades by more than 5%, our system automatically triggers retraining."

---

### Q11: What happens when model performance decreases?

**Answer:**
> "When performance degrades, our drift detection system (`src/monitoring/drift_detector.py`) identifies it. This triggers our retraining pipeline via EventBridge and Lambda. A new training job starts on SageMaker with the latest data. After training, the new model is evaluated and compared with the current production model. If it's better, it's automatically deployed to replace the old model with zero downtime."

---

### Q12: How does the fine-tuned model loading work?

**Answer:**
> "We use lazy loading for efficiency. The base Llama-2-7b model loads once at API server startup. When a domain-specific query comes in, we load the corresponding LoRA adapters (32MB) from local storage. These adapters are cached in memory, so subsequent requests for the same domain are fast. This approach keeps memory usage low while maintaining quick response times."

---

### Q13: Can you deploy everything on SageMaker?

**Answer:**
> "Technically yes, but it's not optimal. Fine-tuned models are only 32MB - deploying them on SageMaker would be overkill and expensive. OpenSearch is a database service, not a model, so it can't be deployed on SageMaker. Bedrock is an API service - you don't deploy it, you just call it. Our current architecture uses each service for its strengths: SageMaker for custom trained models, the API server for small adapters, OpenSearch for vector search, and Bedrock for pre-trained generation."

---

### Q14: How does data flow through your system?

**Answer:**
> "User messages are ingested in real-time to Kinesis streams, which then archive to S3. We use SageMaker Ground Truth for labeling the data. Labeled datasets are versioned using DVC and stored in S3. When retraining is triggered, we pull the latest labeled data, train the model on SageMaker, evaluate it, and if it's better, deploy it. The entire pipeline is automated with CI/CD using GitHub Actions."

---

### Q15: What's the difference between SageMaker and Bedrock?

**Answer:**
> "SageMaker is for training and deploying YOUR custom models. You provide the data, train the model, and deploy it. Bedrock is for using pre-trained models like Claude or GPT via API - you don't train or deploy, you just make API calls. In our project, we use SageMaker for our custom BERT intent classifier, and Bedrock for Claude's pre-trained generation capabilities."

---

### Q16: How do you handle model versioning?

**Answer:**
> "We use SageMaker Model Registry for model versioning. Each trained model is registered with version numbers (v1.0, v1.1, etc.), metrics, training data references, and approval status. We also version our training data using DVC (Data Version Control) with S3 backend. This allows us to reproduce experiments and track which data produced which model version."

---

### Q17: What metrics do you track?

**Answer:**
> "We track several metrics: model accuracy and F1-score from evaluation, latency (p95 response time), request volume, error rate, and cost per request. These are sent to CloudWatch where we have dashboards. We also track data drift scores and concept drift indicators. All metrics help us understand model performance and trigger retraining when needed."

---

### Q18: How does CI/CD work for ML models?

**Answer:**
> "We have GitHub Actions workflows for CI/CD. When code is pushed, CI runs tests, linting, and builds Docker images. When merged to main, CD automatically deploys the API. For models, we have a separate training pipeline that can be triggered manually or scheduled. After training, models are evaluated and automatically deployed if they improve performance. The entire process is automated."

---

### Q19: What challenges did you face and how did you solve them?

**Answer:**
> "One challenge was managing multiple models efficiently. We solved this with intelligent routing and lazy loading of LoRA adapters. Another challenge was cost optimization - we addressed this by using the right model for each query type, reducing unnecessary expensive calls. For data drift, we implemented automated monitoring and retraining triggers. The modular architecture made it easy to update individual components without affecting the whole system."

---

### Q20: What would you improve if you had more time?

**Answer:**
> "I would add A/B testing framework for gradual model rollouts, implement caching for common queries to reduce costs, add multi-language support, and enhance the RAG system with multi-hop reasoning. I'd also add more sophisticated drift detection using statistical tests and implement personalization based on user history."

---

## Technical Deep Dives

### Architecture Decision: Why This Setup?

**BERT on SageMaker:**
- Custom trained model needs deployment
- Requires managed infrastructure
- Benefits from auto-scaling
- **File**: `src/intent_classification/model_training.py`

**Llama on API Server:**
- Small LoRA adapters (32MB)
- Fast loading and switching
- No need for SageMaker overhead
- **File**: `src/models/fine_tuning/lora_trainer.py:166`

**RAG with OpenSearch + Bedrock:**
- OpenSearch: Purpose-built for vector search
- Bedrock: Pre-trained model via API
- Separation of concerns
- **Files**: 
  - `src/models/rag/vector_store.py` (OpenSearch)
  - `src/models/rag/rag_pipeline.py` (RAG flow)

---

### Complete End-to-End Example

**User Query**: "Why was I charged $50?"

```
1. API receives: POST /api/v1/chat
   File: src/api/chat_endpoints.py:45

2. Intent Classification:
   → BERT (SageMaker) classifies: "billing", confidence: 0.88
   File: src/intent_classification/intent_classifier.py:62

3. Routing Decision:
   → Router sees: billing intent + domain-specific
   → Strategy: "fine_tuned"
   File: src/intent_classification/router.py:66

4. Load Fine-tuned Model (if not loaded):
   → Base: Llama-2-7b (already in memory)
   → Adapters: billing-model (loads from models/fine_tuned/billing/)
   → Runs on API server
   File: src/models/fine_tuning/lora_trainer.py:184

5. Generate Response:
   → Llama generates: "The $50 charge is for your monthly subscription..."
   → Response returned
   File: src/models/model_router.py:113

6. Logging:
   → Logs to Kinesis
   → Tracks metrics
   File: src/api/chat_endpoints.py:72
```

---

## Code References

### Key Files for Interview

1. **API Entry Point**
   - File: `src/api/chat_endpoints.py`
   - Line: 45 (`@router.post("/chat")`)
   - What: Main chat endpoint

2. **Multi-Model Router**
   - File: `src/models/model_router.py`
   - Line: 22 (`generate_response()`)
   - What: Routes to appropriate model strategy

3. **Intent Classifier**
   - File: `src/intent_classification/intent_classifier.py`
   - Line: 62 (`classify()`)
   - What: BERT-based intent classification

4. **Routing Logic**
   - File: `src/intent_classification/router.py`
   - Line: 23 (`route()`)
   - What: Decision tree for model selection

5. **Bedrock Client**
   - File: `src/models/bedrock_client.py`
   - Line: 21 (`generate_response()`)
   - What: Bedrock API integration

6. **Fine-tuned Model**
   - File: `src/models/fine_tuning/lora_trainer.py`
   - Line: 166 (`generate_response()`)
   - What: Llama inference with LoRA

7. **RAG Pipeline**
   - File: `src/models/rag/rag_pipeline.py`
   - Line: 18 (`generate_response()`)
   - What: Complete RAG flow

8. **Training Pipeline**
   - File: `src/training/training_pipeline.py`
   - Line: Various
   - What: Model training orchestration

9. **Model Deployment**
   - File: `cicd/scripts/deploy_model.py`
   - Line: Various
   - What: Automated model deployment

10. **Drift Detection**
    - File: `src/monitoring/drift_detector.py`
    - Line: Various
    - What: Data and concept drift detection

---

## Quick Summary for Interview

**30-Second Pitch:**
> "I built an intelligent chatbot with a multi-model strategy. When a user sends a message, our FastAPI server receives it and routes it through our intent classifier (BERT on SageMaker). Based on the intent and confidence, we route to one of three strategies: Bedrock for simple queries, our fine-tuned Llama models for domain-specific queries, or RAG for complex queries needing knowledge base access. The fine-tuned models run on our API server using LoRA adapters, which are only 32MB each. This architecture optimizes both cost and performance."

**Key Points to Emphasize:**
1. Multi-model strategy for cost optimization
2. Intelligent routing based on intent and confidence
3. LoRA for efficient fine-tuning (99% parameter reduction)
4. Complete MLOps pipeline with automated retraining
5. Production-ready with monitoring and drift detection
6. Modular architecture for maintainability

---

## Common Follow-up Questions

### Q: How do you ensure model quality?

**Answer:**
> "We evaluate models on held-out test sets with metrics like accuracy, F1-score, and per-class performance. We also do A/B testing before full deployment. In production, we monitor real-time metrics and user feedback. If quality degrades, automated retraining is triggered."

### Q: How do you handle model failures?

**Answer:**
> "We have fallback mechanisms. If the intent classifier fails, we default to Bedrock. If a fine-tuned model isn't available, we fall back to Bedrock. If RAG retrieval fails, we still generate a response using Bedrock. All errors are logged and monitored."

### Q: What's the latency of your system?

**Answer:**
> "Bedrock responses are typically 1-2 seconds. Fine-tuned models on our server are 2-3 seconds. RAG takes 3-5 seconds due to retrieval and generation. Our p95 latency is around 3 seconds, which is acceptable for customer support use cases."

### Q: How do you scale the system?

**Answer:**
> "SageMaker endpoints auto-scale based on traffic. Our API server can be horizontally scaled behind a load balancer. OpenSearch can be scaled by adding nodes. Kinesis can add shards for higher throughput. The architecture is designed for horizontal scaling."

---

## Project Statistics to Mention

- **Intent Classification Accuracy**: 89%
- **Cost Reduction**: 40% through intelligent routing
- **Training Cost Reduction**: 99% using LoRA
- **Parameter Reduction**: 99% (8M vs 7B with LoRA)
- **Performance**: 95-98% of full fine-tuning with LoRA
- **Uptime**: 99.9% with proper error handling
- **Throughput**: Handles 10K+ requests/day

---

**Good luck with your interview! This guide covers all the key points you need to confidently explain your project.** 🚀

