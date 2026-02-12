# Complete Data Pipeline Guide: Kinesis → Preprocessing → OpenSearch → Automated Training

## 📋 Overview

This guide explains the complete data flow:
1. **Kinesis** - Real-time data ingestion
2. **Data Pipeline** - Processing and storage
3. **Preprocessing** - Data cleaning and preparation
4. **OpenSearch** - Vector storage for RAG
5. **Automated Training** - Triggered retraining

---

## 🗂️ File Locations

### 1. Kinesis Ingestion
**File**: `src/data_collection/kinesis_ingestion.py`
- **Class**: `KinesisIngestion`
- **Key Methods**:
  - `ingest_chat_message()` - Ingest single message (line 45)
  - `batch_ingest()` - Batch ingestion (line 97)
  - `consume_stream()` - Consume from stream (line 144)
  - `archive_to_s3()` - Archive to S3 (line 186)

### 2. Data Storage (S3)
**File**: `src/data_collection/s3_storage.py`
- **Class**: `S3Storage`
- **Key Methods**:
  - `upload_data()` - Upload to S3 (line 23)
  - `download_data()` - Download from S3
  - `upload_model()` - Upload model artifacts (line 168)

### 3. Data Versioning
**File**: `src/data_collection/data_versioning.py`
- **Class**: `DataVersioning`
- **Key Methods**:
  - `track_dataset()` - Track with DVC (line 49)
  - `push_dataset()` - Push to remote (line 94)

### 4. OpenSearch Vector Store
**File**: `src/models/rag/vector_store.py`
- **Class**: `VectorStore`
- **Key Methods**:
  - `create_index()` - Create OpenSearch index (line 21)
  - `add_documents()` - Add documents with embeddings (line 50)
  - `search()` - Search similar documents (line 87)

### 5. Data Preparation (Preprocessing)
**File**: `src/models/fine_tuning/data_preparation.py`
- **Class**: `FineTuningDataPreparation`
- **Key Methods**:
  - `extract_qa_pairs()` - Extract Q&A from tickets (line 13)
  - `augment_data()` - Data augmentation (line 42)
  - `prepare_domain_specific_data()` - Domain filtering (line 72)

### 6. Automated Training Triggers
**File**: `src/training/retraining_trigger.py`
- **Class**: `RetrainingTrigger`
- **Key Methods**:
  - `create_scheduled_retraining()` - EventBridge schedule (line 19)
  - `lambda_handler()` - Lambda trigger (line 82)

### 7. Incremental Learning
**File**: `src/training/incremental_learning.py`
- **Class**: `IncrementalLearning`
- **Key Methods**:
  - `check_retraining_conditions()` - Check if retrain needed (line 50)
  - `trigger_retraining()` - Trigger training (line 82)

### 8. API Integration (Entry Point)
**File**: `src/api/chat_endpoints.py`
- **Function**: `chat()` - Main endpoint (line 45)
- **Line 72**: Calls `kinesis_ingestion.ingest_chat_message()`

---

## 🔄 Complete Data Flow

### Step 1: User Message → Kinesis Stream

**Flow:**
```
User sends message via API
    ↓
src/api/chat_endpoints.py (line 45)
    ↓
kinesis_ingestion.ingest_chat_message() (line 72)
    ↓
Kinesis Stream (customer-chat)
```

**Code Location**: `src/api/chat_endpoints.py` (line 71-80)

```python
# After generating response, ingest to Kinesis
kinesis_ingestion.ingest_chat_message(
    user_id=request.user_id,
    message=request.message,
    session_id=conversation_id,
    metadata={
        "intent": response.get("intent"),
        "strategy": response.get("strategy")
    }
)
```

**What Happens:**
- Message sent to Kinesis stream
- Partitioned by `user_id` (line 76)
- Includes metadata (intent, strategy, timestamp)

---

### Step 2: Kinesis → S3 Archive (Raw Data)

**Flow:**
```
Kinesis Stream
    ↓
Kinesis Consumer (Lambda/Kinesis Analytics)
    ↓
Batch Processing
    ↓
S3 Raw Data (s3://bucket/raw/YYYY/MM/DD/)
```

**Code Location**: `src/data_collection/kinesis_ingestion.py` (line 186-215)

**Method**: `archive_to_s3()`

```python
def archive_to_s3(self, records: List[Dict[str, Any]], date_prefix: str):
    """Archive records to S3 for long-term storage."""
    key = f"{data_collection_config.RAW_DATA_PREFIX}/{date_prefix}/batch_{int(time.time())}.json"
    
    self.s3_client.put_object(
        Bucket=self.bucket,
        Key=key,
        Body=json.dumps(records, indent=2),
        ContentType="application/json"
    )
```

**What Happens:**
- Records batched from Kinesis
- Archived to S3 with date prefix
- Format: `raw/2024/01/15/batch_1234567890.json`

---

### Step 3: Data Preprocessing & Cleaning

**Flow:**
```
S3 Raw Data
    ↓
Data Preparation Pipeline
    ↓
Cleaned/Processed Data
```

**Code Locations:**

#### A. For Fine-tuning Data
**File**: `src/models/fine_tuning/data_preparation.py`

**Methods:**
1. **`extract_qa_pairs()`** (line 13-40)
   - Extracts Q&A pairs from support tickets
   - Filters empty questions/answers
   - Adds domain labels

2. **`augment_data()`** (line 42-70)
   - Data augmentation (rephrasing)
   - Increases dataset size

3. **`prepare_domain_specific_data()`** (line 72-106)
   - Filters by domain (billing, technical, etc.)
   - Loads from S3 or local
   - Returns cleaned, formatted data

**Example:**
```python
# Load raw data
data = s3_storage.download_data("raw/2024/01/15/batch_123.json")

# Extract Q&A pairs
qa_pairs = data_preparation.extract_qa_pairs(data)
# Result: [{"instruction": "...", "response": "...", "domain": "billing"}]

# Augment
augmented = data_preparation.augment_data(qa_pairs)
# Result: 2x dataset size with variations
```

#### B. For Intent Classification Data
**File**: `src/intent_classification/model_training.py`

**Method**: `prepare_data()` (line 68-108)

**What it does:**
- Loads labeled data
- Extracts text and labels
- Splits into train/val/test
- Handles missing values

---

### Step 4: Cleaned Data → OpenSearch (RAG)

**Flow:**
```
Cleaned Documents
    ↓
Generate Embeddings (Sentence Transformers)
    ↓
Store in OpenSearch (Vector Store)
```

**Code Location**: `src/models/rag/vector_store.py`

**Method**: `add_documents()` (line 50-85)

```python
def add_documents(self, documents: List[Dict[str, Any]]):
    """Add documents to vector store."""
    for i, doc in enumerate(documents):
        text = doc.get("text", doc.get("content", ""))
        metadata = doc.get("metadata", {})
        
        # Generate embedding (preprocessing happens here)
        embedding = self.embedding_model.encode(text).tolist()
        
        # Index document
        self.opensearch.index(
            index=self.index_name,
            id=i,
            body={
                "text": text,              # Cleaned text
                "metadata": metadata,      # Domain, source, etc.
                "embedding": embedding    # Vector embedding
            }
        )
    
    # Refresh index
    self.opensearch.indices.refresh(index=self.index_name)
```

**What Happens:**
1. **Text Cleaning**: Handled by Sentence Transformers (lowercase, tokenization)
2. **Embedding Generation**: Converts text to 384-dim vector (default)
3. **Storage**: Saves text + metadata + embedding in OpenSearch
4. **Index Refresh**: Makes documents searchable

**Preprocessing Details:**
- Sentence Transformers automatically:
  - Lowercases text
  - Tokenizes
  - Handles special characters
  - Truncates to max length (512 tokens)
- No explicit cleaning code needed (handled by library)

---

### Step 5: Automated Training Trigger

**Flow:**
```
New Data Accumulated
    ↓
Check Retraining Conditions
    ↓
Trigger Training (EventBridge/Lambda)
    ↓
SageMaker Training Job
```

**Code Locations:**

#### A. Check Conditions
**File**: `src/training/incremental_learning.py`

**Method**: `check_retraining_conditions()` (line 50-68)

```python
def check_retraining_conditions(self) -> bool:
    """Check if retraining conditions are met."""
    # Check for new labeled data
    new_data_count = self._count_new_labeled_data()
    if new_data_count >= pipeline_config.MIN_NEW_SAMPLES_FOR_RETRAIN:
        return True
    
    # Check model performance degradation
    if self._check_performance_degradation():
        return True
    
    return False
```

**Conditions:**
1. **New Data Threshold**: Minimum new labeled samples (e.g., 1000)
2. **Performance Degradation**: Accuracy drops below threshold
3. **Scheduled**: Weekly/monthly schedule

#### B. Scheduled Trigger (EventBridge)
**File**: `src/training/retraining_trigger.py`

**Method**: `create_scheduled_retraining()` (line 19-65)

```python
def create_scheduled_retraining(
    self,
    schedule_expression: str = None,  # e.g., "cron(0 2 * * ? *)" (daily at 2 AM)
    lambda_function_name: str = "retraining-trigger"
) -> str:
    """Create scheduled retraining rule."""
    rule_name = "customer-chatbot-retraining-schedule"
    
    # Create EventBridge rule
    response = self.eventbridge.put_rule(
        Name=rule_name,
        ScheduleExpression=schedule_expression,  # Cron expression
        State="ENABLED"
    )
    
    # Add Lambda target
    self.eventbridge.put_targets(
        Rule=rule_name,
        Targets=[{
            "Id": "1",
            "Arn": f"arn:aws:lambda:...:function:{lambda_function_name}"
        }]
    )
```

**What Happens:**
- EventBridge rule created (e.g., weekly schedule)
- Triggers Lambda function at scheduled time
- Lambda calls `incremental_learning.check_retraining_conditions()`
- If conditions met → triggers training

#### C. Lambda Handler (Trigger Training)
**File**: `src/training/retraining_trigger.py`

**Method**: `lambda_handler()` (line 82-120)

```python
def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for retraining trigger."""
    # Check conditions
    if incremental_learning.check_retraining_conditions():
        # Trigger retraining
        model_path = incremental_learning.trigger_retraining()
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Retraining triggered",
                "model_path": model_path
            })
        }
```

#### D. Trigger Training
**File**: `src/training/incremental_learning.py`

**Method**: `trigger_retraining()` (line 82-112)

```python
def trigger_retraining(self, model_type: str = "intent_classifier") -> str:
    """Trigger model retraining."""
    # Get latest training data
    data_path = self._get_latest_training_data()
    
    if model_type == "intent_classifier":
        model_path = training_pipeline.run_intent_classification_training(
            data_path,
            use_sagemaker=True  # Train on SageMaker
        )
    else:
        model_path = training_pipeline.run_fine_tuning_pipeline(
            model_type,
            data_path
        )
    
    return model_path
```

**What Happens:**
1. Gets latest training data from S3
2. Calls training pipeline
3. Trains on SageMaker (or locally)
4. Returns model path

---

## 📊 Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE FLOW                         │
└─────────────────────────────────────────────────────────────┘

1. USER MESSAGE
   │
   ▼
┌─────────────────┐
│  FastAPI API    │  src/api/chat_endpoints.py:45
│  /api/v1/chat   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Kinesis Stream  │  src/data_collection/kinesis_ingestion.py:45
│ (customer-chat) │  Method: ingest_chat_message()
└────────┬────────┘
         │
         ├─────────────────────────────────┐
         │                                   │
         ▼                                   ▼
┌─────────────────┐                  ┌─────────────────┐
│ S3 Raw Archive  │                  │ Real-time       │
│ (raw/YYYY/MM/DD)│                  │ Processing      │
└────────┬────────┘                  └────────┬────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────┐                  ┌─────────────────┐
│ Data Labeling   │                  │ Data             │
│ (Ground Truth)  │                  │ Preprocessing    │
└────────┬────────┘                  │                  │
         │                           │ src/models/      │
         ▼                           │ fine_tuning/     │
┌─────────────────┐                  │ data_preparation │
│ S3 Labeled Data │                  │ .py              │
│ (labeled/...)   │                  └────────┬────────┘
└────────┬────────┘                           │
         │                                    │
         ├────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ OpenSearch      │  src/models/rag/vector_store.py:50
│ Vector Store    │  Method: add_documents()
│ (RAG)           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training Data   │
│ (S3)            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check Conditions│  src/training/incremental_learning.py:50
│ (New Data?      │  Method: check_retraining_conditions()
│ Performance?)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EventBridge      │  src/training/retraining_trigger.py:19
│ Scheduled Rule  │  Method: create_scheduled_retraining()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Lambda Trigger   │  src/training/retraining_trigger.py:82
│                 │  Method: lambda_handler()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trigger Training│  src/training/incremental_learning.py:82
│                 │  Method: trigger_retraining()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SageMaker       │  src/training/training_pipeline.py
│ Training Job    │  Method: run_intent_classification_training()
└─────────────────┘
```

---

## 🔍 Detailed Code Walkthrough

### 1. Kinesis Ingestion (Entry Point)

**File**: `src/data_collection/kinesis_ingestion.py`

**How it works:**
1. User sends message → API receives it
2. API calls `kinesis_ingestion.ingest_chat_message()` (line 72 in chat_endpoints.py)
3. Message formatted as JSON record
4. Sent to Kinesis stream with partition key = `user_id`
5. Returns success/failure

**Key Code:**
```python
# Line 45-95
def ingest_chat_message(self, user_id, message, session_id, metadata):
    record = {
        "user_id": user_id,
        "message": message,
        "session_id": session_id,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    
    response = self.kinesis.put_record(
        StreamName=self.stream_name,
        Data=json.dumps(record),
        PartitionKey=user_id  # Partition by user
    )
```

---

### 2. Data Preprocessing

**File**: `src/models/fine_tuning/data_preparation.py`

**How it works:**
1. Load raw data from S3 (line 88-92)
2. Extract Q&A pairs (line 13-40)
   - Filters empty questions/answers
   - Extracts customer_message → instruction
   - Extracts agent_response → response
   - Adds domain label
3. Augment data (line 42-70)
   - Creates variations (rephrasing)
   - Increases dataset size
4. Format for training (line 108-117)
   - Saves as JSON

**Key Code:**
```python
# Line 13-40: Extract Q&A pairs
def extract_qa_pairs(self, support_tickets):
    qa_pairs = []
    for ticket in support_tickets:
        question = ticket.get("customer_message", "")
        answer = ticket.get("agent_response", "")
        
        if question and answer:  # Filter empty
            qa_pairs.append({
                "instruction": question,
                "response": answer,
                "domain": ticket.get("category", "general")
            })
    return qa_pairs
```

---

### 3. OpenSearch Storage (RAG)

**File**: `src/models/rag/vector_store.py`

**How it works:**
1. Documents passed to `add_documents()` (line 50)
2. For each document:
   - Extract text (line 62)
   - Generate embedding using Sentence Transformers (line 66)
   - Store in OpenSearch with metadata (line 69-77)
3. Refresh index (line 80)

**Preprocessing in OpenSearch:**
- Sentence Transformers handles:
  - Lowercasing
  - Tokenization
  - Special character handling
  - Truncation (max 512 tokens)
- No explicit cleaning code needed

**Key Code:**
```python
# Line 50-85: Add documents
def add_documents(self, documents):
    for i, doc in enumerate(documents):
        text = doc.get("text", "")  # Cleaned text
        metadata = doc.get("metadata", {})
        
        # Generate embedding (preprocessing happens here)
        embedding = self.embedding_model.encode(text).tolist()
        
        # Store in OpenSearch
        self.opensearch.index(
            index=self.index_name,
            body={
                "text": text,
                "metadata": metadata,
                "embedding": embedding
            }
        )
```

---

### 4. Automated Training Trigger

**File**: `src/training/retraining_trigger.py`

**How it works:**

#### A. Scheduled Trigger (EventBridge)
1. Create EventBridge rule (line 40-45)
   - Schedule: Cron expression (e.g., weekly)
   - State: ENABLED
2. Add Lambda target (line 50-58)
   - Lambda function: `retraining-trigger`
3. EventBridge triggers Lambda at schedule

#### B. Lambda Handler
1. Lambda receives event (line 82)
2. Calls `check_retraining_conditions()` (line 97)
3. If conditions met → triggers training (line 99)
4. Returns status

**Key Code:**
```python
# Line 82-120: Lambda handler
def lambda_handler(self, event, context):
    if incremental_learning.check_retraining_conditions():
        model_path = incremental_learning.trigger_retraining()
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Retraining triggered",
                "model_path": model_path
            })
        }
```

#### C. Check Conditions
**File**: `src/training/incremental_learning.py`

**Method**: `check_retraining_conditions()` (line 50-68)

**Checks:**
1. **New Data Count** (line 58-61)
   - Counts new labeled data in S3
   - If >= threshold (e.g., 1000) → trigger
2. **Performance Degradation** (line 64-66)
   - Checks if accuracy dropped
   - If below threshold → trigger

**Key Code:**
```python
# Line 50-68: Check conditions
def check_retraining_conditions(self) -> bool:
    # Check new data
    new_data_count = self._count_new_labeled_data()
    if new_data_count >= pipeline_config.MIN_NEW_SAMPLES_FOR_RETRAIN:
        return True
    
    # Check performance
    if self._check_performance_degradation():
        return True
    
    return False
```

---

## 📁 Complete File Reference

| Component | File | Class/Method | Line |
|-----------|------|--------------|------|
| **Kinesis Ingestion** | `src/data_collection/kinesis_ingestion.py` | `KinesisIngestion.ingest_chat_message()` | 45 |
| **S3 Storage** | `src/data_collection/s3_storage.py` | `S3Storage.upload_data()` | 23 |
| **Data Versioning** | `src/data_collection/data_versioning.py` | `DataVersioning.track_dataset()` | 49 |
| **Data Prep (Fine-tuning)** | `src/models/fine_tuning/data_preparation.py` | `FineTuningDataPreparation.extract_qa_pairs()` | 13 |
| **OpenSearch Storage** | `src/models/rag/vector_store.py` | `VectorStore.add_documents()` | 50 |
| **Check Conditions** | `src/training/incremental_learning.py` | `IncrementalLearning.check_retraining_conditions()` | 50 |
| **Scheduled Trigger** | `src/training/retraining_trigger.py` | `RetrainingTrigger.create_scheduled_retraining()` | 19 |
| **Lambda Handler** | `src/training/retraining_trigger.py` | `RetrainingTrigger.lambda_handler()` | 82 |
| **Trigger Training** | `src/training/incremental_learning.py` | `IncrementalLearning.trigger_retraining()` | 82 |
| **API Entry Point** | `src/api/chat_endpoints.py` | `chat()` | 45 |

---

## 🎯 Summary

**Data Flow:**
1. **User Message** → API (`src/api/chat_endpoints.py:45`)
2. **Kinesis** → Stream (`src/data_collection/kinesis_ingestion.py:45`)
3. **S3 Archive** → Raw data (`src/data_collection/kinesis_ingestion.py:186`)
4. **Preprocessing** → Clean data (`src/models/fine_tuning/data_preparation.py:13`)
5. **OpenSearch** → Vector store (`src/models/rag/vector_store.py:50`)
6. **Check Conditions** → Retrain? (`src/training/incremental_learning.py:50`)
7. **EventBridge** → Schedule (`src/training/retraining_trigger.py:19`)
8. **Lambda** → Trigger (`src/training/retraining_trigger.py:82`)
9. **Training** → SageMaker (`src/training/incremental_learning.py:82`)

**Key Points:**
- Kinesis handles real-time streaming
- S3 stores raw and processed data
- Preprocessing extracts and cleans data
- OpenSearch stores embeddings for RAG
- EventBridge schedules retraining
- Lambda triggers training when conditions met

All code is in the `src/` directory, organized by module! 🚀


