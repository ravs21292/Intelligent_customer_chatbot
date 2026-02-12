# MLOps Pipeline Deep Dive

## 🎯 Complete MLOps Lifecycle Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline Architecture                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Data       │ --> │   Training   │ --> │  Deployment  │
│  Pipeline    │     │   Pipeline   │     │   Pipeline   │
└──────────────┘     └──────────────┘     └──────────────┘
      │                     │                     │
      │                     │                     │
      ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Versioning  │     │  Monitoring  │     │   Tracking   │
│   (DVC/S3)   │     │  (CloudWatch)│     │  (MLflow/    │
│              │     │              │     │   Registry)  │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📊 Data Pipeline: From Collection to Training

### 1. Data Collection Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Real-Time Data Collection Pipeline              │
└─────────────────────────────────────────────────────────────┘

User Chat Message
      │
      ▼
┌─────────────────┐
│  FastAPI API    │  Receives message
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Kinesis Stream  │  Real-time streaming
│ (customer-chat) │  Partitioned by user_id
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kinesis        │  Consumes in batches
│  Consumer       │  (Lambda/Kinesis Analytics)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  S3 Raw Data    │  Archived daily
│  s3://bucket/   │  Partitioned by date
│  raw/YYYY/MM/DD │  Format: JSON/Parquet
└─────────────────┘
```

**Code Implementation** (`src/data_collection/kinesis_ingestion.py`):

```python
# Step 1: User sends message via API
# Step 2: API ingests to Kinesis
def ingest_chat_message(user_id, message, session_id):
    record = {
        "user_id": user_id,
        "message": message,
        "session_id": session_id,
        "timestamp": time.time(),
        "metadata": {}
    }
    
    # Send to Kinesis
    kinesis.put_record(
        StreamName="customer-chat-stream",
        Data=json.dumps(record),
        PartitionKey=user_id  # Partition by user for ordering
    )
    
    # Archive to S3 (batch process)
    # Runs every hour via Lambda
    archive_to_s3(records, date_prefix="2024/01/15")
```

### 2. Data Labeling Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              Data Labeling with SageMaker Ground Truth       │
└─────────────────────────────────────────────────────────────┘

Unlabeled Data (S3)
      │
      ▼
┌─────────────────┐
│ Create Manifest │  JSONL format for Ground Truth
│ File            │  {"source": "text", "metadata": {...}}
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ground Truth    │  Labeling job created
│ Labeling Job    │  - Public workforce OR
└────────┬────────┘    - Private team
         │
         ▼
┌─────────────────┐
│ Workers Label   │  Intent classification
│ Data            │  - billing
└────────┬────────┘  - technical_support
         │           - product_inquiry
         │           - complaint
         ▼
┌─────────────────┐
│ Labeled Data    │  Output in S3
│ (S3)            │  s3://bucket/labeled/v1/
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DVC Versioning  │  Track labeled dataset
│                 │  dvc add data/labeled/v1/
└─────────────────┘
```

**Code Implementation** (`src/data_collection/labeling_pipeline.py`):

```python
# Create labeling job
def create_labeling_job(data_records):
    # 1. Create manifest file
    manifest_uri = create_manifest_file(data_records, "intent-labeling-v1")
    
    # 2. Create Ground Truth job
    job_arn = sagemaker.create_labeling_job(
        LabelingJobName="intent-classification-v1",
        InputConfig={
            "DataSource": {"S3DataSource": {"ManifestS3Uri": manifest_uri}}
        },
        OutputConfig={
            "S3OutputPath": "s3://bucket/labeled/intent/"
        },
        LabelCategoryConfig={
            "categories": [
                "billing", "technical_support", "product_inquiry",
                "complaint", "refund", "general_inquiry"
            ]
        }
    )
    
    # 3. Wait for completion
    wait_for_job_completion("intent-classification-v1")
    
    # 4. Download labeled data
    labeled_data = download_labeled_data(output_s3_uri)
    
    # 5. Version with DVC
    data_versioning.track_dataset("data/labeled/v1/", "intent-labeled-v1")
```

### 3. Data Versioning with DVC

**Why DVC?**
- Track data changes like code
- Reproduce experiments with exact data
- Share datasets across team
- S3 backend for large files

**Workflow**:

```bash
# 1. Add data to DVC
dvc add data/labeled/intent_dataset.json

# Creates: data/labeled/intent_dataset.json.dvc
# Tracks: file hash, size, S3 location

# 2. Commit to Git
git add data/labeled/intent_dataset.json.dvc
git commit -m "Add labeled dataset v1"

# 3. Push data to S3
dvc push  # Uploads to s3://bucket/dvc-storage/

# 4. Pull data (on another machine)
dvc pull  # Downloads from S3
```

**Code Implementation** (`src/data_collection/data_versioning.py`):

```python
# Track dataset version
def track_dataset(dataset_path, dataset_name, metadata):
    # Add to DVC
    subprocess.run(["dvc", "add", dataset_path])
    
    # Create metadata
    metadata_file = f"{dataset_path}.meta"
    with open(metadata_file, "w") as f:
        json.dump({
            "dataset_name": dataset_name,
            "version": "v1",
            "record_count": len(dataset),
            "created_at": datetime.utcnow().isoformat(),
            "labels": ["billing", "technical_support", ...]
        }, f)
    
    # Commit to DVC
    subprocess.run(["dvc", "commit", "-f", f"{dataset_path}.dvc"])
    
    # Push to remote
    subprocess.run(["dvc", "push", f"{dataset_path}.dvc"])
```

---

## 🎓 Training Pipeline: Model Development

### Models Used in This Project

#### 1. Intent Classification Model (BERT)

**Model**: `bert-base-uncased`
- **Purpose**: Classify user messages into 8 intent categories
- **Input**: Raw text message
- **Output**: Intent label + confidence score
- **Training**: Supervised learning on labeled data

**Architecture**:
```
Input Text: "I need help with my billing"
    │
    ▼
┌─────────────────┐
│ BERT Tokenizer  │  Tokenize to subwords
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BERT Encoder    │  Extract features
│ (12 layers)     │  [CLS] token = sentence embedding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification  │  8 classes
│ Head            │  Softmax → probabilities
└────────┬────────┘
         │
         ▼
Output: {"intent": "billing", "confidence": 0.92}
```

#### 2. Fine-Tuned Domain Models (LoRA)

**Base Model**: `meta-llama/Llama-2-7b-chat-hf`
- **Purpose**: Domain-specific responses (billing, technical, etc.)
- **Method**: LoRA (Low-Rank Adaptation)
- **Why LoRA?**: 10x faster, 3x less memory, only train 1% of parameters

**LoRA Architecture**:
```
Base Model (7B parameters)
    │
    ├─── Frozen Layers (99%)
    │    - Embeddings
    │    - Most attention layers
    │
    └─── Trainable LoRA Layers (1%)
         - q_proj: Query projection
         - v_proj: Value projection
         - LoRA rank: 8
         - LoRA alpha: 16
```

#### 3. RAG System (No Training)

**Components**:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: OpenSearch with k-NN
- **LLM**: AWS Bedrock (Claude) for generation

**No Training Needed**: Uses pre-trained embeddings + pre-trained LLM

### Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Complete Training Pipeline                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Labeled Data   │  s3://bucket/labeled/v1/
│ (DVC Versioned)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Prep       │  Split: Train/Val/Test (70/15/15)
│                 │  Format: BERT tokenization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SageMaker       │  Training job
│ Training Job    │  - Instance: ml.g4dn.xlarge
│                 │  - Epochs: 3
│                 │  - Batch: 32
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Artifacts │  Saved to S3
│ s3://bucket/    │  - model.tar.gz
│ models/v1/      │  - tokenizer/
│                 │  - config.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluation      │  Test set metrics
│                 │  - Accuracy: 0.89
│                 │  - F1: 0.87
│                 │  - Confusion matrix
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │  Register if improved
│ (SageMaker)     │  - Version: v1.0
│                 │  - Metrics tracked
└─────────────────┘
```

**Code Implementation** (`src/intent_classification/model_training.py`):

```python
def train_intent_classifier(data_path, use_sagemaker=True):
    # 1. Prepare data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        prepare_data(data_path, test_size=0.15, val_size=0.15)
    
    if use_sagemaker:
        # 2. Upload data to S3
        train_s3_uri = upload_to_s3(train_data, "s3://bucket/training/train/")
        val_s3_uri = upload_to_s3(val_data, "s3://bucket/training/val/")
        
        # 3. Create SageMaker training job
        estimator = HuggingFace(
            entry_point="train.py",
            source_dir="src/intent_classification",
            instance_type="ml.g4dn.xlarge",
            role=SAGEMAKER_ROLE_ARN,
            transformers_version="4.26",
            pytorch_version="1.13",
            hyperparameters={
                "model_name": "bert-base-uncased",
                "num_labels": 8,
                "epochs": 3,
                "batch_size": 32,
                "learning_rate": 2e-5
            }
        )
        
        # 4. Start training
        estimator.fit({
            "training": train_s3_uri,
            "validation": val_s3_uri
        }, job_name=f"intent-classifier-{timestamp}")
        
        # 5. Model artifacts saved to S3
        model_uri = estimator.model_data
        # s3://bucket/models/intent-classifier-123456/model.tar.gz
        
        return model_uri
    else:
        # Local training
        model_path = train_local(train_texts, train_labels, ...)
        return model_path
```

**Training Script** (`src/intent_classification/train.py`):

```python
# This runs inside SageMaker container
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

def main():
    # Load hyperparameters
    model_name = os.environ["SM_HP_MODEL_NAME"]
    num_labels = int(os.environ["SM_HP_NUM_LABELS"])
    
    # Load data
    train_dataset = load_dataset("json", data_files=os.environ["SM_CHANNEL_TRAINING"])
    val_dataset = load_dataset("json", data_files=os.environ["SM_CHANNEL_VALIDATION"])
    
    # Initialize model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.environ["SM_MODEL_DIR"],  # Saved to S3
        num_train_epochs=3,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    trainer.save_model()  # Saves to SM_MODEL_DIR → S3
```

### Fine-Tuning Pipeline (LoRA)

**Code Implementation** (`src/models/fine_tuning/lora_trainer.py`):

```python
def train_fine_tuned_model(domain, training_data):
    # 1. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16
    )
    
    # 2. Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,              # Rank (low-rank dimension)
        lora_alpha=16,    # Scaling factor
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Only train these
    )
    
    # 3. Apply LoRA (only 1% parameters trainable)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output: trainable params: 8M / 7B (0.1%)
    
    # 4. Train
    trainer = Trainer(
        model=model,
        train_dataset=training_data,
        args=TrainingArguments(
            output_dir=f"models/fine_tuned/{domain}",
            num_train_epochs=3,
            per_device_train_batch_size=4
        )
    )
    
    trainer.train()
    
    # 5. Save (only LoRA weights, not full model)
    model.save_pretrained(f"models/fine_tuned/{domain}")
    # Saves adapter weights (~32MB vs 14GB for full model)
```

---

## 🚀 Deployment Pipeline: From Model to Production

### Deployment Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Model Deployment Pipeline                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Model Artifacts │  s3://bucket/models/v1.0/
│ (S3)            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │  Register model version
│ (SageMaker)    │  - Version: v1.0
│                 │  - Metrics: accuracy=0.89
│                 │  - Approval: Pending
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ A/B Testing     │  Compare with current model
│                 │  - 10% traffic to new model
│                 │  - Monitor metrics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SageMaker       │  Create endpoint
│ Endpoint        │  - Instance: ml.m5.xlarge
│                 │  - Auto-scaling: 1-5 instances
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ API Integration  │  Update API to use endpoint
│                 │  - Update endpoint name
│                 │  - Health check
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Production      │  Model serving live
│                 │  - Real-time inference
│                 │  - Monitoring enabled
└─────────────────┘
```

**Code Implementation** (`src/intent_classification/deployment.py`):

```python
def deploy_model(model_uri, endpoint_name):
    # 1. Create model
    model = Model(
        model_data=model_uri,
        role=SAGEMAKER_ROLE_ARN,
        framework_version="1.13",
        py_version="py39"
    )
    
    # 2. Register in model registry
    model_package = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        model_package_group_name="intent-classifier",
        approval_status="PendingManualApproval"  # Requires approval
    )
    
    # 3. Approve model (manual or automated)
    # If metrics improved, auto-approve
    if evaluate_model(model_uri) > current_model_metrics:
        approve_model(model_package.model_package_arn)
    
    # 4. Create endpoint configuration
    endpoint_config = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name,
        auto_scaling_config={
            "MinCapacity": 1,
            "MaxCapacity": 5,
            "TargetValue": 70.0  # CPU utilization
        }
    )
    
    # 5. Update API configuration
    update_api_endpoint(endpoint_name)
    
    return endpoint_name
```

### CI/CD Integration

**GitHub Actions Workflow** (`cicd/.github/workflows/cd.yml`):

```yaml
name: Deploy Model

on:
  push:
    branches: [ main ]
    paths:
      - 'src/intent_classification/**'
      - 'src/models/**'

jobs:
  deploy-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Train Model
        run: |
          python -m src.training.training_pipeline \
            --model-type intent_classifier \
            --use-sagemaker
      
      - name: Evaluate Model
        run: |
          python -m src.intent_classification.evaluation
      
      - name: Deploy if Improved
        run: |
          python cicd/scripts/deploy_model.py \
            --compare-with current \
            --auto-approve-if-better
```

---

## 🔄 Retraining Pipeline: Continuous Improvement

### When Retraining is Triggered

```
┌─────────────────────────────────────────────────────────────┐
│              Retraining Triggers                              │
└─────────────────────────────────────────────────────────────┘

1. Scheduled Retraining
   └── Weekly (Sunday 2 AM)
       └── EventBridge → Lambda → SageMaker Training

2. Data Threshold
   └── New labeled data >= 1000 samples
       └── Lambda checks → Triggers training

3. Performance Degradation
   └── Accuracy drops > 5%
       └── Drift detection → Triggers training

4. Manual Trigger
   └── GitHub Actions workflow_dispatch
       └── Manual training job
```

**Code Implementation** (`src/training/retraining_trigger.py`):

```python
def lambda_handler(event, context):
    """Lambda function triggered by EventBridge"""
    
    # 1. Check retraining conditions
    conditions = {
        "new_data": check_new_data_threshold(),
        "performance": check_performance_degradation(),
        "drift": check_data_drift()
    }
    
    # 2. If any condition met, trigger training
    if any(conditions.values()):
        # Get latest data
        latest_data = get_latest_labeled_data()
        
        # Start training job
        job_name = f"retrain-{int(time.time())}"
        training_pipeline.run_intent_classification_training(
            data_path=latest_data,
            use_sagemaker=True,
            job_name=job_name
        )
        
        # 3. Wait for completion (async)
        # Use Step Functions or EventBridge for orchestration
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Retraining triggered",
                "job_name": job_name,
                "conditions": conditions
            })
        }
    
    return {"statusCode": 200, "body": "Conditions not met"}
```

**EventBridge Rule**:

```python
# Create scheduled rule
eventbridge.put_rule(
    Name="weekly-retraining",
    ScheduleExpression="cron(0 2 ? * SUN)",  # Sunday 2 AM
    State="ENABLED"
)

# Add Lambda target
eventbridge.put_targets(
    Rule="weekly-retraining",
    Targets=[{
        "Id": "1",
        "Arn": "arn:aws:lambda:region:account:function:retraining-trigger"
    }]
)
```

### Retraining Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              Retraining Workflow                              │
└─────────────────────────────────────────────────────────────┘

Trigger Event
    │
    ▼
┌─────────────────┐
│ Collect New Data│  From Kinesis → S3 → Labeled
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Merge Datasets  │  Old + New data
│                 │  Maintain train/val/test split
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train New Model │  SageMaker training job
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluate       │  Compare with current model
│                 │  - Accuracy
│                 │  - F1 score
│                 │  - Per-class metrics
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Better?    Worse?
    │         │
    │         └───► Keep current model
    │              Log metrics
    │
    ▼
┌─────────────────┐
│ Register Model  │  New version in registry
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ A/B Test        │  10% traffic to new model
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Pass?      Fail?
    │         │
    │         └───► Rollback to previous
    │
    ▼
┌─────────────────┐
│ Full Deployment │  100% traffic to new model
└─────────────────┘
```

---

## 📦 Model Versioning

### SageMaker Model Registry

**Purpose**: Track model versions, metrics, and approvals

**Structure**:
```
Model Package Group: intent-classifier
├── Model Package v1.0
│   ├── Model URI: s3://bucket/models/v1.0/
│   ├── Metrics: {accuracy: 0.89, f1: 0.87}
│   ├── Training Data: s3://bucket/labeled/v1/
│   ├── Status: Approved
│   └── Endpoint: intent-classifier-prod
│
├── Model Package v1.1
│   ├── Model URI: s3://bucket/models/v1.1/
│   ├── Metrics: {accuracy: 0.91, f1: 0.89}
│   ├── Training Data: s3://bucket/labeled/v2/
│   ├── Status: Approved
│   └── Endpoint: intent-classifier-prod (updated)
│
└── Model Package v1.2
    ├── Model URI: s3://bucket/models/v1.2/
    ├── Metrics: {accuracy: 0.88, f1: 0.86}
    ├── Training Data: s3://bucket/labeled/v3/
    ├── Status: Rejected (worse than v1.1)
    └── Endpoint: (not deployed)
```

**Code Implementation**:

```python
def register_model(model_uri, metrics, training_data_uri):
    """Register model in SageMaker Model Registry"""
    
    # Create model package
    model_package = sagemaker_client.create_model_package(
        ModelPackageGroupName="intent-classifier",
        ModelPackageDescription=f"Intent classifier trained on {training_data_uri}",
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "Accuracy": {"Value": metrics["accuracy"]},
                    "F1Score": {"Value": metrics["f1"]}
                }
            }
        },
        SourceAlgorithmSpecification={
            "SourceAlgorithms": [{
                "AlgorithmName": "BERT",
                "ModelDataUrl": model_uri
            }]
        },
        MetadataProperties={
            "TrainingData": training_data_uri,
            "TrainingJobName": training_job_name,
            "TrainingTime": datetime.utcnow().isoformat()
        }
    )
    
    # Auto-approve if better than current
    current_model = get_current_production_model()
    if metrics["accuracy"] > current_model["accuracy"]:
        approve_model_package(model_package["ModelPackageArn"])
    
    return model_package
```

### Model Versioning Best Practices

1. **Semantic Versioning**: v1.0.0 (major.minor.patch)
2. **Metadata Tracking**: Training data, hyperparameters, metrics
3. **Lineage**: Track which data → which model → which endpoint
4. **Rollback**: Keep previous versions for quick rollback

---

## 📊 Data Versioning with DVC

### DVC Workflow

```bash
# 1. Initial dataset
dvc add data/labeled/v1/intent_dataset.json
# Creates: data/labeled/v1/intent_dataset.json.dvc
# Tracks: hash, size, S3 location

# 2. Commit to Git
git add data/labeled/v1/intent_dataset.json.dvc
git commit -m "Add labeled dataset v1 (1000 samples)"

# 3. Push to S3
dvc push
# Uploads to: s3://bucket/dvc-storage/

# 4. New version of data
# Add more samples, relabel, etc.
dvc add data/labeled/v2/intent_dataset.json
git commit -m "Add labeled dataset v2 (2000 samples)"

# 5. Compare versions
dvc diff data/labeled/v1/ data/labeled/v2/
# Shows: Added 1000 samples, changed 50 labels

# 6. Reproduce experiment
dvc checkout data/labeled/v1/  # Switch to v1
python train.py  # Train with v1 data
dvc checkout data/labeled/v2/  # Switch to v2
python train.py  # Train with v2 data
```

**Code Integration** (`src/data_collection/data_versioning.py`):

```python
def create_data_snapshot(datasets, snapshot_name):
    """Create snapshot of multiple datasets"""
    
    snapshot_metadata = {
        "snapshot_name": snapshot_name,
        "created_at": datetime.utcnow().isoformat(),
        "datasets": {
            name: {
                "path": path,
                "dvc_hash": get_dvc_hash(path),
                "record_count": count_records(path)
            }
            for name, path in datasets.items()
        }
    }
    
    # Save metadata
    snapshot_file = f"data/snapshots/{snapshot_name}.json"
    with open(snapshot_file, "w") as f:
        json.dump(snapshot_metadata, f)
    
    # Track with DVC
    data_versioning.track_dataset(snapshot_file, snapshot_name)
    
    return snapshot_file
```

---

## 📈 Performance Tracking & Monitoring

### Metrics Collection

**CloudWatch Metrics** (`src/utils/metrics.py`):

```python
class MetricsCollector:
    def track_model_performance(self, model_name, accuracy, latency, cost):
        """Track model performance metrics"""
        
        # Accuracy metric
        self.put_metric(
            "model_accuracy",
            accuracy,
            unit="Percent",
            dimensions={"model": model_name, "version": "v1.1"}
        )
        
        # Latency metric
        self.put_metric(
            "model_latency",
            latency,
            unit="Seconds",
            dimensions={"model": model_name}
        )
        
        # Cost metric
        self.put_metric(
            "model_cost",
            cost,
            unit="None",
            dimensions={"model": model_name}
        )
        
        # Request count
        self.put_metric(
            "model_requests",
            1,
            dimensions={"model": model_name, "intent": intent}
        )
```

### Performance Dashboard

**CloudWatch Dashboard**:

```
┌─────────────────────────────────────────────────┐
│         Model Performance Dashboard              │
├─────────────────────────────────────────────────┤
│                                                   │
│  Accuracy Over Time                              │
│  ┌─────────────────────────────────────────┐   │
│  │ 0.95 ┤                                    │   │
│  │ 0.90 ┤     ●───●───●                      │   │
│  │ 0.85 ┤   ●─●       ●───●                  │   │
│  │ 0.80 ┤ ●─●             ●                  │   │
│  │      └───────────────────────────────────┘   │
│  │      v1.0  v1.1  v1.2  v1.3  v1.4          │
│                                                   │
│  Latency (p95)                                   │
│  ┌─────────────────────────────────────────┐   │
│  │ 2.0s ┤                                    │   │
│  │ 1.5s ┤     ●───●───●                      │   │
│  │ 1.0s ┤   ●─●       ●───●                  │   │
│  │ 0.5s ┤ ●─●             ●                  │   │
│  │      └───────────────────────────────────┘   │
│                                                   │
│  Request Volume                                  │
│  ┌─────────────────────────────────────────┐   │
│  │ 10K ┤                                    │   │
│  │  8K ┤     ●───●───●                      │   │
│  │  6K ┤   ●─●       ●───●                  │   │
│  │  4K ┤ ●─●             ●                  │   │
│  │      └───────────────────────────────────┘   │
│                                                   │
└─────────────────────────────────────────────────┘
```

### Performance Tracking Code

```python
def track_performance(model_name, predictions, true_labels):
    """Track model performance after deployment"""
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    
    # Track in CloudWatch
    metrics_collector.track_model_performance(
        model_name=model_name,
        accuracy=accuracy,
        latency=avg_latency,
        cost=estimated_cost
    )
    
    # Store in database for historical tracking
    db.store_metrics({
        "model": model_name,
        "timestamp": datetime.utcnow(),
        "accuracy": accuracy,
        "f1": f1,
        "sample_count": len(predictions)
    })
    
    # Check for degradation
    if accuracy < baseline_accuracy - 0.05:  # 5% drop
        alert_manager.send_alert(
            "performance_degradation",
            f"Model {model_name} accuracy dropped to {accuracy}",
            severity="high"
        )
```

---

## 🔍 Drift Detection

### Types of Drift

1. **Data Drift**: Input data distribution changes
2. **Concept Drift**: Relationship between input and output changes
3. **Model Drift**: Model performance degrades over time

### Data Drift Detection

**Implementation** (`src/training/drift_detection.py`):

```python
def detect_data_drift(reference_data, current_data):
    """Detect data drift using statistical tests"""
    
    # 1. Extract features
    reference_features = extract_features(reference_data)
    current_features = extract_features(current_data)
    
    # 2. Statistical tests
    drift_results = {}
    
    # Kolmogorov-Smirnov test (for continuous features)
    for feature in continuous_features:
        ks_stat, p_value = ks_2samp(
            reference_features[feature],
            current_features[feature]
        )
        drift_results[feature] = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": p_value < 0.05
        }
    
    # Population Stability Index (PSI)
    for feature in categorical_features:
        psi = calculate_psi(
            reference_features[feature],
            current_features[feature]
        )
        drift_results[feature] = {
            "psi": psi,
            "drift_detected": psi > 0.2  # Threshold
        }
    
    # 3. Overall drift score
    overall_drift = sum(
        1 for r in drift_results.values() if r["drift_detected"]
    ) / len(drift_results)
    
    return {
        "drift_detected": overall_drift > 0.3,  # 30% features drifted
        "drift_score": overall_drift,
        "feature_drift": drift_results
    }
```

### Concept Drift Detection

```python
def detect_concept_drift(model_performance_history):
    """Detect concept drift from performance degradation"""
    
    if len(model_performance_history) < 10:
        return {"drift_detected": False}
    
    # Recent performance (last 5 days)
    recent_avg = np.mean(model_performance_history[-5:])
    
    # Baseline performance (first 5 days)
    baseline_avg = np.mean(model_performance_history[:5])
    
    # Degradation
    degradation = baseline_avg - recent_avg
    
    # Drift threshold: 5% accuracy drop
    drift_detected = degradation > 0.05
    
    return {
        "drift_detected": drift_detected,
        "degradation": degradation,
        "baseline": baseline_avg,
        "recent": recent_avg,
        "threshold": 0.05
    }
```

### SageMaker Model Monitor

**Setup** (`src/monitoring/model_monitor.py`):

```python
def setup_model_monitor(endpoint_name):
    """Set up SageMaker Model Monitor"""
    
    # 1. Create baseline (from training data)
    baseline_data = s3://bucket/training/baseline/
    baseline_statistics = create_baseline_statistics(baseline_data)
    
    # 2. Create monitoring schedule
    monitor = DataQualityMonitor(
        role=SAGEMAKER_ROLE_ARN,
        instance_count=1,
        instance_type="ml.t3.medium"
    )
    
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{endpoint_name}-monitor",
        endpoint_input=endpoint_name,
        output_s3_uri="s3://bucket/monitoring/output/",
        statistics=baseline_statistics,
        constraints=None,  # Will be generated from baseline
        schedule_cron_expression="cron(0 * * * ? *)"  # Hourly
    )
    
    return monitor
```

**Monitoring Schedule**:
- Runs every hour
- Compares current data with baseline
- Detects violations (data drift)
- Sends alerts if drift detected

### Drift Detection Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              Drift Detection Workflow                       │
└─────────────────────────────────────────────────────────────┘

Hourly Monitoring (SageMaker Model Monitor)
    │
    ▼
┌─────────────────┐
│ Collect Data    │  Last hour's predictions
│ from Endpoint   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compare with    │  Statistical tests
│ Baseline        │  - KS test
│                 │  - PSI
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Drift?     No Drift
    │         │
    │         └───► Continue monitoring
    │
    ▼
┌─────────────────┐
│ Alert Team      │  SNS notification
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trigger         │  Option 1: Retrain
│ Action          │  Option 2: Investigate
│                 │  Option 3: Update baseline
└─────────────────┘
```

---

## 🔄 Complete MLOps Pipeline Integration

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────┐
│         Complete MLOps Pipeline (CI/CD + Training)          │
└─────────────────────────────────────────────────────────────┘

Developer Push Code
    │
    ▼
┌─────────────────┐
│ CI Pipeline     │  GitHub Actions
│ (GitHub)        │  - Lint, test, build
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Pass?      Fail?
    │         │
    │         └───► Fix and retry
    │
    ▼
Merge to Main
    │
    ▼
┌─────────────────┐
│ CD Pipeline     │  Auto-deploy API
│ (GitHub)        │  - Build Docker
│                 │  - Deploy to ECS
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Pipeline   │  Continuous
│                 │  - Kinesis → S3
│                 │  - Labeling
│                 │  - DVC versioning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retraining      │  Triggered by:
│ Trigger         │  - Schedule (weekly)
│                 │  - Data threshold
│                 │  - Performance drop
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training        │  SageMaker
│ Pipeline        │  - Data prep
│                 │  - Model training
│                 │  - Evaluation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │  Version control
│                 │  - Register model
│                 │  - Compare metrics
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Better?    Worse?
    │         │
    │         └───► Reject, keep current
    │
    ▼
┌─────────────────┐
│ A/B Testing     │  Gradual rollout
│                 │  - 10% → 50% → 100%
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Deployment      │  Update endpoint
│                 │  - SageMaker endpoint
│                 │  - API configuration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Monitoring      │  Continuous
│                 │  - Performance tracking
│                 │  - Drift detection
│                 │  - Alerting
└─────────────────┘
         │
         └───► Loop back to retraining trigger
```

---

## 📝 Summary

### Key Components

1. **Data Pipeline**: Kinesis → S3 → Labeling → DVC versioning
2. **Training Pipeline**: SageMaker training jobs (BERT + LoRA)
3. **Deployment Pipeline**: Model Registry → A/B Testing → Production
4. **Retraining**: Automated triggers (schedule, data threshold, performance)
5. **Versioning**: Model Registry (SageMaker) + DVC (data)
6. **Monitoring**: CloudWatch metrics + SageMaker Model Monitor
7. **Drift Detection**: Statistical tests + performance tracking

### Models Used

1. **Intent Classifier**: BERT-base-uncased (8 classes)
2. **Fine-tuned Models**: Llama-2-7b with LoRA (domain-specific)
3. **RAG**: Sentence transformers + Bedrock (no training)

### Automation Level

- ✅ **Fully Automated**: Data collection, labeling (partial), retraining triggers
- ✅ **Semi-Automated**: Model approval (can be automated with thresholds)
- ✅ **Manual**: Initial labeling, major architecture changes

This MLOps pipeline ensures **continuous improvement** with minimal manual intervention! 🚀

