# MLOps Pipeline Quick Reference

## 🎯 Pipeline Overview

```
Data → Training → Evaluation → Registry → Deployment → Monitoring → Retraining
```

## 📊 Data Pipeline

### Flow
```
Kinesis → S3 (Raw) → Labeling → S3 (Labeled) → DVC Versioning
```

### Key Commands
```bash
# Ingest data
python -m src.data_collection.kinesis_ingestion

# Create labeling job
python -m src.data_collection.labeling_pipeline

# Version data
dvc add data/labeled/v1/
dvc push
```

## 🎓 Training Pipeline

### Models Used

| Model | Purpose | Training Method | Location |
|-------|---------|----------------|----------|
| **BERT-base-uncased** | Intent Classification | Supervised (SageMaker) | `src/intent_classification/` |
| **Llama-2-7b + LoRA** | Domain-specific responses | Fine-tuning (LoRA) | `src/models/fine_tuning/` |
| **Sentence Transformers** | RAG embeddings | Pre-trained (no training) | `src/models/rag/` |

### Training Commands
```bash
# Train intent classifier
python -m src.training.training_pipeline \
    --model-type intent_classifier \
    --data-path s3://bucket/labeled/v1/ \
    --use-sagemaker

# Fine-tune domain model
python -m src.training.training_pipeline \
    --model-type fine_tuned \
    --domain billing \
    --data-path data/domain_specific/billing.json
```

## 🚀 Deployment Pipeline

### Flow
```
Model Artifacts → Registry → Evaluation → A/B Test → Production
```

### Deployment Commands
```bash
# Deploy model
python cicd/scripts/deploy_model.py \
    --model-uri s3://bucket/models/v1.1/ \
    --endpoint-name intent-classifier-prod \
    --compare-with current \
    --auto-approve-if-better
```

## 🔄 Retraining Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Scheduled** | Weekly (Sunday 2 AM) | EventBridge → Lambda → Training |
| **Data Threshold** | New data >= 1000 samples | Lambda checks → Training |
| **Performance** | Accuracy drop > 5% | Drift detection → Training |
| **Manual** | GitHub Actions dispatch | Manual training job |

## 📦 Versioning

### Model Versioning (SageMaker Registry)
```
Model Package Group: intent-classifier
├── v1.0 (Approved, Production)
├── v1.1 (Approved, Staging)
└── v1.2 (Pending, Testing)
```

### Data Versioning (DVC)
```bash
# Track dataset
dvc add data/labeled/v1/
git commit -m "Dataset v1"

# Switch versions
dvc checkout data/labeled/v1/  # Use v1
dvc checkout data/labeled/v2/  # Use v2
```

## 📈 Monitoring

### Metrics Tracked
- **Accuracy**: Model performance
- **Latency**: Response time (p95)
- **Cost**: Inference cost per request
- **Request Volume**: Requests per hour
- **Error Rate**: Failed requests

### CloudWatch Dashboard
```
Namespace: CustomerChatbot
Metrics:
  - model_accuracy
  - model_latency
  - model_cost
  - model_requests
  - data_drift_detected
  - concept_drift_detected
```

## 🔍 Drift Detection

### Types of Drift

1. **Data Drift**
   - Method: KS test, PSI
   - Threshold: 30% features drifted
   - Frequency: Hourly (SageMaker Model Monitor)

2. **Concept Drift**
   - Method: Performance degradation
   - Threshold: 5% accuracy drop
   - Frequency: Daily

3. **Model Drift**
   - Method: Prediction distribution change
   - Threshold: 10% average change
   - Frequency: Real-time

### Detection Code
```python
from src.monitoring.drift_detector import drift_detector

# Data drift
drift_results = drift_detector.detect_data_drift(
    reference_data=baseline_df,
    current_data=current_df
)

# Concept drift
concept_drift = drift_detector.detect_concept_drift(
    model_performance_history=[0.89, 0.90, 0.88, 0.85, 0.83]
)
```

## 🔄 CI/CD Integration

### CI Pipeline (GitHub Actions)
```yaml
Trigger: Push to main/develop
Steps:
  1. Code quality (Black, Flake8, MyPy)
  2. Unit tests
  3. Integration tests
  4. Build Docker image
```

### CD Pipeline (GitHub Actions)
```yaml
Trigger: Merge to main
Steps:
  1. Run full test suite
  2. Build production Docker
  3. Push to ECR
  4. Deploy to ECS
  5. Update SageMaker endpoints
  6. Health check
```

### Training Pipeline (GitHub Actions)
```yaml
Trigger: Schedule (weekly) OR Manual
Steps:
  1. Check retraining conditions
  2. Trigger SageMaker training
  3. Evaluate model
  4. Compare with current
  5. Register if improved
  6. Deploy if approved
```

## 📝 Key Files

| Component | File Location |
|-----------|---------------|
| **Data Collection** | `src/data_collection/` |
| **Training** | `src/training/training_pipeline.py` |
| **Intent Classifier** | `src/intent_classification/model_training.py` |
| **Fine-tuning** | `src/models/fine_tuning/lora_trainer.py` |
| **Deployment** | `cicd/scripts/deploy_model.py` |
| **Drift Detection** | `src/monitoring/drift_detector.py` |
| **Monitoring** | `src/monitoring/model_monitor.py` |
| **CI/CD** | `cicd/.github/workflows/` |

## 🎯 Complete Workflow Example

```python
# 1. Data Collection
kinesis_ingestion.ingest_chat_message(user_id, message, session_id)

# 2. Data Labeling (weekly batch)
labeling_pipeline.create_labeling_job(data_records)

# 3. Data Versioning
data_versioning.track_dataset("data/labeled/v2/", "intent-v2")

# 4. Training (triggered automatically)
training_pipeline.run_intent_classification_training(
    data_path="s3://bucket/labeled/v2/",
    use_sagemaker=True
)

# 5. Evaluation
metrics = intent_trainer.evaluate(model_path, test_data)

# 6. Model Registry
register_model(model_uri, metrics)

# 7. Deployment (if improved)
deploy_model(model_uri, endpoint_name)

# 8. Monitoring
model_monitor.setup_model_monitor(endpoint_name)

# 9. Drift Detection (hourly)
drift_results = drift_detector.detect_data_drift(baseline, current)

# 10. Retraining (if drift detected)
if drift_results["drift_detected"]:
    trigger_retraining()
```

## 🚨 Alerts & Notifications

### Alert Conditions
- **Performance Degradation**: Accuracy drop > 5%
- **Data Drift**: > 30% features drifted
- **High Latency**: p95 > 3 seconds
- **High Error Rate**: > 5% requests fail

### Alert Channels
- CloudWatch Alarms → SNS → Email/Slack
- SageMaker Model Monitor → SNS notifications

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| **Accuracy** | > 85% | 89% |
| **Latency (p95)** | < 3s | 1.8s |
| **Availability** | 99.9% | 99.95% |
| **Cost per Request** | < $0.01 | $0.008 |

---

**For detailed explanations, see [MLOPS_PIPELINE_DEEP_DIVE.md](MLOPS_PIPELINE_DEEP_DIVE.md)**

