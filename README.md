# Intelligent Customer Support Chatbot with Multi-Model Strategy
This project is part of my backend engineering portfolio. The backend architecture, API design, and implementation were designed and built by me while exploring and learning core backend development concepts.
##  Project Overview

An intelligent customer support chatbot that uses multiple LLM strategies (pre-trained, fine-tuned, RAG) based on query type. Includes sentiment analysis, intent classification, and automated escalation with full MLOps pipeline for continuous learning.

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              API Gateway / FastAPI Layer                      │
│              (Module 6: API & Integration)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Intent Classification & Router                   │
│              (Module 2: Intent Classification)               │
└──────┬───────────────┬───────────────┬───────────────────────┘
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│ Pre-trained │ │ Fine-tuned  │ │    RAG      │
│  (Bedrock)  │ │ (SageMaker) │ │ (OpenSearch)│
└─────────────┘ └─────────────┘ └─────────────┘
       │               │               │
┌──────┴───────────────┴───────────────┴───────┐
│         Response Aggregation & Scoring        │
└───────────────────────────────────────────────┘
```

##  Project Structure

```
Intelligent_customer_chatbot/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── docker-compose.yml
│
├── config/
│   ├── __init__.py
│   ├── aws_config.py
│   ├── model_config.py
│   └── pipeline_config.py
│
├── data/
│   ├── raw/                    # Raw data storage
│   ├── processed/              # Processed data
│   ├── labeled/                # Labeled datasets
│   └── .dvc/                   # DVC versioning
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_collection/        # Module 1
│   │   ├── __init__.py
│   │   ├── kinesis_ingestion.py
│   │   ├── s3_storage.py
│   │   ├── data_versioning.py
│   │   └── labeling_pipeline.py
│   │
│   ├── intent_classification/  # Module 2
│   │   ├── __init__.py
│   │   ├── model_training.py
│   │   ├── intent_classifier.py
│   │   ├── router.py
│   │   └── evaluation.py
│   │
│   ├── models/                 # Module 3 & 4
│   │   ├── __init__.py
│   │   ├── bedrock_client.py
│   │   ├── fine_tuning/
│   │   │   ├── __init__.py
│   │   │   ├── lora_trainer.py
│   │   │   ├── data_preparation.py
│   │   │   └── model_evaluator.py
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py
│   │   │   ├── retriever.py
│   │   │   └── rag_pipeline.py
│   │   └── model_router.py
│   │
│   ├── training/               # Module 4 & 5
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   ├── incremental_learning.py
│   │   ├── retraining_trigger.py
│   │   └── drift_detection.py
│   │
│   ├── api/                    # Module 6
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── chat_endpoints.py
│   │   ├── websocket_handler.py
│   │   └── integrations.py
│   │
│   ├── monitoring/             # Module 7
│   │   ├── __init__.py
│   │   ├── model_monitor.py
│   │   ├── performance_tracker.py
│   │   └── alerting.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── helpers.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_experimentation.ipynb
│   └── evaluation.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_data_collection.py
│   ├── test_intent_classification.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_integrations.py
│
├── infrastructure/
│   ├── terraform/              # Infrastructure as Code
│   ├── cloudformation/         # AWS CloudFormation templates
│   └── docker/
│       ├── Dockerfile.api
│       └── Dockerfile.training
│
├── cicd/
│   ├── .github/
│   │   └── workflows/
│   │       ├── ci.yml
│   │       ├── cd.yml
│   │       └── training_pipeline.yml
│   ├── sagemaker_pipelines/
│   │   ├── training_pipeline.py
│   │   ├── evaluation_pipeline.py
│   │   └── deployment_pipeline.py
│   └── scripts/
│       ├── deploy.sh
│       └── test.sh
│
└── docs/
    ├── aws_setup.md
    ├── deployment_guide.md
    ├── api_documentation.md
    └── architecture.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- AWS Account with appropriate permissions
- Docker (optional, for local development)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Intelligent_customer_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your AWS credentials and configuration

# Initialize DVC for data versioning
dvc init
dvc remote add -d s3-remote s3://your-bucket/dvc-storage
```

### AWS Setup

See [docs/aws_setup.md](docs/aws_setup.md) for detailed AWS service setup instructions.

### Running the Application

```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000

# Run training pipeline
python -m src.training.training_pipeline

# Run data ingestion
python -m src.data_collection.kinesis_ingestion
```

## 📚 Modules

### Module 1: Data Collection & Versioning
- Real-time data ingestion from Kinesis
- S3 storage with versioning
- SageMaker Ground Truth labeling
- DVC for data versioning

### Module 2: Intent Classification & Routing
- BERT-based intent classification
- Multi-class routing logic
- Real-time inference

### Module 3: Multi-Model Strategy
- Pre-trained models (AWS Bedrock)
- Fine-tuned domain models
- RAG system with OpenSearch

### Module 4: Fine-tuning Pipeline
- LoRA-based fine-tuning
- Domain-specific model training
- Hyperparameter optimization

### Module 5: Incremental Learning & Retraining
- Automated retraining triggers
- Concept drift detection
- Continuous model improvement

### Module 6: API & Integration Layer
- FastAPI REST API
- WebSocket support
- CRM integrations
- Escalation handling

### Module 7: CI/CD & MLOps Pipeline
- GitHub Actions CI/CD
- SageMaker Pipelines
- Model monitoring
- Automated deployments

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Monitoring

- CloudWatch dashboards for model performance
- SageMaker Model Monitor for drift detection
- Custom metrics for business KPIs

## 📄 License

MIT License

## 👤 Author:Ravinder Singh

Built for interview showcase demonstrating production ML systems expertise.

