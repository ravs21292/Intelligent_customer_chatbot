# Deployment Guide

Complete guide for deploying the Intelligent Customer Support Chatbot.

## Pre-Deployment Checklist

- [ ] AWS services configured (see [aws_setup.md](aws_setup.md))
- [ ] Environment variables configured in `.env`
- [ ] Dependencies installed
- [ ] Tests passing
- [ ] Data versioning initialized

## Local Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd Intelligent_customer_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS credentials
```

### 2. Initialize Data Versioning

```bash
# Initialize DVC
dvc init
dvc remote add -d s3-remote s3://your-bucket/dvc-storage

# Configure AWS credentials for DVC
aws configure
```

### 3. Run Local Development Server

```bash
# Start API server
uvicorn src.api.main:app --reload --port 8000

# In another terminal, test the API
curl http://localhost:8000/health
```

## Training Models

### Intent Classification Model

```bash
# Prepare training data
python -m src.data_collection.labeling_pipeline

# Train locally
python -m src.training.training_pipeline \
    --model-type intent_classifier \
    --data-path data/labeled/intent_dataset.json \
    --output-dir models/intent_classifier

# Or train on SageMaker
python -m src.training.training_pipeline \
    --model-type intent_classifier \
    --data-path s3://bucket/training-data \
    --use-sagemaker
```

### Fine-tuned Domain Models

```bash
# Fine-tune for billing domain
python -m src.training.training_pipeline \
    --model-type fine_tuned \
    --domain billing \
    --data-path data/domain_specific/billing.json
```

## Deploying to AWS

### Option 1: EC2/ECS Deployment

```bash
# Build Docker image
docker build -t customer-chatbot-api:latest \
    -f infrastructure/docker/Dockerfile.api .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag customer-chatbot-api:latest \
    YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/customer-chatbot-api:latest

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/customer-chatbot-api:latest

# Deploy to ECS
aws ecs update-service \
    --cluster customer-chatbot-cluster \
    --service customer-chatbot-api \
    --force-new-deployment
```

### Option 2: Lambda + API Gateway

```bash
# Package for Lambda
zip -r deployment-package.zip src/ config/ -x "*.pyc" "__pycache__/*"

# Update Lambda function
aws lambda update-function-code \
    --function-name customer-chatbot-api \
    --zip-file fileb://deployment-package.zip
```

### Option 3: SageMaker Endpoints

```bash
# Deploy intent classifier endpoint
python -m src.intent_classification.deployment \
    --model-path s3://bucket/models/intent_classifier \
    --endpoint-name intent-classifier-endpoint
```

## Setting Up CI/CD

### GitHub Actions

1. Add AWS credentials to GitHub Secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. Push to main branch triggers deployment

### Manual Deployment

```bash
# Run deployment script
./cicd/scripts/deploy.sh production
```

## Post-Deployment

### 1. Verify Deployment

```bash
# Health check
curl https://your-api-endpoint.com/health

# Test chat endpoint
curl -X POST https://your-api-endpoint.com/api/v1/chat \
    -H "Content-Type: application/json" \
    -d '{
        "message": "Hello, I need help with my account",
        "user_id": "test-user-123"
    }'
```

### 2. Set Up Monitoring

```bash
# Create CloudWatch dashboard
python -m src.monitoring.setup_dashboard

# Enable model monitoring
python -m src.monitoring.model_monitor \
    --endpoint-name intent-classifier-endpoint
```

### 3. Initialize Knowledge Base (RAG)

```bash
# Index documents
python -m src.models.rag.index_documents \
    --data-path data/knowledge_base/ \
    --index-name customer-support-kb
```

## Production Checklist

- [ ] Environment variables secured (use AWS Secrets Manager)
- [ ] API rate limiting configured
- [ ] Monitoring and alerting set up
- [ ] Backup and disaster recovery plan
- [ ] Load testing completed
- [ ] Security audit completed
- [ ] Documentation updated
- [ ] Team trained on operations

## Rollback Procedure

```bash
# Rollback to previous model version
aws sagemaker update-endpoint \
    --endpoint-name intent-classifier-endpoint \
    --endpoint-config-name previous-config

# Rollback API deployment
./cicd/scripts/rollback.sh production
```

## Scaling

### Horizontal Scaling

- Increase ECS service count
- Add Kinesis shards
- Scale OpenSearch cluster

### Vertical Scaling

- Upgrade SageMaker instance types
- Increase Lambda memory
- Upgrade OpenSearch instance types

## Monitoring

Access monitoring dashboards:
- CloudWatch: AWS Console → CloudWatch → Dashboards
- SageMaker Model Monitor: SageMaker Console → Model Monitor
- Application Logs: CloudWatch Logs

## Support

For issues or questions:
1. Check logs in CloudWatch
2. Review monitoring dashboards
3. Check GitHub Issues
4. Contact DevOps team

