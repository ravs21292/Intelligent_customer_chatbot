# Quick Start Guide

Get the Intelligent Customer Support Chatbot running in 5 minutes!

## Prerequisites

- Python 3.9+
- AWS Account (for full functionality)
- AWS CLI configured

## Step 1: Clone and Setup

```bash
# Navigate to project directory
cd Intelligent_customer_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your AWS credentials
# Minimum required:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_REGION
```

## Step 3: Initialize Data Versioning (Optional)

```bash
# Initialize DVC
dvc init

# Configure remote (if using S3)
dvc remote add -d s3-remote s3://your-bucket/dvc-storage
```

## Step 4: Run the API

```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000
```

## Step 5: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I need help with my account",
    "user_id": "test-user-123"
  }'
```

## Local Development (Without AWS)

For local development without AWS services:

1. **Mock AWS Services**: Use local mocks for Kinesis, S3, etc.
2. **Use Local Models**: Load models locally instead of SageMaker
3. **Skip Bedrock**: Use local Hugging Face models

Example mock setup:
```python
# In config/aws_config.py, add mock mode
if os.getenv("USE_MOCKS", "False").lower() == "true":
    # Use local mocks
    pass
```

## Next Steps

1. **Set up AWS Services**: See [docs/aws_setup.md](docs/aws_setup.md)
2. **Train Models**: See [docs/deployment_guide.md](docs/deployment_guide.md)
3. **Deploy**: Follow deployment guide

## Troubleshooting

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### AWS Credentials
```bash
# Verify AWS credentials
aws sts get-caller-identity
```

### Port Already in Use
```bash
# Use different port
uvicorn src.api.main:app --reload --port 8001
```

## Development Tips

1. **Use Hot Reload**: API auto-reloads on code changes
2. **Check Logs**: Monitor console for errors
3. **Test Incrementally**: Test each module separately
4. **Use Debug Mode**: Set `API_DEBUG=True` in `.env`

## Need Help?

- Check [docs/aws_setup.md](docs/aws_setup.md) for AWS setup
- See [docs/deployment_guide.md](docs/deployment_guide.md) for deployment
- Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture overview

