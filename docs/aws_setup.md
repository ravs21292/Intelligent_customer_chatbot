# AWS Setup Guide

This guide provides step-by-step instructions for setting up AWS services required for the Intelligent Customer Support Chatbot.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured with credentials
- Python 3.9+ installed
- Terraform (optional, for infrastructure as code)

## Required AWS Services

### 1. S3 Buckets

Create three S3 buckets for data, models, and logs:

```bash
# Data bucket
aws s3 mb s3://customer-chatbot-data --region us-east-1

# Models bucket
aws s3 mb s3://customer-chatbot-models --region us-east-1

# Logs bucket
aws s3 mb s3://customer-chatbot-logs --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket customer-chatbot-data \
    --versioning-configuration Status=Enabled
```

**Update `.env` file:**
```
S3_BUCKET_DATA=customer-chatbot-data
S3_BUCKET_MODELS=customer-chatbot-models
S3_BUCKET_LOGS=customer-chatbot-logs
```

### 2. Kinesis Data Stream

Create Kinesis stream for real-time data ingestion:

```bash
aws kinesis create-stream \
    --stream-name customer-chat-stream \
    --shard-count 2 \
    --region us-east-1
```

**Or use Python:**
```python
from src.data_collection.kinesis_ingestion import kinesis_ingestion
kinesis_ingestion.create_stream_if_not_exists(shard_count=2)
```

### 3. SageMaker Setup

#### Create IAM Role for SageMaker

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
    --role-name SageMakerExecutionRole \
    --assume-role-policy-document file://trust-policy.json

# Attach policies
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

**Get Role ARN and update `.env`:**
```
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
```

### 4. AWS Bedrock

#### Enable Bedrock Access

1. Go to AWS Bedrock console
2. Navigate to "Model access"
3. Request access to Claude models (anthropic.claude-v2)
4. Wait for approval (usually instant)

**Update `.env`:**
```
BEDROCK_MODEL_ID=anthropic.claude-v2
BEDROCK_REGION=us-east-1
```

### 5. OpenSearch (for RAG)

#### Create OpenSearch Domain

```bash
aws opensearch create-domain \
    --domain-name customer-support-kb \
    --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
    --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=20 \
    --access-policies '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": "es:*",
            "Resource": "arn:aws:es:us-east-1:YOUR_ACCOUNT_ID:domain/customer-support-kb/*"
        }]
    }'
```

**Wait for domain to be active, then update `.env`:**
```
OPENSEARCH_ENDPOINT=https://search-customer-support-kb-xxxxx.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX_NAME=customer-support-kb
```

### 6. API Gateway & Lambda (Optional)

If deploying API via API Gateway:

```bash
# Create Lambda function for API
aws lambda create-function \
    --function-name customer-chatbot-api \
    --runtime python3.9 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
    --handler src.api.main.handler \
    --zip-file fileb://deployment-package.zip
```

### 7. EventBridge for Scheduled Retraining

EventBridge rules are created automatically by the retraining trigger. Ensure Lambda execution role has EventBridge permissions.

### 8. CloudWatch

CloudWatch is automatically used for metrics and logging. Ensure your IAM roles have CloudWatch permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

## Setup Steps Summary

1. **Create S3 Buckets** ✅
2. **Create Kinesis Stream** ✅
3. **Create SageMaker IAM Role** ✅
4. **Enable Bedrock Access** ✅
5. **Create OpenSearch Domain** ✅
6. **Configure Environment Variables** ✅
7. **Test Connections**

## Testing AWS Setup

Run the setup test script:

```python
from config.aws_config import aws_config
from src.data_collection.s3_storage import s3_storage
from src.data_collection.kinesis_ingestion import kinesis_ingestion

# Test S3
s3_storage.create_bucket_if_not_exists("customer-chatbot-data")

# Test Kinesis
kinesis_ingestion.create_stream_if_not_exists()

# Test Bedrock
from src.models.bedrock_client import bedrock_client
response = bedrock_client.generate_response("Hello, test message")
print(response)

# Test OpenSearch
from src.models.rag.vector_store import vector_store
vector_store.create_index()
```

## Cost Optimization Tips

1. **Use Spot Instances** for SageMaker training
2. **Set up S3 Lifecycle Policies** to archive old data
3. **Use OpenSearch t3.small** for development (upgrade for production)
4. **Monitor CloudWatch costs** and set up billing alerts
5. **Use Kinesis On-Demand** for predictable workloads

## Security Best Practices

1. **Use IAM Roles** instead of access keys where possible
2. **Enable S3 Bucket Encryption**
3. **Use VPC** for SageMaker endpoints in production
4. **Enable CloudTrail** for audit logging
5. **Rotate Access Keys** regularly

## Troubleshooting

### Common Issues

1. **Bedrock Access Denied**: Request model access in Bedrock console
2. **SageMaker Role Issues**: Ensure role has S3 and SageMaker permissions
3. **OpenSearch Connection**: Check security group and access policies
4. **Kinesis Throttling**: Increase shard count or use On-Demand mode

## Next Steps

After AWS setup:
1. Initialize DVC for data versioning
2. Set up CI/CD pipelines
3. Deploy initial models
4. Configure monitoring

See [deployment_guide.md](deployment_guide.md) for deployment instructions.

