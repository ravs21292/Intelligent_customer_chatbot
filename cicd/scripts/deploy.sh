#!/bin/bash

# Deployment script for Customer Chatbot
# Usage: ./deploy.sh [environment] [version]
# Example: ./deploy.sh production v1.2.0

set -e  # Exit on error

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "Deploying Customer Chatbot to $ENVIRONMENT (version: $VERSION)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED} AWS credentials not configured${NC}"
    exit 1
fi

echo -e "${GREEN} AWS credentials verified${NC}"

# Set variables based on environment
if [ "$ENVIRONMENT" == "production" ]; then
    ECR_REPO="customer-chatbot-api-prod"
    ECS_CLUSTER="customer-chatbot-prod"
    ECS_SERVICE="customer-chatbot-api-prod"
elif [ "$ENVIRONMENT" == "staging" ]; then
    ECR_REPO="customer-chatbot-api-staging"
    ECS_CLUSTER="customer-chatbot-staging"
    ECS_SERVICE="customer-chatbot-api-staging"
else
    echo -e "${RED} Invalid environment: $ENVIRONMENT${NC}"
    echo "Valid environments: staging, production"
    exit 1
fi

AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo -e "${YELLOW} Building Docker image...${NC}"
docker build -t ${ECR_REPO}:${VERSION} -f infrastructure/docker/Dockerfile.api .

echo -e "${YELLOW} Tagging image...${NC}"
docker tag ${ECR_REPO}:${VERSION} ${ECR_REGISTRY}/${ECR_REPO}:${VERSION}
docker tag ${ECR_REPO}:${VERSION} ${ECR_REGISTRY}/${ECR_REPO}:latest

echo -e "${YELLOW} Pushing to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_REGISTRY}

docker push ${ECR_REGISTRY}/${ECR_REPO}:${VERSION}
docker push ${ECR_REGISTRY}/${ECR_REPO}:latest

echo -e "${YELLOW} Deploying to ECS...${NC}"
aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --force-new-deployment \
    --region ${AWS_REGION} \
    > /dev/null

echo -e "${GREEN} Deployment initiated!${NC}"
echo -e "${YELLOW} Waiting for service to stabilize...${NC}"

# Wait for deployment to complete
aws ecs wait services-stable \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --region ${AWS_REGION}

echo -e "${GREEN} Deployment complete!${NC}"

# Health check
echo -e "${YELLOW} Running health check...${NC}"
sleep 10

# Get service URL (adjust based on your setup)
# SERVICE_URL=$(aws elbv2 describe-load-balancers --query ...)
# if curl -f ${SERVICE_URL}/health; then
#     echo -e "${GREEN}  Health check passed!${NC}"
# else
#     echo -e "${RED} Health check failed!${NC}"
#     exit 1
# fi

echo -e "${GREEN} Deployment successful!${NC}"

