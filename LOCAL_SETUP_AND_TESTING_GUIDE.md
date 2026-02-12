# Local Setup and Testing Guide

## Step-by-Step Guide to Run and Test the Project Locally

This guide will help you:
1. ✅ Verify project structure
2. ✅ Set up the local environment
3. ✅ Run the API server
4. ✅ Test the complete flow: User Message → Intent Classification → Routing → Response
5. ✅ Debug step by step

---

## Phase 1: Structure Verification

### 1.1 Check Project Structure

First, let's verify the project structure is correct:

```bash
# Navigate to project root
cd Intelligent_customer_chatbot

# Check main directories exist
ls -la src/
ls -la config/
ls -la tests/
```

**Expected Structure:**
```
Intelligent_customer_chatbot/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── intent_classification/  # BERT intent classifier
│   ├── models/           # Multi-model router, Bedrock, RAG
│   ├── data_collection/ # Kinesis, S3, DVC
│   ├── training/        # Training pipelines
│   ├── monitoring/      # Drift detection, monitoring
│   └── utils/           # Helpers, logger, metrics
├── config/              # Configuration files
├── tests/               # Test files
└── requirements.txt     # Dependencies
```

### 1.2 Verify Key Files Exist

```bash
# Check critical files
ls src/api/main.py
ls src/api/chat_endpoints.py
ls src/models/model_router.py
ls src/intent_classification/router.py
ls src/intent_classification/intent_classifier.py
ls config/model_config.py
ls config/aws_config.py
```

---

## Phase 2: Environment Setup

### 2.1 Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 2.2 Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify key packages installed
pip list | grep -E "fastapi|torch|transformers|boto3"
```

**Note:** This will install PyTorch which is large (~2GB). If you only want to test the structure first, you can install minimal dependencies:

```bash
# Minimal setup for structure testing
pip install fastapi uvicorn pydantic python-dotenv
```

### 2.3 Create Environment File

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env  # Linux/Mac
# or create it manually on Windows
```

Add these variables to `.env`:

```env
# AWS Configuration (can use dummy values for local testing)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=dummy_key
AWS_SECRET_ACCESS_KEY=dummy_secret

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# Model Configuration
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=32
BEDROCK_MODEL_ID=anthropic.claude-v2

# OpenSearch (optional for local testing)
OPENSEARCH_ENDPOINT=
OPENSEARCH_INDEX_NAME=customer-support-kb

# Local Development Mode (set to True to skip AWS calls)
USE_MOCKS=False
```

### 2.4 Set Python Path

```bash
# On Windows (PowerShell):
$env:PYTHONPATH = "$PWD"

# On Windows (CMD):
set PYTHONPATH=%CD%

# On Linux/Mac:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or add to your `.env` file:
```env
PYTHONPATH=.
```

---

## Phase 3: Structure Testing (Without Full Dependencies)

### 3.1 Test Imports

Create a test script `test_structure.py`:

```python
# test_structure.py
"""Test if project structure and imports work."""

print("Testing project structure...")

try:
    print("✓ Testing config imports...")
    from config.model_config import model_config, domain_config
    print(f"  - Intent classes: {len(model_config.INTENT_CLASSES)}")
    print(f"  - Routing thresholds: {model_config.ROUTING_THRESHOLDS}")
    
    print("✓ Testing API imports...")
    from src.api.main import app
    print("  - FastAPI app created")
    
    print("✓ Testing intent classification imports...")
    from src.intent_classification.router import model_router
    print("  - Model router imported")
    
    print("✓ Testing model router imports...")
    from src.models.model_router import multi_model_router
    print("  - Multi-model router imported")
    
    print("\n✅ All structure checks passed!")
    print("\nProject structure is correct. You can proceed to full setup.")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Check that:")
    print("1. You're in the project root directory")
    print("2. PYTHONPATH is set correctly")
    print("3. All required packages are installed")
except Exception as e:
    print(f"\n❌ Error: {e}")
```

Run it:
```bash
python test_structure.py
```

---

## Phase 4: Full Local Setup

### 4.1 Install All Dependencies

```bash
# Make sure virtual environment is activated
pip install -r requirements.txt

# This will take a while (PyTorch is large)
```

### 4.2 Handle AWS Dependencies for Local Testing

Since you might not have AWS credentials set up, we need to handle AWS service calls gracefully. The code should handle missing credentials, but for local testing, you can:

**Option A: Use Mock Mode (Recommended for initial testing)**

Create a mock mode in your code or set environment variables to skip AWS calls:

```bash
# In .env file
USE_MOCKS=True
```

**Option B: Use LocalStack (Advanced)**

For full AWS service mocking, you can use LocalStack, but that's more complex.

**Option C: Let it fail gracefully**

The code has error handling, so AWS calls will fail gracefully and you can still test the structure.

---

## Phase 5: Running the API Server

### 5.1 Start the Server

```bash
# Make sure you're in project root and venv is activated
cd Intelligent_customer_chatbot

# Start the API server
uvicorn src.api.main:app --reload --port 8000 --host 0.0.0.0
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.2 Test Health Endpoint

In another terminal:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Or use browser: http://localhost:8000/health
# Or use Python:
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

Expected response:
```json
{"status": "healthy"}
```

### 5.3 Test Root Endpoint

```bash
curl http://localhost:8000/

# Should return API info
```

---

## Phase 6: Testing the Complete Flow

### 6.1 Test Chat Endpoint (Step by Step)

#### Step 1: Send a Simple Message

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I need help with my billing",
    "user_id": "test-user-123"
  }'
```

**Expected Flow:**
1. ✅ Request received at `/api/v1/chat`
2. ✅ Message validated
3. ✅ Intent classification triggered
4. ✅ Routing decision made
5. ✅ Response generated
6. ✅ Response returned

#### Step 2: Check the Response Structure

The response should include:
```json
{
  "response": "...",
  "intent": "billing",
  "confidence": 0.85,
  "strategy": "pre_trained" or "rag" or "fine_tuned",
  "sources": null or [...],
  "escalate": false,
  "conversation_id": "...",
  "timestamp": "..."
}
```

### 6.2 Test Different Intent Types

Test different messages to see routing:

```bash
# General inquiry (should route to pre_trained)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your business hours?", "user_id": "test-1"}'

# Billing query (should route based on confidence)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I was charged incorrectly on my last bill", "user_id": "test-2"}'

# Technical support (might route to RAG)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?", "user_id": "test-3"}'
```

---

## Phase 7: Debugging Step by Step

### 7.1 Enable Debug Logging

Add to your `.env`:
```env
API_DEBUG=True
LOG_LEVEL=DEBUG
```

### 7.2 Add Debug Print Statements

You can add print statements or use a debugger. Here's a debugging script:

```python
# debug_flow.py
"""Debug the complete flow step by step."""

import sys
sys.path.insert(0, '.')

from src.models.model_router import multi_model_router
from src.intent_classification.intent_classifier import intent_classifier
from src.intent_classification.router import model_router

def debug_flow(message: str):
    """Debug the complete flow."""
    print("=" * 60)
    print(f"DEBUGGING FLOW FOR MESSAGE: {message}")
    print("=" * 60)
    
    # Step 1: Intent Classification
    print("\n[STEP 1] Intent Classification")
    print("-" * 60)
    try:
        intent_result = intent_classifier.classify(message)
        print(f"✓ Intent: {intent_result['intent']}")
        print(f"✓ Confidence: {intent_result['confidence']:.3f}")
        if 'top_intents' in intent_result:
            print(f"✓ Top 3 intents: {intent_result['top_intents']}")
    except Exception as e:
        print(f"✗ Error in intent classification: {e}")
        return
    
    # Step 2: Routing Decision
    print("\n[STEP 2] Routing Decision")
    print("-" * 60)
    try:
        routing_decision = model_router.route(message, intent_result)
        print(f"✓ Strategy: {routing_decision['strategy']}")
        print(f"✓ Model Name: {routing_decision.get('model_name', 'N/A')}")
        print(f"✓ Reasoning: {routing_decision.get('reasoning', 'N/A')}")
        print(f"✓ Use RAG: {routing_decision.get('use_rag', False)}")
    except Exception as e:
        print(f"✗ Error in routing: {e}")
        return
    
    # Step 3: Response Generation
    print("\n[STEP 3] Response Generation")
    print("-" * 60)
    try:
        response = multi_model_router.generate_response(message)
        print(f"✓ Response generated")
        print(f"✓ Strategy used: {response.get('strategy', 'N/A')}")
        print(f"✓ Response length: {len(response.get('response', ''))}")
        print(f"✓ Escalate: {response.get('escalate', False)}")
        if response.get('sources'):
            print(f"✓ Sources: {len(response['sources'])} documents")
    except Exception as e:
        print(f"✗ Error in response generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✓ FLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    # Test with different messages
    test_messages = [
        "Hello, I need help with my billing",
        "What are your business hours?",
        "How do I reset my password?",
        "I want a refund for my subscription"
    ]
    
    for msg in test_messages:
        debug_flow(msg)
        print("\n")
```

Run it:
```bash
python debug_flow.py
```

### 7.3 Use Python Debugger

```python
# Add breakpoints in your code
import pdb; pdb.set_trace()

# Or use IDE debugger (VS Code, PyCharm, etc.)
```

### 7.4 Check Logs

The application uses a logger. Check console output for:
- Intent classification results
- Routing decisions
- Model selection
- Errors

---

## Phase 8: Common Issues and Solutions

### Issue 1: Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%CD%  # Windows CMD
$env:PYTHONPATH = "$PWD"  # Windows PowerShell

# Or run from project root:
python -m src.api.main
```

### Issue 2: AWS Credentials Error

**Error:** `NoCredentialsError` or AWS service errors

**Solution:**
- For local testing, the code should handle this gracefully
- Check that error handling in `bedrock_client.py` and other AWS clients catches exceptions
- You can mock AWS services or skip them for structure testing

### Issue 3: Model Not Found

**Error:** Intent classifier model not found

**Solution:**
- The code should fall back to using the base BERT model
- Check `src/intent_classification/intent_classifier.py` - it loads from HuggingFace if local model not found
- First run will download the model (~400MB)

### Issue 4: Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Use different port
uvicorn src.api.main:app --reload --port 8001

# Or kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
# Linux/Mac:
lsof -ti:8000 | xargs kill
```

### Issue 5: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
# Install missing package
pip install transformers

# Or reinstall all
pip install -r requirements.txt
```

---

## Phase 9: Testing Checklist

Use this checklist to verify everything works:

- [ ] Project structure verified
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] `.env` file created with configuration
- [ ] PYTHONPATH set correctly
- [ ] API server starts without errors
- [ ] Health endpoint returns `{"status": "healthy"}`
- [ ] Chat endpoint accepts POST requests
- [ ] Intent classification works (returns intent and confidence)
- [ ] Routing decision is made correctly
- [ ] Response is generated and returned
- [ ] Different message types route to different strategies
- [ ] Error handling works (test with invalid input)
- [ ] Logs show the flow step by step

---

## Phase 10: Next Steps

Once basic structure testing works:

1. **Train Intent Classifier Model:**
   ```bash
   python -m src.intent_classification.model_training
   ```

2. **Set up AWS Services** (if needed):
   - Configure AWS credentials
   - Set up Kinesis stream
   - Set up S3 buckets
   - Set up OpenSearch cluster

3. **Add Test Data:**
   - Create sample training data
   - Add documents to RAG knowledge base

4. **Run Full Tests:**
   ```bash
   pytest tests/
   ```

---

## Quick Test Script

Save this as `quick_test.py`:

```python
"""Quick test of the complete flow."""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}")
    return response.status_code == 200

def test_chat(message: str):
    """Test chat endpoint."""
    payload = {
        "message": message,
        "user_id": "test-user"
    }
    response = requests.post(
        f"{BASE_URL}/api/v1/chat",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"\nMessage: {message}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Intent: {data.get('intent')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Strategy: {data.get('strategy')}")
        print(f"Response: {data.get('response')[:100]}...")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing API...")
    
    # Test health
    if not test_health():
        print("❌ Health check failed. Is the server running?")
        exit(1)
    
    # Test different messages
    messages = [
        "Hello, I need help",
        "What are your business hours?",
        "I have a billing question",
        "How do I reset my password?"
    ]
    
    for msg in messages:
        test_chat(msg)
        print("-" * 60)
```

Run it:
```bash
# Make sure server is running first
python quick_test.py
```

---

## Summary

**To test structure only (fastest):**
1. Create venv and install minimal deps (fastapi, uvicorn)
2. Run `test_structure.py` to verify imports
3. Check that all files are in place

**To test full flow:**
1. Install all dependencies
2. Set up `.env` file
3. Start API server
4. Use `quick_test.py` or curl commands
5. Use `debug_flow.py` for detailed debugging

**To debug step by step:**
1. Enable debug logging
2. Use `debug_flow.py` script
3. Add breakpoints in IDE
4. Check console logs

This should get you started! Let me know if you encounter any specific issues.

