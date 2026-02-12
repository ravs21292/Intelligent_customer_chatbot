# Llama Model (LoRA Fine-tuning) - Complete Guide

## 📋 Focus: Only Llama Model with LoRA Fine-tuning

This guide covers ONLY the Llama fine-tuned model section. It explains:
1. Code structure and implementation
2. Training process with dataset
3. Model and dataset versioning
4. Deployment on SageMaker
5. API inference

---

## 🗂️ Files You Need to Focus On

### Core Training Files
1. **`src/models/fine_tuning/lora_trainer.py`** - Main LoRA training code
2. **`src/models/fine_tuning/data_preparation.py`** - Data preparation
3. **`src/models/fine_tuning/model_evaluator.py`** - Model evaluation

### Pipeline Orchestration
4. **`src/training/training_pipeline.py`** - Training pipeline (lines 76-118)

### Versioning
5. **`src/data_collection/data_versioning.py`** - DVC data versioning
6. **`src/data_collection/s3_storage.py`** - S3 storage for models

### Inference
7. **`src/models/model_router.py`** - Model routing and inference (lines 113-131)
8. **`src/api/chat_endpoints.py`** - API endpoint

### Configuration
9. **`config/model_config.py`** - Model hyperparameters (lines 31-37)

---

## 📝 Step-by-Step: Complete Llama Model Flow

### Step 1: Data Preparation

**File**: `src/models/fine_tuning/data_preparation.py`

#### 1.1 Extract Q&A Pairs from Support Tickets

**Code** (lines 13-40):

```python
def extract_qa_pairs(self, support_tickets: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    qa_pairs = []
    
    for ticket in support_tickets:
        question = ticket.get("customer_message", ticket.get("question", ""))
        answer = ticket.get("agent_response", ticket.get("answer", ""))
        
        if question and answer:
            qa_pairs.append({
                "instruction": question,
                "response": answer,
                "domain": ticket.get("category", "general")
            })
    
    return qa_pairs
```

**What it does:**
- Takes support tickets (JSON format)
- Extracts customer questions and agent responses
- Creates Q&A pairs with domain labels
- Returns list of training examples

**Example Input:**
```json
{
  "customer_message": "Why was I charged $50?",
  "agent_response": "The $50 charge is for your monthly subscription renewal...",
  "category": "billing"
}
```

**Example Output:**
```python
{
  "instruction": "Why was I charged $50?",
  "response": "The $50 charge is for your monthly subscription renewal...",
  "domain": "billing"
}
```

#### 1.2 Prepare Domain-Specific Data

**Code** (lines 72-106):

```python
def prepare_domain_specific_data(self, domain: str, data_source: str) -> List[Dict[str, str]]:
    # Load data (from S3 or local file)
    if data_source.startswith("s3://"):
        data = s3_storage.download_data(data_source.replace("s3://", "").split("/", 1)[1])
    else:
        with open(data_source, "r") as f:
            data = json.load(f)
    
    # Filter by domain (e.g., "billing", "technical_support")
    domain_data = [
        item for item in data
        if item.get("domain", item.get("category", "")) == domain
    ]
    
    # Extract Q&A pairs
    qa_pairs = self.extract_qa_pairs(domain_data)
    
    # Augment data (increase dataset size)
    augmented = self.augment_data(qa_pairs)
    
    return augmented
```

**What it does:**
- Loads data from S3 or local file
- Filters data for specific domain (billing, technical, etc.)
- Extracts Q&A pairs
- Augments data (creates variations)

**Usage:**
```python
from src.models.fine_tuning.data_preparation import data_preparation

# Prepare billing domain data
billing_data = data_preparation.prepare_domain_specific_data(
    domain="billing",
    data_source="data/domain_specific/billing.json"
)
```

---

### Step 2: Dataset Versioning with DVC

**File**: `src/data_collection/data_versioning.py`

#### 2.1 Track Dataset with DVC

**Code** (lines 49-92):

```python
def track_dataset(self, dataset_path: str, dataset_name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    # Add dataset to DVC
    subprocess.run(["dvc", "add", dataset_path], check=True, capture_output=True)
    
    # Create metadata file
    if metadata:
        metadata_path = f"{dataset_path}.meta"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Commit to DVC
    subprocess.run(["dvc", "commit", "-f", dataset_path + ".dvc"], check=True, capture_output=True)
    
    return True
```

**What it does:**
- Tracks dataset with DVC (like Git for data)
- Creates `.dvc` file with dataset hash
- Stores metadata (dataset name, version, record count)
- Commits to DVC repository

**Usage:**
```python
from src.data_collection.data_versioning import data_versioning

# Track billing dataset
metadata = {
    "dataset_name": "billing-v1",
    "version": "v1.0",
    "record_count": 1000,
    "domain": "billing",
    "created_at": "2024-01-15T10:30:00Z"
}

data_versioning.track_dataset(
    dataset_path="data/domain_specific/billing.json",
    dataset_name="billing-v1",
    metadata=metadata
)

# Push to S3
data_versioning.push_dataset("data/domain_specific/billing.json")
```

**Result:**
- Dataset tracked in DVC
- `.dvc` file created with hash
- Metadata file created
- Dataset pushed to S3 (`s3://bucket/dvc-storage/`)

---

### Step 3: Training Pipeline

**File**: `src/training/training_pipeline.py` (lines 76-118)

#### 3.1 Run Fine-tuning Pipeline

**Code**:

```python
def run_fine_tuning_pipeline(
    self,
    domain: str,
    data_path: str,
    output_dir: str = None
) -> str:
    logger.info(f"Starting fine-tuning pipeline for domain: {domain}")
    
    from src.models.fine_tuning.data_preparation import data_preparation
    
    # Step 1: Prepare data
    training_data = data_preparation.prepare_domain_specific_data(domain, data_path)
    
    # Step 2: Create dataset
    from datasets import Dataset
    dataset = lora_trainer.prepare_dataset(training_data)
    
    # Step 3: Split train/val
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    # Step 4: Train
    output_dir = output_dir or f"models/fine_tuned/{domain}"
    model_path = lora_trainer.train(
        train_dataset,
        val_dataset,
        output_dir
    )
    
    return model_path
```

**Usage:**
```python
from src.training.training_pipeline import training_pipeline

# Train billing domain model
model_path = training_pipeline.run_fine_tuning_pipeline(
    domain="billing",
    data_path="data/domain_specific/billing.json",
    output_dir="models/fine_tuned/billing"
)
```

---

### Step 4: LoRA Training (Core Code)

**File**: `src/models/fine_tuning/lora_trainer.py`

#### 4.1 Prepare Dataset Format

**Code** (lines 30-60):

```python
def prepare_dataset(self, data: List[Dict[str, str]], format_type: str = "instruction") -> Dataset:
    if format_type == "instruction":
        formatted_data = [
            {
                "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
            }
            for item in data
        ]
    
    return Dataset.from_list(formatted_data)
```

**What it does:**
- Formats Q&A pairs into instruction-response format
- Creates HuggingFace Dataset object

**Example Output:**
```
### Instruction:
Why was I charged $50?

### Response:
The $50 charge is for your monthly subscription renewal...
```

#### 4.2 Load Base Model and Configure LoRA

**Code** (lines 87-109):

```python
# Load base model (Llama-2-7b-chat-hf)
tokenizer = AutoTokenizer.from_pretrained(self.base_model)  # "meta-llama/Llama-2-7b-chat-hf"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    self.base_model,
    torch_dtype=torch.float16,  # Half precision to save memory
    device_map="auto"            # Auto-distribute across GPUs
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=self.lora_rank,              # 8 (from config)
    lora_alpha=self.lora_alpha,     # 16 (from config)
    lora_dropout=self.lora_dropout, # 0.1 (from config)
    target_modules=["q_proj", "v_proj"]  # Which layers to adapt
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 6,738,415,616 || trainable%: 0.12
```

**What it does:**
- Loads Llama-2-7b base model from Hugging Face
- Configures LoRA (only train 0.12% of parameters)
- Applies LoRA adapters to q_proj and v_proj layers

#### 4.3 Tokenize and Train

**Code** (lines 111-156):

```python
# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

train_dataset = train_data.map(tokenize_function, batched=True)
val_dataset = val_data.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,                    # "models/fine_tuned/billing"
    num_train_epochs=epochs,                  # 3
    per_device_train_batch_size=batch_size,  # 4
    per_device_eval_batch_size=batch_size,   # 4
    learning_rate=learning_rate,               # 2e-4
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",                    # Save after each epoch
    evaluation_strategy="epoch",              # Evaluate after each epoch
    load_best_model_at_end=True               # Load best model
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (not masked LM)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)  # Saves only LoRA adapters (~32MB)
tokenizer.save_pretrained(output_dir)
```

**What happens:**
- Tokenizes training data
- Trains only LoRA adapters (8M parameters)
- Saves adapters to `models/fine_tuned/billing/`

**Output:**
```
models/fine_tuned/billing/
├── adapter_config.json      # LoRA configuration
├── adapter_model.bin        # LoRA weights (32MB)
├── tokenizer_config.json
└── tokenizer.json
```

---

### Step 5: Model Versioning

#### 5.1 Upload Model to S3 with Version

**File**: `src/data_collection/s3_storage.py`

**Code** (upload_model method):

```python
def upload_model(self, model_path: str, model_name: str, version: str) -> str:
    s3_prefix = f"{pipeline_config.S3_MODEL_PATH}/{model_name}/v{version}/"
    
    # Upload all files in model directory
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, model_path)
            s3_key = f"{s3_prefix}{relative_path}"
            
            self.s3_client.upload_file(
                local_path,
                self.model_bucket,
                s3_key
            )
    
    return s3_prefix
```

**Usage:**
```python
from src.data_collection.s3_storage import s3_storage

# Upload trained model
s3_path = s3_storage.upload_model(
    model_path="models/fine_tuned/billing",
    model_name="billing-model",
    version="1.0"
)
# Returns: s3://bucket/models/billing-model/v1.0/
```

#### 5.2 Link Model Version with Dataset Version

**Create Model Metadata:**

```python
# After training, create metadata linking model to dataset
model_metadata = {
    "model_name": "billing-model",
    "version": "1.0",
    "dataset_version": "billing-v1.0",  # Links to dataset version
    "dataset_path": "data/domain_specific/billing.json",
    "dataset_hash": "abc123...",  # From DVC
    "training_date": "2024-01-15T10:30:00Z",
    "hyperparameters": {
        "lora_rank": 8,
        "lora_alpha": 16,
        "epochs": 3,
        "learning_rate": 2e-4
    },
    "metrics": {
        "train_loss": 0.745,
        "val_loss": 0.856
    },
    "s3_path": "s3://bucket/models/billing-model/v1.0/"
}

# Save metadata
metadata_path = "models/fine_tuned/billing/metadata.json"
with open(metadata_path, "w") as f:
    json.dump(model_metadata, f, indent=2)
```

**This creates traceability:**
- Model v1.0 → Trained on dataset billing-v1.0
- Can reproduce exact training setup
- Track which data produced which model

---

### Step 6: Deployment on SageMaker

#### 6.1 Create SageMaker Model

**File**: Create new file `src/models/fine_tuning/sagemaker_deployment.py`

```python
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import Session
import boto3

def deploy_lora_model_to_sagemaker(
    model_s3_path: str,
    endpoint_name: str,
    instance_type: str = "ml.g4dn.xlarge"
) -> str:
    """
    Deploy LoRA fine-tuned model to SageMaker endpoint.
    
    Args:
        model_s3_path: S3 path to model (e.g., s3://bucket/models/billing-model/v1.0/)
        endpoint_name: Name for SageMaker endpoint
        instance_type: SageMaker instance type
        
    Returns:
        Endpoint name
    """
    session = Session()
    
    # Create HuggingFace model
    # Note: For LoRA, we need to package base model + adapters
    huggingface_model = HuggingFaceModel(
        model_data=model_s3_path,  # S3 path to model.tar.gz
        role="arn:aws:iam::account:role/SageMakerExecutionRole",
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        entry_point="inference.py",  # Custom inference script
        source_dir="src/models/fine_tuning"  # Directory with inference code
    )
    
    # Deploy endpoint
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    
    return endpoint_name
```

#### 6.2 Create Inference Script for SageMaker

**File**: `src/models/fine_tuning/inference.py` (for SageMaker container)

```python
"""
Inference script for SageMaker endpoint.
This runs inside SageMaker container.
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def model_fn(model_dir):
    """Load model when endpoint starts."""
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    """Parse input."""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data.get("text", "")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Generate prediction."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # Format prompt
    prompt = f"### Instruction:\n{input_data}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    
    return {"response": response}

def output_fn(prediction, response_content_type):
    """Format output."""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
```

#### 6.3 Package Model for SageMaker

**Before deploying, package model:**

```python
import tarfile
import os

def package_model_for_sagemaker(local_model_path: str, output_path: str):
    """Package LoRA model for SageMaker deployment."""
    with tarfile.open(output_path, "w:gz") as tar:
        # Add LoRA adapters
        tar.add(
            os.path.join(local_model_path, "adapter_model.bin"),
            arcname="adapter_model.bin"
        )
        tar.add(
            os.path.join(local_model_path, "adapter_config.json"),
            arcname="adapter_config.json"
        )
        # Add tokenizer files
        for file in ["tokenizer.json", "tokenizer_config.json"]:
            if os.path.exists(os.path.join(local_model_path, file)):
                tar.add(
                    os.path.join(local_model_path, file),
                    arcname=file
                )
    
    # Upload to S3
    s3_storage.upload_data(
        data=open(output_path, "rb").read(),
        key=f"models/billing-model/v1.0/model.tar.gz",
        bucket=pipeline_config.S3_BUCKET_MODELS
    )
```

---

### Step 7: API Inference

#### 7.1 Load Model in API Server

**File**: `src/models/model_router.py` (update lines 113-131)

```python
from src.models.fine_tuning.lora_trainer import lora_trainer
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class MultiModelRouter:
    def __init__(self):
        # ... existing code ...
        self.fine_tuned_models = {}  # Cache loaded models
    
    def load_fine_tuned_model(self, model_name: str, model_path: str):
        """Load fine-tuned model into memory."""
        try:
            # Load base model (once, shared across domains)
            if "base_model" not in self.fine_tuned_models:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-2-7b-chat-hf",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.fine_tuned_models["base_model"] = base_model
            
            # Load LoRA adapters for this domain
            base_model = self.fine_tuned_models["base_model"]
            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Cache model
            self.fine_tuned_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            
            logger.info(f"Loaded fine-tuned model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def _generate_with_fine_tuned(
        self,
        model_name: str,
        message: str,
        intent: str
    ) -> Dict[str, Any]:
        """Generate response using fine-tuned model."""
        # Load model if not already loaded
        if model_name not in self.fine_tuned_models:
            model_path = f"models/fine_tuned/{model_name.replace('-model', '')}"
            self.load_fine_tuned_model(model_name, model_path)
        
        model_dict = self.fine_tuned_models[model_name]
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        
        # Format prompt
        prompt = f"### Instruction:\n{message}\n\n### Response:\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return {
            "response": response,
            "model": model_name,
            "strategy": "fine_tuned"
        }
```

#### 7.2 API Endpoint Usage

**File**: `src/api/chat_endpoints.py` (line 65)

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # ... validation ...
    
    # Generate response (includes fine-tuned model routing)
    response = multi_model_router.generate_response(
        message=request.message,
        conversation_history=request.conversation_history,
        user_id=request.user_id
    )
    
    # Response includes fine-tuned model output if routed there
    return ChatResponse(
        response=response.get("response", ""),
        intent=response.get("intent"),
        strategy=response.get("strategy"),  # "fine_tuned"
        # ...
    )
```

---

## 🔄 Complete Workflow Summary

### Training Workflow

```
1. Prepare Data
   File: src/models/fine_tuning/data_preparation.py
   → Extract Q&A pairs from support tickets
   → Filter by domain (billing, technical, etc.)
   → Augment data

2. Version Dataset
   File: src/data_collection/data_versioning.py
   → Track with DVC
   → Push to S3
   → Get dataset hash/version

3. Train Model
   File: src/models/fine_tuning/lora_trainer.py
   → Load Llama-2-7b base model
   → Apply LoRA adapters
   → Train on domain data
   → Save adapters (32MB)

4. Version Model
   File: src/data_collection/s3_storage.py
   → Upload to S3 with version
   → Create metadata linking to dataset version
   → Register in model registry

5. Deploy (Optional - SageMaker)
   File: src/models/fine_tuning/sagemaker_deployment.py
   → Package model
   → Deploy to SageMaker endpoint
   → OR use on API server (current implementation)
```

### Inference Workflow

```
1. API Request
   File: src/api/chat_endpoints.py
   → User sends message

2. Intent Classification
   File: src/intent_classification/intent_classifier.py
   → Classify intent (e.g., "billing")

3. Routing Decision
   File: src/intent_classification/router.py
   → Route to "fine_tuned" strategy

4. Load Model (if not loaded)
   File: src/models/model_router.py
   → Load base Llama model (once)
   → Load domain-specific LoRA adapters
   → Cache in memory

5. Generate Response
   File: src/models/model_router.py (_generate_with_fine_tuned)
   → Format prompt
   → Generate with fine-tuned model
   → Return response

6. Return to User
   File: src/api/chat_endpoints.py
   → Format response
   → Return to user
```

---

## 📊 Model-Dataset Versioning Example

### Version 1.0

**Dataset:**
- Path: `data/domain_specific/billing.json`
- DVC Hash: `abc123def456...`
- Version: `billing-v1.0`
- Records: 1000

**Model:**
- Path: `models/fine_tuned/billing/`
- S3: `s3://bucket/models/billing-model/v1.0/`
- Version: `billing-model-v1.0`
- Trained on: `billing-v1.0` (dataset hash: `abc123...`)
- Metrics: `train_loss: 0.745, val_loss: 0.856`

### Version 2.0 (After Retraining)

**Dataset:**
- Path: `data/domain_specific/billing.json` (updated)
- DVC Hash: `xyz789uvw012...` (different hash = different data)
- Version: `billing-v2.0`
- Records: 2000 (added more data)

**Model:**
- Path: `models/fine_tuned/billing/` (new training)
- S3: `s3://bucket/models/billing-model/v2.0/`
- Version: `billing-model-v2.0`
- Trained on: `billing-v2.0` (dataset hash: `xyz789...`)
- Metrics: `train_loss: 0.623, val_loss: 0.712` (improved!)

**Traceability:**
- Model v2.0 → Dataset v2.0 → Can reproduce exact training
- Compare v1.0 vs v2.0 performance
- Rollback to v1.0 if v2.0 performs worse

---

## 🎯 Key Files Summary

| File | Purpose | Key Methods |
|------|---------|-------------|
| `src/models/fine_tuning/lora_trainer.py` | LoRA training | `train()`, `generate_response()` |
| `src/models/fine_tuning/data_preparation.py` | Data prep | `prepare_domain_specific_data()`, `extract_qa_pairs()` |
| `src/training/training_pipeline.py` | Pipeline | `run_fine_tuning_pipeline()` |
| `src/data_collection/data_versioning.py` | DVC versioning | `track_dataset()`, `push_dataset()` |
| `src/data_collection/s3_storage.py` | S3 storage | `upload_model()` |
| `src/models/model_router.py` | Inference | `_generate_with_fine_tuned()`, `load_fine_tuned_model()` |
| `src/api/chat_endpoints.py` | API endpoint | `chat()` |

---

## 💡 Interview Talking Points

**Q: How do you train the Llama model?**
> "I use LoRA (Low-Rank Adaptation) to fine-tune Llama-2-7b. First, I prepare domain-specific data by extracting Q&A pairs from support tickets. Then I format them into instruction-response format. I load the base Llama model, apply LoRA adapters to only the query and value projection layers, and train only 8 million parameters (0.12% of the model). This reduces training time from days to hours while maintaining 95-98% of full fine-tuning performance."

**Q: How do you version models and datasets?**
> "I use DVC (Data Version Control) for dataset versioning, which tracks dataset hashes and metadata. When I train a model, I create metadata that links the model version to the dataset version it was trained on. Both are stored in S3 with version numbers. This allows me to reproduce experiments and track which data produced which model."

**Q: Where does the model run?**
> "Currently, the fine-tuned models run on our API server. The base Llama model loads once at startup, and domain-specific LoRA adapters (32MB each) load on-demand. This is efficient because the adapters are small. Alternatively, we can deploy to SageMaker endpoints for managed infrastructure, but for small adapters, running on the API server is more cost-effective."

---

**This guide covers ONLY the Llama model section. Focus on these files to understand the complete flow!** 🚀

