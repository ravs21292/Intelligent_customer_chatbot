# LoRA Code Walkthrough: Step-by-Step in This Project

## 🔍 Complete Code Flow

### Step 1: Initialize LoRA Trainer

**File**: `src/models/fine_tuning/lora_trainer.py`

```python
# Line 24-28: Configuration loaded from config
def __init__(self):
    self.base_model = model_config.FINE_TUNE_BASE_MODEL  # "meta-llama/Llama-2-7b-chat-hf"
    self.lora_rank = model_config.LORA_RANK              # 8
    self.lora_alpha = model_config.LORA_ALPHA            # 16
    self.lora_dropout = model_config.LORA_DROPOUT        # 0.1
```

**What happens:**
- Loads configuration from `config/model_config.py`
- Sets up LoRA hyperparameters

---

### Step 2: Load Base Model

**Code** (Lines 88-96):

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(self.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    self.base_model,                    # "meta-llama/Llama-2-7b-chat-hf"
    torch_dtype=torch.float16,          # Use half precision (saves memory)
    device_map="auto"                   # Auto-distribute across GPUs
)
```

**Memory Usage:**
- Full precision (float32): ~28GB
- Half precision (float16): ~14GB ✅ (what we use)

**What gets loaded:**
```
Llama-2-7b-chat-hf/
├── config.json
├── tokenizer.json
├── pytorch_model.bin (7B parameters, 14GB)
└── ...
```

---

### Step 3: Configure LoRA

**Code** (Lines 99-105):

```python
# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,        # Language modeling
    r=self.lora_rank,                    # 8 (rank)
    lora_alpha=self.lora_alpha,           # 16 (scaling)
    lora_dropout=self.lora_dropout,      # 0.1 (dropout)
    target_modules=["q_proj", "v_proj"] # Which layers to adapt
)
```

**Visual Breakdown:**

```
LoraConfig:
├── r = 8
│   └── Low-rank dimension
│       └── A: (8 × 4096) matrix
│       └── B: (4096 × 8) matrix
│
├── lora_alpha = 16
│   └── Scaling factor
│   └── Final adaptation = (alpha/r) × (B × A) = 2 × (B × A)
│
├── lora_dropout = 0.1
│   └── 10% dropout in LoRA layers (prevents overfitting)
│
└── target_modules = ["q_proj", "v_proj"]
    └── Only adapt Query and Value projections
    └── Keeps attention mechanism intact
```

**Why q_proj and v_proj?**
- **q_proj**: Query projection (what to attend to)
- **v_proj**: Value projection (what information to extract)
- These are the most important for task adaptation
- Only 2 layers per transformer block = minimal parameters

---

### Step 4: Apply LoRA to Model

**Code** (Lines 108-109):

```python
# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**What `get_peft_model` does:**

```python
# Internally, PEFT library does:

# 1. Freeze all original weights
for param in model.parameters():
    param.requires_grad = False  # Don't update these

# 2. Add LoRA adapters to target modules
for module_name in ["q_proj", "v_proj"]:
    original_module = get_module(model, module_name)
    
    # Create LoRA matrices
    lora_A = nn.Linear(in_features, r, bias=False)      # (4096, 8)
    lora_B = nn.Linear(r, out_features, bias=False)    # (8, 4096)
    
    # Wrap original module
    lora_module = LoRALinear(original_module, lora_A, lora_B)
    set_module(model, module_name, lora_module)

# 3. Enable gradients only for LoRA parameters
for param in model.parameters():
    if 'lora' in param.name:
        param.requires_grad = True  # Only these are trainable
```

**Output of `print_trainable_parameters()`:**
```
trainable params: 8,388,608 || all params: 6,738,415,616 || trainable%: 0.12
```

**Breakdown:**
- **All params**: 6.7B (original model)
- **Trainable**: 8.4M (LoRA adapters)
- **Percentage**: 0.12% (only 1/1000th!)

---

### Step 5: Prepare Training Data

**Code** (Lines 30-60):

```python
def prepare_dataset(self, data, format_type="instruction"):
    if format_type == "instruction":
        formatted_data = [
            {
                "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
            }
            for item in data
        ]
    # ...
    return Dataset.from_list(formatted_data)
```

**Example Input:**
```python
data = [
    {
        "instruction": "How do I reset my password?",
        "response": "To reset your password, go to Settings > Security..."
    }
]
```

**Example Output:**
```
### Instruction:
How do I reset my password?

### Response:
To reset your password, go to Settings > Security > Reset Password...
```

---

### Step 6: Tokenize Data

**Code** (Lines 112-124):

```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,        # Cut if too long
        max_length=512,         # Maximum 512 tokens
        padding="max_length"   # Pad to 512
    )

train_dataset = train_data.map(tokenize_function, batched=True)
```

**What happens:**

```
Text: "### Instruction:\nHow do I reset...\n\n### Response:\n..."
    │
    ▼
Tokenizer
    │
    ▼
Token IDs: [1234, 5678, 9012, ..., 3456]
    │
    ▼
Padding (if needed)
    │
    ▼
Final: [1234, 5678, ..., 0, 0, 0]  (512 tokens)
```

**Memory per example:**
- 512 tokens × 4 bytes (int32) = 2KB per example
- Batch of 4 = 8KB per batch

---

### Step 7: Configure Training

**Code** (Lines 127-138):

```python
training_args = TrainingArguments(
    output_dir=output_dir,                    # "models/fine_tuned/billing"
    num_train_epochs=epochs,                  # 3
    per_device_train_batch_size=batch_size,  # 4
    per_device_eval_batch_size=batch_size,   # 4
    learning_rate=learning_rate,                # 2e-4
    warmup_steps=100,                         # Warmup for 100 steps
    logging_steps=10,                         # Log every 10 steps
    save_strategy="epoch",                    # Save after each epoch
    evaluation_strategy="epoch" if val_dataset else "no",
    load_best_model_at_end=True if val_dataset else False
)
```

**Why these settings?**

| Setting | Value | Reason |
|---------|-------|--------|
| **batch_size** | 4 | Small due to GPU memory (LoRA still needs base model in memory) |
| **learning_rate** | 2e-4 | Higher than full fine-tuning (0.5e-5) because we're training fewer params |
| **epochs** | 3 | Usually enough for LoRA (converges faster) |
| **warmup_steps** | 100 | Gradually increase learning rate (prevents instability) |

---

### Step 8: Create Trainer

**Code** (Lines 141-153):

```python
# Data collator (for batching)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (not masked LM)
)

# Trainer
trainer = Trainer(
    model=model,                    # LoRA-adapted model
    args=training_args,             # Training config
    train_dataset=train_dataset,    # Training data
    eval_dataset=val_dataset,       # Validation data
    data_collator=data_collator     # How to batch
)
```

**Data Collator Explanation:**

```python
# DataCollatorForLanguageModeling does:
# For causal LM (next token prediction):

Input:  [1234, 5678, 9012, 3456]
Labels: [5678, 9012, 3456, -100]  # Shifted by 1, -100 = ignore

# Model learns: given [1234, 5678, 9012], predict 3456
```

---

### Step 9: Train!

**Code** (Line 156):

```python
trainer.train()
```

**What happens during training:**

```
Epoch 1/3:
  Step 1/100:
    Forward pass:
      Input → Base Model (frozen) → LoRA adapters (trainable) → Output
      Loss = 2.345
    Backward pass:
      Gradients flow ONLY through LoRA adapters
      Update: A and B matrices only
      
  Step 10/100: Loss = 1.987
  Step 20/100: Loss = 1.654
  ...
  Step 100/100: Loss = 1.234
  
Epoch 2/3:
  Step 1/100: Loss = 1.123
  ...
  
Epoch 3/3:
  Step 1/100: Loss = 0.987
  ...
  Final: Loss = 0.856
```

**Gradient Flow:**

```
Loss
  │
  ▼
Output Layer (backprop)
  │
  ▼
LoRA Adapters (B × A) ← Gradients flow here ✅
  │
  ▼
Base Model (W) ← NO gradients (frozen) ❌
  │
  ▼
Input
```

**Only LoRA parameters get updated:**
```python
# Pseudo-code of what happens:
for batch in dataloader:
    # Forward
    output = model(input)
    loss = criterion(output, labels)
    
    # Backward (only LoRA params have gradients)
    loss.backward()
    
    # Update (only LoRA params)
    optimizer.step()  # Only updates A and B matrices
```

---

### Step 10: Save Model

**Code** (Lines 159-164):

```python
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

**What gets saved:**

```
models/fine_tuned/billing/
├── adapter_config.json          # LoRA configuration
│   {
│     "r": 8,
│     "lora_alpha": 16,
│     "target_modules": ["q_proj", "v_proj"],
│     ...
│   }
│
├── adapter_model.bin            # LoRA weights (A and B matrices)
│   └── Only 32MB! (vs 14GB for full model)
│
├── tokenizer_config.json        # Tokenizer config
├── tokenizer.json               # Tokenizer
└── special_tokens_map.json      # Special tokens
```

**Important:** Base model is NOT saved (we load it separately)

---

### Step 11: Inference (Using Fine-tuned Model)

**Code** (Lines 166-203):

```python
def generate_response(self, model_path, prompt, max_length=200):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model + LoRA adapters
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()
```

**Inference Flow:**

```
User Input: "How do I cancel my subscription?"
    │
    ▼
Tokenize: [1234, 5678, ...]
    │
    ▼
Base Model (Frozen)
    │
    ├─── W × Input  (Original computation)
    │
    └─── LoRA Adapters (Loaded from adapter_model.bin)
         │
         └─── B × A × Input  (Adaptation)
    │
    ▼
Output = W×Input + (alpha/r)×B×A×Input
    │
    ▼
Decode: "To cancel your subscription, go to Account Settings..."
```

---

## 📊 Memory and Performance Breakdown

### During Training

```
GPU Memory Usage:
├── Base Model (frozen):     14GB
├── LoRA Adapters (trainable): 32MB
├── Optimizer states:         64MB  (Adam optimizer for LoRA)
├── Gradients:                32MB  (only for LoRA)
├── Activations:              2GB   (forward pass)
└── Total:                    ~16GB

vs Full Fine-tuning:
├── Base Model (trainable):   14GB
├── Optimizer states:         56GB  (Adam for all params)
├── Gradients:                14GB  (for all params)
├── Activations:              2GB
└── Total:                    ~86GB  (5x more!)
```

### During Inference

```
GPU Memory Usage:
├── Base Model:               14GB
├── LoRA Adapters:            32MB
├── Activations:              2GB
└── Total:                    ~16GB

Can load multiple adapters:
├── Base Model:               14GB  (shared)
├── Billing Adapter:          32MB
├── Technical Adapter:        32MB
├── Product Adapter:          32MB
└── Total:                    ~14.1GB  (vs 42GB for 3 full models)
```

---

## 🎯 Real Example from This Project

### Training Billing Domain Model

```python
# 1. Initialize trainer
from src.models.fine_tuning.lora_trainer import lora_trainer

# 2. Prepare data
training_data = [
    {
        "instruction": "Why was I charged $50?",
        "response": "The $50 charge is for your monthly subscription renewal..."
    },
    # ... 1000 more examples
]

dataset = lora_trainer.prepare_dataset(training_data)

# 3. Train
model_path = lora_trainer.train(
    train_data=dataset,
    output_dir="models/fine_tuned/billing",
    epochs=3,
    batch_size=4,
    learning_rate=2e-4
)

# Output:
# Starting LoRA fine-tuning...
# trainable params: 8,388,608 || all params: 6,738,415,616 || trainable%: 0.12
# Epoch 1/3: Loss = 2.345 → 1.234
# Epoch 2/3: Loss = 1.123 → 0.987
# Epoch 3/3: Loss = 0.856 → 0.745
# Fine-tuned model saved to models/fine_tuned/billing
```

### Using Fine-tuned Model

```python
# Generate response
response = lora_trainer.generate_response(
    model_path="models/fine_tuned/billing",
    prompt="### Instruction:\nWhy was I charged $50?\n\n### Response:\n",
    max_length=200
)

# Output:
# "The $50 charge is for your monthly subscription renewal. 
#  This is an automatic charge that occurs on the same date 
#  each month. You can view your billing history in Account Settings."
```

---

## 🔑 Key Points

1. **Only 0.12% parameters trained** (8M out of 7B)
2. **Base model stays frozen** (never updated)
3. **LoRA adapters are small** (32MB vs 14GB)
4. **Training is fast** (hours vs days)
5. **Multiple adapters possible** (one base, many domains)
6. **Performance is excellent** (95-98% of full fine-tuning)

---

**LoRA makes fine-tuning practical and cost-effective!** 🚀

