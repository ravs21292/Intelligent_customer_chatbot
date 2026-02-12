"""LoRA-based fine-tuning for efficient model training."""

import os
import json
from typing import Dict, Any, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from config.model_config import model_config
from config.pipeline_config import pipeline_config
from src.utils.logger import logger


class LoRATrainer:
    """Trains models using LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
    
    def __init__(self):
        self.base_model = model_config.FINE_TUNE_BASE_MODEL
        self.lora_rank = model_config.LORA_RANK
        self.lora_alpha = model_config.LORA_ALPHA
        self.lora_dropout = model_config.LORA_DROPOUT
    
    def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        format_type: str = "instruction"
    ) -> Dataset:
        """
        Prepare dataset for fine-tuning.
        
        Args:
            data: List of training examples with 'instruction' and 'response' keys
            format_type: Format type ('instruction' or 'conversation')
            
        Returns:
            HuggingFace Dataset
        """
        if format_type == "instruction":
            formatted_data = [
                {
                    "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
                }
                for item in data
            ]
        else:
            formatted_data = [
                {
                    "text": f"User: {item.get('user', '')}\nAssistant: {item.get('assistant', '')}"
                }
                for item in data
            ]
        
        return Dataset.from_list(formatted_data)
    
    def train(
        self,
        train_data: Dataset,
        val_data: Optional[Dataset] = None,
        output_dir: str = "models/fine_tuned",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ) -> str:
        """
        Fine-tune model using LoRA.
        
        Args:
            train_data: Training dataset
            val_data: Optional validation dataset
            output_dir: Output directory
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Path to saved model
        """
        logger.info("Starting LoRA fine-tuning...")
        
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj"]  # Adjust based on model architecture
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        train_dataset = train_data.map(tokenize_function, batched=True)
        if val_data:
            val_dataset = val_data.map(tokenize_function, batched=True)
        else:
            val_dataset = None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
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
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuned model saved to {output_dir}")
        return output_dir
    
    def generate_response(
        self,
        model_path: str,
        prompt: str,
        max_length: int = 200
    ) -> str:
        """
        Generate response using fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated response
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.eval()
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""


# Global LoRA trainer instance
lora_trainer = LoRATrainer()

