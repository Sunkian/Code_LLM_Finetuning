import os
import random
import json
import re
import time
from tqdm import tqdm as tqdm_bar
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# Load your dataset
dataset = load_dataset('json', data_files='/home/ec2-user/EvolInstruct/alice/Instruction_Filters/ALL_COMBINED/FILTEREDevolved_codealpaca_v1codellamainst70B.jsonl')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Nondzu/Mistral-7B-codealpaca-lora', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('Nondzu/Mistral-7B-codealpaca-lora')

# Configure the model for LoRA fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # This is the LoRA rank parameter
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj']  # Adjust based on model architecture
)

# Apply the LoRA configuration
model = get_peft_model(model, lora_config)

# Preprocess the data
def preprocess_function(examples):
    inputs = examples["instruction"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="/opt/ml/model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="/opt/ml/logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],  # If you have a validation split
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('/opt/ml/model')
tokenizer.save_pretrained('/opt/ml/model')
