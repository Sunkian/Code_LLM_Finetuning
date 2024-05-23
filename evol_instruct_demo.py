import os
import random
import json
import re
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set the cache directory
os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('/home/ec2-user/.cache/huggingface/hub')
print("Cache directory:", os.environ['TRANSFORMERS_CACHE'])

# Initialize the pipeline with a smaller model
print("Initializing pipeline...")
pipe = pipeline("text-generation", model="codellama/CodeLlama-7b-Instruct-hf")
print("Pipeline initialized")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
print("Tokenizer loaded")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
print("Model loaded")

# Define a prompt
prompt = "Once upon a time"

# Generate text using the pipeline
print("Generating text...")
generated_text = pipe(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print("Generated text:")
print(generated_text[0]['generated_text'])

seed_tasks_path = "/home/ec2-user/EvolInstruct/converted_alpaca_20k.json"

def load_instructions(file_path: str):
    """Load JSON file in Evol Format"""
    with open(file_path, "r") as json_file:
        return json.load(json_file)

# Test the loading function
print("Loading instructions...")
instructions = load_instructions(seed_tasks_path)
print("Instructions loaded")
