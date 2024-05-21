import os
import random
import json
import re
import time
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM


# Initialize the pipeline with the token
pipe = pipeline("text-generation", model="codellama/CodeLlama-70b-Instruct-hf")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-70b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-70b-Instruct-hf")

# Define a prompt
prompt = "Once upon a time"

# Generate text using the pipeline
generated_text = pipe(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(generated_text[0]['generated_text'])


seed_tasks_path = "/home/ec2-user/EvolInstruct/converted_alpaca_20k.json"

def load_instructions(file_path: str):
    """Load JSON file in Evol Format"""
    with open(file_path, "r") as json_file:
        return json.load(json_file)
