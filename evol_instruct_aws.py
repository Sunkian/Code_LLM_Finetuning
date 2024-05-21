import random
from TextGenerationInference import TGI, GenerateRequest, GenerateParameters
import json 
import re
import time 
from tqdm import tqdm as tqdm_bar

tgi_client = TGI(endpoint_name="huggingface-pytorch-tgi-inference-2024-04-18-07-21-41-572", region_name="us-east-1")

#seed_tasks_path = "./merged_good_codealpaca.json"
seed_tasks_path = "/home/ec2-user/EvolInstruct/converted_alpaca_20k.json"

def load_instructions(file_path: str):
    """Load JSON file in Evol Format"""
    with open(file_path, "r") as json_file:
        return json.load(json_file)

def evolve_instructions(instructions, tgi):
    methods = [
        'Add new constraints and requirements to the original problem, adding approximately 10 additional words.',
        'Replace a commonly used requirement in the programming task with a less common and more specific one.',
        'If the original problem can be solved with only a few logical steps, please add more reasoning steps.',
        'Provide a piece of erroneous code as a reference to increase misdirection.',
        'Propose higher time or space complexity requirements, but please refrain from doing so frequently.'
    ]

    tasks_evolution = []

    # Generate prompts for batch processing
    prompts = []
    for task in instructions:
        chosen_method = random.choice(methods)
        prompt = f"Please increase the difficulty of the given programming test question a bit without using the following method: '{chosen_method}'.\n\n#Given Test#\n{task['instruction']}\n\n#Rewritten Test#\n"
        prompts.append(prompt)

    params = GenerateParameters(max_new_tokens=512, temperature=0.7, stop=["\n\n", "def ", "\"\"\""])

    # Batch generate responses
    batch_size = 64
    batch_responses = batch_generate(tgi, params, prompts, batch_size)

    # Parse responses and create tasks_evolution
    for i, task in enumerate(instructions):
        response = batch_responses[i]

        # Extracting content after "#Rewritten Test#"
        rewritten_content_start = response.find('#Rewritten Test#')
        if rewritten_content_start != -1:
            # Add the length of '#Rewritten Test#\n' to start extracting after this marker
            rewritten_instruction = response[rewritten_content_start + len('#Rewritten Test#\n'):]
        else:
            rewritten_instruction = "Rewritten instruction not found."
        
        tasks_evolution.append({
            "original_instruction": task['instruction'],
            "original_output" : task['output'],
            "chosen_method": random.choice(methods),  # Consider choosing the method outside the loop if it needs to match with the prompt
            "rewritten_instruction": rewritten_instruction.strip()  # Strip to remove leading/trailing whitespace
        })

    return tasks_evolution

def remove_instruction_duplication(rewritten_instruction, rewritten_output):
    """
    Removes duplicated rewritten_instruction content from the start of rewritten_output.
    """
    # Normalize both strings for a fair comparison
    normalized_instruction = rewritten_instruction.lower().strip()
    normalized_output = rewritten_output.lower().strip()

    # Find the end of the overlapping part
    overlap_end = 0
    for i in range(min(len(normalized_instruction), len(normalized_output))):
        if normalized_instruction[:i+1] == normalized_output[:i+1]:
            overlap_end = i+1
        else:
            break

    # Remove the overlapping part from the original rewritten_output
    if overlap_end > 0:
        # Use the original casing and whitespace by slicing the original rewritten_output
        cleaned_output = rewritten_output[overlap_end:].strip()
    else:
        cleaned_output = rewritten_output

    return cleaned_output

def clean_code_section(rewritten_output):
    """
    First tries to extract the section after '#Code#' and then removes specified stop words and duplicated instruction.
    """
    # Define stop words that should be removed from the output
    stop_words = ["#TODO", "#Output#", "#Explanation#", "#Result#", '#Example Output#', '#Example Output#', "#Code#", "#Example#"]
    
    # Attempt to extract everything after '#Code#'
    code_start_index = rewritten_output.find('#Code#')
    if code_start_index != -1:
        cleaned_output = rewritten_output[code_start_index + len('#Code#\n'):]
    else:
        cleaned_output = rewritten_output

    # Remove the stop words from the cleaned_output
    for stop_word in stop_words:
        cleaned_output = cleaned_output.replace(stop_word, '')

    # Additional clean-up to remove excessive newlines or spaces
    cleaned_output = re.sub('\n\s*\n', '\n\n', cleaned_output)
    cleaned_output = cleaned_output.strip()

    return cleaned_output

# Filter tasks based on the length of the rewritten_instruction
def length_filter(tasks):
    filtered_tasks = []
    filtered_out_tasks = []  # To track tasks filtered out by length
    for task in tasks:
        rewritten_instruction = task["rewritten_instruction"].strip()
        if rewritten_instruction and 5 <= len(rewritten_instruction) <= 700:
            filtered_tasks.append(task)
        else:
            filtered_out_tasks.append(task)
    return filtered_tasks

def generate_responses(tasks_evolution, tgi):
    prompts = []
    for task in tasks_evolution:
        prompt = task['rewritten_instruction'] + 'Please provide the Python code to achieve the above\n' + '\n#Code#'
        prompts.append(prompt)
        
    params = GenerateParameters(max_new_tokens=512, temperature=0.2, stop=["#Code#", "#TODO", "#Result#", "#Output#", "#Explanation#"])
    batch_size = 64
    batch_responses = batch_generate(tgi, params, prompts, batch_size)

    # Initialize an empty list to store tasks that meet all criteria, including having non-empty rewritten_output
    filtered_tasks = []

    for i, task in enumerate(tasks_evolution):
        rewritten_response = batch_responses[i]
        # Apply cleaning and validation steps to the response
        cleaned_response = clean_code_section(rewritten_response)
        cleaned_response = remove_instruction_duplication(task["rewritten_instruction"], cleaned_response)

        # Add an additional check to ensure rewritten_output is not empty
        if cleaned_response.strip():  # Checks if cleaned_response is not empty or just whitespace
            task["rewritten_output"] = cleaned_response
            filtered_tasks.append(task)
    
    return filtered_tasks



def batch_generate(model, params, prompts, batch_size):
    all_results = []
    for i in tqdm_bar(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i: i + batch_size]
        requests = [GenerateRequest(inputs=prompt, parameters=params) for prompt in batch_prompts]
        batch_responses = model.create_from_objects(requests)
        all_results.extend(batch_responses)
    return all_results

if __name__ == "__main__":
        start_time = time.time()
        prev_tasks = load_instructions(seed_tasks_path)
        
        evolutions = 10
        all_tasks = []  # List to store tasks for all evolutions
        for evolution in range(1, evolutions + 1):
            print(f'Evolution {evolution}:')
            print("Generating New Instructions and their answers")
            final_tasks = []
            final_tasks.extend(generate_responses(evolve_instructions(prev_tasks, tgi_client), tgi_client))
            filtered = length_filter(final_tasks)
            all_tasks.extend(filtered)  # Add tasks for the current evolution to the list
            print('Done !')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total execution time: {execution_time} seconds")

        # Write all tasks to a single JSON file
        with open('evolved_codealpaca_v1codellamainst70B_10gens.json', 'w') as json_file:
            json.dump(all_tasks, json_file, indent=4)
