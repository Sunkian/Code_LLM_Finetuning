import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='Nondzu/Mistral-7B-codealpaca-lora')
    parser.add_argument('--train_file', type=str, default='s3://tii-llm-code-reboot/datasets/alice-evolinstruct-codealpaca/capstone/FILTEREDevolved_codealpaca_v1codellamainst70B.jsonl')
    parser.add_argument('--output_dir', type=str, default='s3://tii-llm-code-reboot/datasets/alice-evolinstruct-codealpaca/capstone/')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data_files = {"train": args.train_file}
    dataset = load_dataset('json', data_files=data_files)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
    )

    trainer.train()

if __name__ == '__main__':
    main()
