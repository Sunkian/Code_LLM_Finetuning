# M3.2 Capstone Project proposal : Fine-tune a Code LLM on Synthetic Data

Large Language Models (LLMs) for code, often referred to as Code LLMs, represent a significant
advancement in the intersection of artificial intelligence and software development. These models
are trained on vast amounts of code and accompanying documentation to understand and generate
programming languages. Their training can be enhanced by using not only real-world data but also
synthetic data, which helps in creating more robust models capable of understanding diverse coding
scenarios and handling edge cases effectively. This approach to training, involving both types of data,
is a key area of interest that we will explore in this project. The importance of Code LLMs extends
beyond mere productivity enhancements; they are fundamentally transforming how developers
interact with code, making software development more efficient and accessible, especially for those
who may not have extensive programming experience. As of now, Code LLMs are increasingly
integrated into development environments, providing real-time suggestions and corrections, thus
embedding AI deeply into the software creation process. Their evolution continues to be a dynamic
area of research and development, promising even more sophisticated tools and capabilities in the
near future. This project aims to propose a method to collect and prepare the data, as well as fine-
tuning a model and evaluate it.

## How to use

```sh
conda init
conda create -n evol-instruct python=3.11
conda activate evol-instruct

pip3 install -r requirements.txt
```

## Data source and preparation process

The seed dataset can be found here : https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json
We will leverage CodeLlama-70B-Instruct (https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf) to augment the real-life dataset code_alpaca_20k.

Steps :

1. Call a model from HuggingFace
2. Do a simple inference with this model 
3. Augment a dataset with the model