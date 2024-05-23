FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install transformers datasets peft sagemaker

COPY fine-tuning_sagemaker.py /opt/ml/code/fine-tuning_sagemaker.py

ENV SM_MODEL_DIR=/opt/ml/model
ENV SM_OUTPUT_DATA_DIR=/opt/ml/output/data
ENV SM_CHANNEL_TRAINING=/opt/ml/input/data/training

ENTRYPOINT ["python", "/opt/ml/code/fine-tuning_sagemaker.py"]
