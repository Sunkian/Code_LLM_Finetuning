import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    # role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
    # role = "arn:aws:iam::296133277686:SMRole"
    role = iam.get_role(RoleName="SMRole")["Role"]["Arn"]

bucket_name = 'your-s3-bucket-name'
training_data_s3_path = 's3://tii-llm-code-reboot/datasets/alice-evolinstruct-codealpaca/capstone/FILTEREDevolved_codealpaca_v1codellamainst70B.jsonl'

estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.p4d.24xlarge',
    framework_version='1.10.0',
    py_version='py38',
    hyperparameters={
        'model_name_or_path': 'Nondzu/Mistral-7B-codealpaca-lora',
        'train_file': '/home/ec2-user/EvolInstruct/alice/Instruction_Filters/ALL_COMBINED/FILTEREDevolved_codealpaca_v1codellamainst70B.jsonl',
    },
    base_job_name='sagemaker-finetuning-codealpaca-alice'
)

estimator.fit({'training': training_data_s3_path})
