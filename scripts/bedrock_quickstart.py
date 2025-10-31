"""
To have your credentials set up, the easiest way is running:
    aws configure

ONLY USE us-west-2 AS THE SET REGION!

More setup details in README.

Bedrock Python client documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
"""

import boto3
import json
from botocore.exceptions import ClientError

REGION="us-west-2" # don't change this

model_id = "openai.gpt-oss-20b-1:0"

def get_model_names():
    resp = client.list_foundation_models()
    for model in resp["modelSummaries"]:
        print(model["modelId"], "-", model["modelName"])

# Uncomment to get a list of models
# get_model_names()

client = boto3.client("bedrock-runtime", region_name=REGION)

prompt = "In one line, explain what a RAG is."

messages = [{"role": "user", "content": [{"text": prompt}]}]

# API call to the model
response = client.converse(
    modelId=model_id,
    messages=messages,
)

print(f"Bedrock quickstart using: {model_id}")

print(f"User: {prompt}")

print("\n\nResponse:")
print(response['output']['message']['content'][-1]['text'])
