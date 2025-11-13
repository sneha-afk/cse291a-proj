import boto3, botocore
import os

# Set the API key as environment keys OR using aws configure
# Create the Bedrock client

client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"
)

# print("boto3:", boto3.__version__, "botocore:", botocore.__version__)

sts = boto3.client("sts", region_name="us-west-2")
# print(sts.get_caller_identity())


# Define the model and message
model_id = "openai.gpt-oss-120b-1:0"
messages = [{"role": "user", "content": [{"text": "Hello! Can you tell me about Amazon Bedrock?"}]}]

# Make the API call
response = client.converse(
    modelId=model_id,
    messages=messages,
)

# Print the response

print(response['output']['message']['content'][0]['reasoningContent']['reasoningText']['text'])
