"""
Processes user query into a knowledge docs query (via AWS Bedrock)
"""

import os
import json
from dotenv import load_dotenv
import boto3

# Load environment variables from .env file
load_dotenv()

# AWS / Bedrock config
REGION = os.getenv("AWS_REGION", "us-west-2") or "us-west-2"
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

# Same family you used in your other script
MODEL_ID = "openai.gpt-oss-20b-1:0"


def send_request_bedrock(model_id: str, messages: list, print_prompt: bool = True) -> str:
    """
    Thin wrapper around Bedrock converse API.
    messages should be a list of:
    { "role": "user" | "assistant" | "system", "content": [ { "text": "..." } ] }
    """
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
    )

    if print_prompt:
        print(f"Using model: {model_id}\n")
        print("Input messages:")
        for m in messages:
            print(f"{m['role']}: {m['content'][0]['text']}")
        print("\n----\nResponse:")

    output_text = response["output"]["message"]["content"][-1]["text"]
    print(output_text)
    return output_text


def parse_query(question: str):
    # I made the JSON example valid (quoted keys/strings) and told the model to return ONLY JSON.
    metaprompt = """
Rewrite the user query into a small JSON object for retrieval over knowledge documents.

Given a user query of:
"How does Tesla evaluate its energy segment growth and what strategies are they working on to increase its profitability"

Here's an example of what the JSON structure would look like:
{
  "company": "Tesla",
  "year": null,
  "tags": ["evaluate", "energy", "growth", "profitability", "strategies"]
}

Your task:
- Read the user's question.
- Infer the company (or companies), year (or null if not specified), and a short list of tags.
- Return ONLY a valid JSON object with the fields:
  - company: string or null (or a list of strings if multiple companies)
  - year: number or null
  - tags: array of short strings

Do NOT include backticks, prose, or any explanation. Just the JSON.
"""

    messages = [
        {
            "role": "user",
            "content": [{"text": metaprompt}],
        },
        {
            "role": "user",
            "content": [{"text": question.strip()}],
        },
    ]

    raw = send_request_bedrock(MODEL_ID, messages)
    # If you want a parsed Python dict:
    # return json.loads(raw)
    return raw


if __name__ == "__main__":
    parse_query(
        "Compare tesla's performance in quarter 4 2023 against nvidia's in quarter 4 of 2023"
    )