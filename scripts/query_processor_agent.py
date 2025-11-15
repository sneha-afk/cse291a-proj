"""
Processes user query into a retrieval query for CSV data (via AWS Bedrock)
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

# Use your Bedrock GPT-OSS model here
MODEL_ID = "openai.gpt-oss-20b-1:0"  # same family you used in your other script


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
    metaprompt = """
You are an assistant that rewrites a natural language financial query into a Qdrant search query JSON
for better retrieval over CSV stock data.

Given a user query of: 
"Find the performance of Apple stock in the second quarter 2023 and compare it with Microsoft in second quarter 2023"

Here's an example of what a Qdrant query would look like:

{
  "searches": [
    {
      "limit": 1000,
      "with_payload": true,
      "with_vector": false,
      "filter": {
        "must": [
          { "key": "ticker", "match": { "value": "AAPL" } },
          { "key": "date_ts", "range": { "gte": 1680307200, "lt": 1688083200 } }
        ]
      }
    },
    {
      "limit": 1000,
      "with_payload": true,
      "with_vector": false,
      "filter": {
        "must": [
          { "key": "ticker", "match": { "value": "MSFT" } },
          { "key": "date_ts", "range": { "gte": 1680307200, "lt": 1688083200 } }
        ]
      }
    }
  ]
}

Your task:
- Read the user's question.
- Infer the correct tickers, date ranges, and any other needed filters.
- Return ONLY a valid JSON object in this exact "searches" format.
- Do NOT include backticks, explanations, or any surrounding text. Just JSON.
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
    # If you want a parsed Python dict instead of raw JSON string:
    # return json.loads(raw)
    return raw


if __name__ == "__main__":
    parse_query(
        "Compare tesla's performance in quarter 4 2023 against nvidia's in quarter 4 of 2023"
    )
