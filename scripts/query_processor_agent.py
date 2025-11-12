"""
Processes user query into a retrieval query for CSV data
"""

from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse

# Load environment variables from .env file
load_dotenv()


def parse_query(question: str):
    metaprompt = """Rewrite the user query into this qdrant query format for better retrieval.
    Given a user query of: "Find the performance of Apple stock in the second quarter 2023 and compare it with Microsoft in second quarter 2023"
    Here's an example of what a qdrant query would look like: 
    ```
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
    ```
    """

    response: ChatResponse = chat(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": metaprompt},
            {"role": "user", "content": question.strip()},
        ],
    )
    print(response.message.content)
    return response.message.content


parse_query(
    "Compare tesla's performance in quarter 4 2023 against nivida's in quarter 4 of 2023"
)
