"""
Processes user query into a knowledge docs query
"""

from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse

# Load environment variables from .env file
load_dotenv()


def parse_query(question: str):
    metaprompt = """Rewrite the user query into this json for retrieval.
    Given a user query of: "How does Tesla evaluate its energy segment growth and what strategies are they working on to increase its profitability"
    Here's an example of what the json structure would look like: 
    ```
    {
        company: Tesla,
        year: None,
        tags: ["evaluate", "energy", "growth", "profitability", "strategies"],
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
