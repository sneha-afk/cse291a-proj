import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

collection_name = "knowledge_base"
embedding_model_name = "BAAI/bge-small-en-v1.5"

if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
    print(QDRANT_URL)
    qdrant_client = QdrantClient(url=QDRANT_URL)
else:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    

REGION="us-west-2" # don't change this

rewriter_model: str = "openai.gpt-oss-20b-1:0"
answer_model: str = "openai.gpt-oss-120b-1:0"

bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

def send_request(model: str, messages, print_prompt: bool = True) -> str:
    response = bedrock_client.converse(
        modelId=model,
        messages=messages,
    )

    print(f"Using model: {model}\n\n")
    if print_prompt:
        print(f"Input messages:")
        for msg_dict in messages:
            print(f"{msg_dict["role"]}: {msg_dict["content"][0]["text"]}")
    print("\n----\nResponse:")
    print(response['output']['message']['content'][-1]['text'])
    return response['output']['message']['content'][-1]['text']

def rewrite_prompt(original_question: str) -> str:
    sys_prompt = """
    Rewrite the following prompt to optimize it for efficient RAG retrieval. You can insert keywords, reword it, etc.
    Be brief when rewriting it, use bullet points if needed. Maximize keywords and potentially identifying files, i.e
    [COMPANY]_[YEAR].{txt,csv} and other helpful retrieval techniques.
    Our use case is providing accurate financial information given annual reports and CSVs containing stock information.
    """

    messages = [
        {"role": "user", "content": [{"text": sys_prompt}]},
        {"role": "user", "content": [{"text": original_question}]}
    ]

    return send_request(rewriter_model, messages)

def rag(question: str, n_points: int = 10):
    retrieval_optimized_prompt = rewrite_prompt(question)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=models.Document(text=retrieval_optimized_prompt, model=embedding_model_name),
        limit=n_points,
    )

    context = "\n".join(f"Relevant Document {i}, {r.payload["document"]}: {r.payload["content"]}" for i, r in enumerate(results.points))
    docs = "\n".join(f"Relevant Document {i}, {r.payload["document"]}, chunk index {r.payload["part_index"]}" for i, r in enumerate(results.points))
    print("Retrieved the following docs:")
    print(docs)

    metaprompt = f"""
    Answer the following question using the provided context.
    If you can't find the answer, do not pretend you know it, explain why you are unable to answer the prompt.

    Context:
    {context.strip()}

    Question:
    {question}
    """

    messages = [
        {"role": "user", "content": [{"text": metaprompt}]},
    ]

    _ = send_request(answer_model, messages, False)

original_prompt: str = "Compare the year-end revenues for Google Search in the past two years and provide insight into what factors contribute to the figures."
rag(original_prompt, n_points=10)
