"""
Retrevial script of relevant docs
"""
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
from ollama import chat
from ollama import ChatResponse

# Load environment variables from .env file
load_dotenv()

# Read Qdrant credentials
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Name of collection on Qdrant
collection_name = "knowledge_base"

# Embedding model being used
model_name = "BAAI/bge-small-en-v1.5"

# Initialize the Qdrant client
# Skip API key if running locally
if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
    print(QDRANT_URL) 
    client = QdrantClient(url=QDRANT_URL)
else:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

question = "What's the difference between Meta and Microsoft in performance"
n_points = 10

results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=question, model=model_name),
        limit=n_points,
    )
context = "\n".join(f"Relevant Document {i}, {r.payload["document"]}: {r.payload["content"]}" for i, r in enumerate(results.points))
docs = "\n".join(f"Relevant Document {i}, {r.payload["document"]}" for i, r in enumerate(results.points))
print(docs)