"""
Sets up embeddings in Qdrant
"""
from uuid import uuid4
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
from pathlib import Path
from ollama import chat
from ollama import ChatResponse

# Load environment variables from .env file
load_dotenv()

# Read Qdrant credentials
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize the Qdrant client
# Skip API key if running locally
if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
    print(QDRANT_URL) 
    client = QdrantClient(url=QDRANT_URL, timeout=60)
else:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Name of collection on Qdrant
collection_name = "knowledge_base"

# Embedding model being used
model_name = "BAAI/bge-small-en-v1.5"

#WARNING DELETE
client.delete_collection(collection_name=collection_name)

# Check if collection exists
collections = [col.name for col in client.get_collections().collections]
if collection_name not in collections:
    print(f"Creating new collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        ),
    )
else:
    print(f"Collection '{collection_name}' already exists — skipping creation.")



# # Loop through all .txt files in the folder
# documents = []
# for file_path in folder.rglob("*2022.txt"):
#     with file_path.open("r", encoding="utf-8") as f:
#         content = f.read()
#         documents.append((file_path.name, content))


CHUNK_SIZE = 1024         # chars per chunk
BATCH_SIZE = 256          # points per upsert


def chunk_text(text: str, size: int):
    for start in range(0, len(text), size):
        yield text[start:start+size]


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def points_for_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    for part_idx, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
        yield models.PointStruct(
            id=str(uuid4()),  # unique per chunk
            vector=models.Document(text=chunk, model=model_name),
            payload={
                "document": path.name,
                "content": chunk,
                "part_index": part_idx,
                "year": 2022,
            },
        )

# Iterate files lazily, build points lazily, upsert in batches
def all_points():
    # Define the folder path
    folder = Path("dataset")
    for file_path in folder.rglob("*AMZN_2020.txt"):
        yield from points_for_file(file_path)
    for file_path in folder.rglob("*AMZN_2021.txt"):
        yield from points_for_file(file_path)

for batch in batched(all_points(), BATCH_SIZE):
    client.upsert(collection_name=collection_name, points=batch, wait=True)



