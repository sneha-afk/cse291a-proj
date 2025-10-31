"""
Sets up embeddings in Qdrant
"""
from uuid import uuid4
import re
from qdrant_client import QdrantClient, models
from pathlib import Path
from our_utils import get_qdrant_client, get_qdrant_config

collection_name, model_name = get_qdrant_config()
client: QdrantClient = get_qdrant_client()

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
BATCH_SIZE = 128          # points per upsert


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

    # Regex to get the year out of the filename
    year_match = re.search(r"\d{4}", path.stem)

    file_ext = path.suffix

    # Extra parsing for csv files can be done here:
    # if file_ext == "csv":
    #     pass

    num_chunks = 0
    for part_idx, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
        yield models.PointStruct(
            id=str(uuid4()),  # unique per chunk
            vector=models.Document(text=chunk, model=model_name),
            payload={
                "document": path.name,
                "content": chunk,
                "part_index": part_idx,
                "year": year_match.group(0) if year_match else None,
            },
        )
        num_chunks += 1

    print(f"\tPROCESSED: {path.name} -> {num_chunks} chunks generated")

# Iterate files lazily, build points lazily, upsert in batches
def all_points():
    # Define the folder path
    folder = Path("dataset")
    for file_path in folder.rglob("*GOOGL*.txt"):
        yield from points_for_file(file_path)
    for file_path in folder.rglob("*MSFT*.txt"):
        yield from points_for_file(file_path)
    for file_path in folder.rglob("*TSLA*.txt"):
        yield from points_for_file(file_path)
    for file_path in folder.rglob("*META*.txt"):
        yield from points_for_file(file_path)

for batch in batched(all_points(), BATCH_SIZE):
    client.upsert(collection_name=collection_name, points=batch, wait=True)
