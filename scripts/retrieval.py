"""
Retrevial script of relevant docs
"""
from qdrant_client import QdrantClient, models
from our_utils import get_qdrant_client, get_qdrant_config

collection_name, model_name = get_qdrant_config()
client: QdrantClient = get_qdrant_client()

question = "Did Microsoft’s stock performance reflect its reported cash from operations trends from 2024?"
n_points = 25

results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=question, model=model_name),
        limit=n_points,
    )

if results:
    docs = "\n".join(f"Relevant Document {i}, {r.payload["document"]}, chunk index {r.payload["part_index"]}" for i, r in enumerate(results.points))
    print(docs)
else:
    print("No points were fetched for this query (did you embed them?)")
