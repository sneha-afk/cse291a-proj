from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

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

#========================================================================

# See documentation for filters:
# https://qdrant.tech/documentation/concepts/filtering/
# "source_type" is either "pdf" or "csv"
# "company" is usually the uppercase stock symbol

# Example: deleting csvs from ORCL
delete_filter = rest.Filter(
    must=[
        rest.FieldCondition(
            key="company",
            match=rest.MatchValue(value="ORCL")
        ),
        rest.FieldCondition(
            key="source_type",
            match=rest.MatchValue(value="csv")
        ),
    ]
)

result = client.delete(
    collection_name=collection_name,
    points_selector=rest.FilterSelector(filter=delete_filter)
)

print("Deletion completed:", result)
