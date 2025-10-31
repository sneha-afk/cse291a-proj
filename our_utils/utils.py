from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

# Returns collection name and model name, MUST be consistent with using embeddings and generating
def get_qdrant_config() -> tuple[str, str]:
    # Name of collection on Qdrant
    collection_name = "knowledge_base"

    # Embedding model being used
    model_name = "BAAI/bge-small-en-v1.5"

    return collection_name, model_name


def get_qdrant_client() -> QdrantClient:
    # Load environment variables from .env file
    found = load_dotenv()
    if not found:
        print("Error: ensure you have a .env with QDRANT_URL set (e.x. http://localhost:6333)")
        exit(1)

    # Read Qdrant credentials
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_URL:
        print("Error: ensure you have a .env with QDRANT_URL set (e.x. http://localhost:6333)")
        exit(1)

    # Initialize the Qdrant client
    # Skip API key if running locally
    if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
        print(f"Running Qdrant locally at: {QDRANT_URL}")
        client = QdrantClient(url=QDRANT_URL)
    else:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    return client
