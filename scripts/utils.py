from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

# Returns the QdrantClient object, collection name, and embedding model name
def setup_qdrant() -> tuple[QdrantClient, str, str]:
    # Load environment variables from .env file
    found = load_dotenv()
    if not found:
        print("[ERROR] Did not find an .env")
        exit(1)

    # Read Qdrant credentials
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_URL:
        print("[ERROR] Did not set up .env correctly, did not find Qdrant URL")
        exit(1)

    # Name of collection on Qdrant
    collection_name = "knowledge_base"

    # Embedding model being used
    model_name = "BAAI/bge-small-en-v1.5"

    # Initialize the Qdrant client
    # Skip API key if running locally
    if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
        client = QdrantClient(url=QDRANT_URL)
    else:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("[INFO] Qdrant running at", QDRANT_URL)
    return client, collection_name, model_name
