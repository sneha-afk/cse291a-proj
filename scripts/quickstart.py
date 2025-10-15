"""
Run `uv sync` to make sure everything is installed

Seting up Qdrant locally with Docker:
    docker pull qdrant/qdrant
    docker run -p 6333:6333 qdrant/qdrant

There's .env-example at root, rename to .env and have: (no API key needed for a local instance)
    QDRANT_URL=http://localhost:6333
    QDRANT_API_KEY=

Install Ollama and run the gpt-oss:20b (lower parameter model)
    ollama run gpt-oss:20b
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

# Initialize the Qdrant client
# Skip API key if running locally
if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL)
else:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Name of collection on Qdrant
collection_name = "knowledge_base"

# Embedding model being used
model_name = "BAAI/bge-small-en-v1.5"

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

documents = [
    "Qdrant is a vector database & vector similarity search engine. It deploys as an API service providing search for the nearest high-dimensional vectors. With Qdrant, embeddings or neural network encoders can be turned into full-fledged applications for matching, searching, recommending, and much more!",
    "Docker helps developers build, share, and run applications anywhere — without tedious environment configuration or management.",
    "PyTorch is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing.",
    "MySQL is an open-source relational database management system (RDBMS). A relational database organizes data into one or more data tables in which data may be related to each other; these relations help structure the data. SQL is a language that programmers use to create, modify and extract data from the relational database, as well as control user access to the database.",
    "NGINX is a free, open-source, high-performance HTTP server and reverse proxy, as well as an IMAP/POP3 proxy server. NGINX is known for its high performance, stability, rich feature set, simple configuration, and low resource consumption.",
    "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
    "SentenceTransformers is a Python framework for state-of-the-art sentence, text and image embeddings. You can use this framework to compute sentence / text embeddings for more than 100 languages. These embeddings can then be compared e.g. with cosine-similarity to find sentences with a similar meaning. This can be useful for semantic textual similar, semantic search, or paraphrase mining.",
    "The cron command-line utility is a job scheduler on Unix-like operating systems. Users who set up and maintain software environments use cron to schedule jobs (commands or shell scripts), also known as cron jobs, to run periodically at fixed times, dates, or intervals.",
]
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=idx,
            vector=models.Document(text=document, model=model_name),
            payload={"document": document},
        )
        for idx, document in enumerate(documents)
    ],
)

def rag(question: str, n_points: int = 3) -> str:
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=question, model=model_name),
        limit=n_points,
    )

    context = "\n".join(r.payload["document"] for r in results.points)

    metaprompt = f"""
    Answer the following question using the provided context.
    If you can't find the answer, do not pretend you know it, but only answer "I don't know".

    Context:
    {context.strip()}
    """

    response: ChatResponse = chat(model='gpt-oss:20b', stream= True,
        messages=[
            {
                'role': 'system',
                'content': metaprompt
            },
            {
                'role': 'user',
                'content': question.strip()
            },
        ])

    # Receive the chunks from the streaming reponse, print as they arrive
    for chunk in response:
        print(chunk.message.content, end='', flush=True)

# Example use
rag("what is qdrant?")
