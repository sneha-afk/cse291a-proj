"""
Sets up embeddings in Qdrant for CSV files
"""
from uuid import uuid4
from dotenv import load_dotenv
import os
import re
from qdrant_client import QdrantClient, models
from pathlib import Path
from datetime import datetime

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


CHUNK_SIZE = 1           # rows per chunk (adjust as needed)
BATCH_SIZE = 128          # points per upsert


def extract_date_from_row(row: str):
    """
    Extracts date from a CSV row where the first column is the date.
    Expected format: 2024-12-31,$189.30 ,17466920,$191.08 ,$191.96 ,$188.51
    """
    try:
        # Get the first column (date)
        date_str = row.split(',')[0].strip()
        
        # Parse the date (YYYY-MM-DD format)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Return month name and year
        return date_obj.strftime('%B'), date_obj.year
    except Exception as e:
        print(f"\t\tWarning: Could not parse date from row: {e}")
        return None, None


def chunk_csv_rows(rows: list, headers: str, company: str, size: int):
    """
    Chunks CSV rows and prepends headers with company, month, and year to each chunk.
    
    Args:
        rows: List of CSV rows (as strings)
        headers: Header row string
        company: Company name (e.g., 'GOOGL')
        size: Number of data rows per chunk
    """
    for start in range(0, len(rows), size):
        chunk_rows = rows[start:start+size]
        
        # Extract date from first row of this chunk
        month, year = extract_date_from_row(chunk_rows[0])
        
        # Create enhanced header
        if month and year:
            enhanced_header = f"{headers}"
        else:
            enhanced_header = f"{headers}"
        
        # Prepend enhanced headers to each chunk
        chunk_with_headers = enhanced_header + "\n" + "\n".join(chunk_rows)
        yield chunk_with_headers, month, year


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def points_for_csv_file(path: Path, company: str):
    with path.open("r", encoding="utf-8") as f:
        # Read all lines
        lines = f.readlines()
    
    if not lines:
        print(f"\tSKIPPED: {path.name} (empty file)")
        return
    
    # First line is headers
    headers = lines[0].strip()
    
    # Rest are data rows
    data_rows = [line.strip() for line in lines[1:] if line.strip()]
    
    if not data_rows:
        print(f"\tSKIPPED: {path.name} (no data rows)")
        return

    # Regex to get the year out of the filename (fallback)
    year_match = re.search(r"\d{4}", path.stem)

    num_chunks = 0
    for part_idx, (chunk, month, year) in enumerate(chunk_csv_rows(data_rows, headers, company, CHUNK_SIZE)):
        yield models.PointStruct(
            id=str(uuid4()),  # unique per chunk
            vector=models.Document(text=chunk, model=model_name),
            payload={
                "document": path.name,
                "content": chunk,
                "part_index": part_idx,
                "company": company,
                "month": month,
                "year": year if year else (year_match.group(0) if year_match else None),
                "headers": headers,  # Store original headers separately if needed
            },
        )
        num_chunks += 1

    print(f"\tPROCESSED: {path.name} -> {num_chunks} chunks generated")


# Iterate files lazily, build points lazily, upsert in batches
def all_points():
    # Define the folder path
    folder = Path("dataset")
    for file_path in folder.rglob("proc_GOOGL*.csv"):
        yield from points_for_csv_file(file_path, "GOOGL")
    for file_path in folder.rglob("proc_MSFT*.csv"):
        yield from points_for_csv_file(file_path, "MSFT")
    for file_path in folder.rglob("proc_TSLA*.csv"):
        yield from points_for_csv_file(file_path, "TSLA")
    for file_path in folder.rglob("proc_META*.csv"):
        yield from points_for_csv_file(file_path, "META")

for batch in batched(all_points(), BATCH_SIZE):
    client.upsert(collection_name=collection_name, points=batch, wait=True)

print("\nAll CSV files processed and uploaded to Qdrant!")