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
import pandas as pd

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
confirm = input("Delete the previous knowledge base? (CAREFUL!) [y/N] >> ").strip()
if confirm.lower()[0] == "y":
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


CHUNK_SIZES = [1,10,30]           # rows per chunk (adjust as needed)
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

    df = pd.DataFrame([row.split(',') for row in data_rows], columns=headers.split(','))
    if 'date' in headers:
        df['date'] = pd.to_datetime(df['date'])

    num_chunks = 0
    for chunk_size in CHUNK_SIZES:
        for part_idx, (chunk, month, year) in enumerate(chunk_csv_rows(data_rows, headers, company, chunk_size)):
            start_idx = part_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(data_rows))
            chunk_data_rows = data_rows[start_idx:end_idx]

            date_range = calculate_chunk_date_range(chunk_data_rows, headers)
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
                    "headers": headers,
                    "chunk_size": chunk_size,
                    "date_range": date_range,
                },
            )
            num_chunks += 1

    print(f"\tPROCESSED: {path.name} -> {num_chunks} chunks generated")

def calculate_chunk_date_range(chunk_rows, headers):
    """Calculate start and end dates for a chunk of CSV rows"""

    return {"start": chunk_rows[0][0:10], "end": chunk_rows[-1][0:10]}

def all_points():
    # Define the folder path
    folder = Path("dataset")
    for file_path in folder.rglob("proc_*.csv"):
        filename = file_path.stem
        ticker = filename.replace("proc_", "").split('-')[0]
        yield from points_for_csv_file(file_path, ticker)

for batch in batched(all_points(), BATCH_SIZE):
    client.upsert(collection_name=collection_name, points=batch, wait=True)

print("\nAll CSV files processed and uploaded to Qdrant!")
