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

STATS_ON = False
confirm = input("Add statistics? [y/N] >> ").strip()
if confirm.lower()[0] == "y":
    STATS_ON = True

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

CHUNK_STRATEGIES = [
    {"type": "time", "period": "daily"},    
    {"type": "rows", "size": 10},
    {"type": "time", "period": "monthly"},   
    {"type": "file", "period": "yearly"}  
]
BATCH_SIZE = 128          # points per upsert

def calculate_chunk_statistics(chunk_rows, headers):
    """Calculate statistics for a chunk of CSV rows"""
    if not chunk_rows:
        return {}
    
    header_list = headers.split(',')
    stats = {}
    
    # Convert chunk rows to DataFrame
    chunk_data = [row.split(',') for row in chunk_rows]
    df_chunk = pd.DataFrame(chunk_data, columns=header_list)
    
    # Calculate average for EVERY column (except Date)
    for column_name in df_chunk.columns:
        if column_name.lower() == 'Date':
            continue
            
        cleaned_series = df_chunk[column_name].str.replace('$', '', regex=False).str.strip()
        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
        valid_values = numeric_series.dropna()
        
        if len(valid_values) > 0:  # If we have valid numbers
            stats[f"{column_name}_avg"] = float(valid_values.mean())
    
    stats["rows_analyzed"] = len(chunk_rows)
    
    return stats

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

        if STATS_ON:
            statistics = calculate_chunk_statistics(chunk_rows, headers)

            stats_summary = ""
            if statistics and not statistics.get('error'):
                if 'close_avg' in statistics:
                    stats_summary = f" | Avg Close: ${statistics['close_avg']:.2f}"
                if 'price_change_pct' in statistics:
                    stats_summary += f" | Trend: {statistics['price_change_pct']:+.2f}%"
                if 'volume_avg' in statistics:
                    stats_summary += f" | Avg Vol: {statistics['volume_avg']:,.0f}"

        # Create enhanced header
        if month and year:
            enhanced_header = f"{headers}"
        else:
            enhanced_header = f"{headers}"

        # Prepend enhanced headers to each chunk
        chunk_with_headers = enhanced_header + "\n" + "\n".join(chunk_rows)
        yield chunk_with_headers, month, year

def chunk_by_time_period(df, headers, company, period_type):
    """Chunk data by day, month, or year"""
    if df['Date'] is None or df['Date'].isna().all():
        print(f"\tCannot chunk by {period_type} - no valid dates")
        return []
    
    chunks = []
    
    if period_type == 'daily':
        # Each day is its own chunk
        for date, day_data in df.groupby(df['Date'].dt.date):
            # Convert DataFrame rows back to original CSV string format
            chunk_rows = []
            for _, row in day_data.iterrows():
                row_values = []
                for col in df.columns:
                    if col == 'Date':
                        # Format date as YYYY-MM-DD only (no time)
                        row_values.append(row[col].strftime('%Y-%m-%d'))
                    else:
                        row_values.append(str(row[col]))
                chunk_rows.append(','.join(row_values))
            
            month = date.strftime('%B')
            year = date.year
            chunks.append((chunk_rows, month, year, 'daily'))
    
    elif period_type == 'monthly':
        # Each month is its own chunk
        for (year, month), month_data in df.groupby([df['Date'].dt.year, df['Date'].dt.month]):
            # Convert DataFrame rows back to original CSV string format
            chunk_rows = []
            for _, row in month_data.iterrows():
                row_values = []
                for col in df.columns:
                    if col == 'Date':
                        # Format date as YYYY-MM-DD only (no time)
                        row_values.append(row[col].strftime('%Y-%m-%d'))
                    else:
                        row_values.append(str(row[col]))
                chunk_rows.append(','.join(row_values))
            
            month_name = datetime(year, month, 1).strftime('%B')
            chunks.append((chunk_rows, month_name, year, 'monthly'))
    
    elif period_type == 'yearly':
        # Each year is its own chunk
        for year, year_data in df.groupby(df['Date'].dt.year):
            # Convert DataFrame rows back to original CSV string format
            chunk_rows = []
            for _, row in year_data.iterrows():
                row_values = []
                for col in df.columns:
                    if col == 'Date':
                        # Format date as YYYY-MM-DD only (no time)
                        row_values.append(row[col].strftime('%Y-%m-%d'))
                    else:
                        row_values.append(str(row[col]))
                chunk_rows.append(','.join(row_values))
            
            chunks.append((chunk_rows, None, year, 'yearly'))
    
    return chunks

def chunk_by_rows(data_rows, headers, company, size):
    """Chunk data by fixed number of rows (your existing approach)"""
    chunks = []
    for start in range(0, len(data_rows), size):
        chunk_rows = data_rows[start:start+size]
        month, year = extract_date_from_row(chunk_rows[0])
        chunks.append((chunk_rows, month, year, f'rows_{size}'))
    return chunks

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
    header_list = headers.split(',')

    # Rest are data rows - keep as strings for consistency
    data_rows = [line.strip() for line in lines[1:] if line.strip()]

    if not data_rows:
        print(f"\tSKIPPED: {path.name} (no data rows)")
        return

    # Regex to get the year out of the filename (fallback)
    year_match = re.search(r"\d{4}", path.stem)
    file_year = year_match.group(0) if year_match else None

    # Create DataFrame for time-based chunking
    df = pd.DataFrame([row.split(',') for row in data_rows], columns=header_list)
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception as e:
            print(f"\tWarning: Could not parse dates in {path.name}: {e}")
            df['Date'] = None
    else:
        print(f"\tNo 'Date' column found in {path.name}. Available columns: {df.columns.tolist()}")
        df['Date'] = None

    num_chunks = 0
    
    for strategy in CHUNK_STRATEGIES:
        if strategy["type"] == "time" and df['Date'] is not None and not df['Date'].isna().all():
            # Time-based chunking (day/month) within this yearly file
            time_chunks = chunk_by_time_period(df, headers, company, strategy["period"])
            for chunk_rows, month, year, chunk_type in time_chunks:
                if not chunk_rows:
                    continue
                    
                # Use file_year as the primary year context
                effective_year = file_year or year
                
                # Convert list of rows back to string format
                chunk_content = headers + "\n" + "\n".join(chunk_rows)
                if STATS_ON:
                    statistics = calculate_chunk_statistics(chunk_rows, headers)
                else:
                    statistics = {}
                date_range = calculate_chunk_date_range(chunk_rows, headers)
                
                yield models.PointStruct(
                    id=str(uuid4()),
                    vector=models.Document(text=chunk_content, model=model_name),
                    payload={
                        "document": path.name,
                        "content": chunk_content,
                        "company": company,
                        "source_type": 'csv',
                        "month": month,
                        "year": effective_year,
                        "file_year": file_year,
                        "headers": headers,
                        "chunk_type": chunk_type,
                        "chunk_strategy": f"time_{strategy['period']}",
                        "date_range": date_range,
                        "statistics": statistics,
                    },
                )
                num_chunks += 1
                
        elif strategy["type"] == "rows":
            # Row-based chunking within this yearly file - use original string data
            row_chunks = chunk_by_rows(data_rows, headers, company, strategy["size"])
            for chunk_rows, month, year, chunk_type in row_chunks:
                # Use file_year as the primary year context
                effective_year = file_year or year
                
                chunk_content = headers + "\n" + "\n".join(chunk_rows)
                if STATS_ON:
                    statistics = calculate_chunk_statistics(chunk_rows, headers)
                else:
                    statistics = {}
                date_range = calculate_chunk_date_range(chunk_rows, headers)
                
                yield models.PointStruct(
                    id=str(uuid4()),
                    vector=models.Document(text=chunk_content, model=model_name),
                    payload={
                        "document": path.name,
                        "content": chunk_content,
                        "company": company,
                        "source_type": 'csv',
                        "month": month,
                        "year": effective_year,
                        "file_year": file_year,
                        "headers": headers,
                        "chunk_type": chunk_type,
                        "chunk_strategy": f"rows_{strategy['size']}",
                        "date_range": date_range,
                        "statistics": statistics,
                    },
                )
                num_chunks += 1

        elif strategy["type"] == "file":
            # Entire file as one chunk (yearly summary) - use original string data
            chunk_content = headers + "\n" + "\n".join(data_rows)
            if STATS_ON:
                statistics = calculate_chunk_statistics(data_rows, headers)
            else:
                statistics = {}
            date_range = calculate_chunk_date_range(data_rows, headers)
            
            yield models.PointStruct(
                id=str(uuid4()),
                vector=models.Document(text=chunk_content, model=model_name),
                payload={
                    "document": path.name,
                    "content": chunk_content,
                    "company": company,
                    "source_type": 'csv',
                    "month": None,
                    "year": file_year,
                    "file_year": file_year,
                    "headers": headers,
                    "chunk_type": 'yearly',
                    "chunk_strategy": "file_yearly",
                    "date_range": date_range,
                    "statistics": statistics,
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
