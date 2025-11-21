import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
import re
from typing import Tuple, List
from qdrant_client import QdrantClient, models

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

collection_name = "knowledge_base"
embedding_model_name = "BAAI/bge-small-en-v1.5"

if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
    print(QDRANT_URL)
    qdrant_client = QdrantClient(url=QDRANT_URL)
else:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

REGION = "us-west-2"

rewriter_model: str = "openai.gpt-oss-20b-1:0"
answer_model: str = "openai.gpt-oss-120b-1:0"

bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)


def send_request(model: str, messages, print_prompt: bool = True) -> str:
    response = bedrock_client.converse(
        modelId=model,
        messages=messages,
    )

    print(f"Using model: {model}\n\n")
    if print_prompt:
        print("Input messages:")
        for msg_dict in messages:
            print(f"{msg_dict['role']}: {msg_dict['content'][0]['text']}")
    print("\n----\nResponse:")
    text = response["output"]["message"]["content"][-1]["text"]
    print(text)
    return text


def extract_filters_from_query(rewritten_query: str) -> Tuple[List[str], List[int]]:
    """
    Extract company tickers and years from the rewritten query.
    Returns: (companies, years)
    """
    companies = []
    years = []
    
    # Common ticker patterns - extract all uppercase 2-5 letter sequences
    # that could be tickers (excluding common words)
    ticker_pattern = r'\b[A-Z]{2,5}\b'
    potential_tickers = re.findall(ticker_pattern, rewritten_query)
    
    # Filter out common non-ticker words
    exclude_words = {'CSV', 'PDF', 'AND', 'THE', 'FOR', 'WITH', 'PRICE', 'YEAR', 'DATE'}
    companies = [t for t in potential_tickers if t not in exclude_words]
    
    # Extract years (4-digit numbers starting with 19 or 20)
    year_pattern = r'\b(?:19|20)\d{2}\b'
    years = [int(y) for y in re.findall(year_pattern, rewritten_query)]
    
    # Remove duplicates while preserving order
    companies = list(dict.fromkeys(companies))
    years = list(dict.fromkeys(years))
    
    print(f"  Extracted companies: {companies}")
    print(f"  Extracted years: {years}")
    
    return companies, years


def rewrite_prompt_csv(original_question: str) -> Tuple[str, List[str], List[int]]:
    """
    Rewrite query for CSV and extract filters.
    Returns: (rewritten_query, companies, years)
    """
    sys_prompt = """
You are generating a dense-retrieval query for CSV stock data.

The CSV files have headers like:
- ticker, date, date_ts, open, high, low, close, volume, year, quarter

Rewrite the user's question into a short keyword query that:
- explicitly mentions tickers (use standard ticker symbols like AAPL, MSFT, AMZN, META, AVGO for Broadcom, TSM for TSMC)
- explicitly mentions years (e.g., 2023)
- uses words like: ticker, date_ts, close price, performance, quarter
- is a SINGLE line of plain text, no JSON, no punctuation except spaces and colons if helpful.
- if weeks are mentioned use daily day data (Mon-Fri) and use ISO dates 

Example output for:
"Find the performance of Apple stock in the second quarter 2023 and compare it with Microsoft in second quarter 2023"

Possible query:
"ticker:AAPL ticker:MSFT year:2023 quarter:Q2 close price performance comparison"
"""

    messages = [
        {"role": "user", "content": [{"text": sys_prompt}]},
        {"role": "user", "content": [{"text": original_question.strip()}]},
    ]

    rewritten = send_request(rewriter_model, messages)
    companies, years = extract_filters_from_query(rewritten)
    
    return rewritten, companies, years


def rewrite_prompt_pdf(original_question: str) -> Tuple[str, List[str], List[int]]:
    """
    Rewrite query for PDF and extract filters.
    Returns: (rewritten_query, companies, years)
    """
    sys_prompt = """
You are generating a retrieval query for PDF financial reports
such as 10-Ks, 10-Qs, and annual reports.

Rewrite the user's question into a short, dense query that:
- Mentions the company ticker symbols (use AAPL, MSFT, AMZN, META, AVGO for Broadcom, TSM for TSMC)
- Explicitly mentions years (e.g., 2023)
- Targets sections like "Results of Operations", "Segment Results", "Revenue", and
  commentary on growth, drivers, and profitability.
- Uses phrases that are likely to appear in those reports, like
  "year-over-year", "segment revenue", "results of operations", "management discussion".

Be concise. No explanations, just the optimized query text.
"""

    messages = [
        {"role": "user", "content": [{"text": sys_prompt}]},
        {"role": "user", "content": [{"text": original_question.strip()}]},
    ]

    rewritten = send_request(rewriter_model, messages)
    companies, years = extract_filters_from_query(rewritten)
    
    return rewritten, companies, years


def rag(question: str, n_points: int = 10):
    # 1) Get optimized queries and extract filters
    print("=" * 60)
    print("REWRITING CSV QUERY")
    print("=" * 60)
    csv_query, csv_companies, csv_years = rewrite_prompt_csv(question)
    
    print("\n" + "=" * 60)
    print("REWRITING PDF QUERY")
    print("=" * 60)
    pdf_query, pdf_companies, pdf_years = rewrite_prompt_pdf(question)
    
    all_points = []

    # 2) Query Qdrant for CSV chunks
    print("\n" + "=" * 60)
    print("QUERYING CSV DATA")
    print("=" * 60)
    
    if csv_companies and csv_years:
        print(csv_years)
        try:
            csv_filter_conditions = [
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value="csv"),
                ),
                models.FieldCondition(
                    key="year",
                    match=models.MatchAny(any=[str(pdf_years[0])]),
                ),
                models.FieldCondition(
                    key="company",
                    match=models.MatchAny(any=csv_companies),
                ),
            ]
            
            csv_results = qdrant_client.query_points(
                collection_name=collection_name,
                query=models.Document(text=csv_query, model=embedding_model_name),
                limit=465,
                query_filter=models.Filter(must=csv_filter_conditions)
            )
            print(f"✓ Found {len(csv_results.points)} CSV points")
            all_points.extend(csv_results.points)
        except Exception as e:
            print(f"✗ CSV query failed: {e}")
    else:
        print("⚠ Skipping CSV query - no companies or years extracted")

    # 3) Query Qdrant for PDF chunks
    print("\n" + "=" * 60)
    print("QUERYING PDF DATA")
    print("=" * 60)
    
    if pdf_companies and pdf_years:
        try:
            print(pdf_years)
            pdf_filter_conditions = [
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value="pdf"),
                ),
                models.FieldCondition(
                    key="year",
                    match=models.MatchAny(any=[str(pdf_years[0])]),
                ),
                models.FieldCondition(
                    key="company",
                    match=models.MatchAny(any=pdf_companies),
                ),
            ]
            
            pdf_results = qdrant_client.query_points(
                collection_name=collection_name,
                query=models.Document(text=pdf_query, model=embedding_model_name),
                limit=n_points,
                query_filter=models.Filter(must=pdf_filter_conditions)
            )
            print(f"✓ Found {len(pdf_results.points)} PDF points")
            all_points.extend(pdf_results.points)
        except Exception as e:
            print(f"✗ PDF query failed: {e}")
    else:
        print("⚠ Skipping PDF query - no companies or years extracted")

    if not all_points:
        print("\n[ERROR] No points retrieved from either CSV or PDF data.")
        return
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL RETRIEVED: {len(all_points)} points")
    print(f"{'=' * 60}\n")

    context_lines = []
    docs_lines = []

    for i, r in enumerate(all_points):
        payload = r.payload
        doc_name = payload.get("document", "unknown_document")
        content = payload.get("content", "")
        part_index = payload.get("part_index", "NA")
        source_type = payload.get("source_type", "unknown")

        context_lines.append(
            f"[{source_type.upper()}] Relevant Document {i}, {doc_name}: {content}"
        )
        docs_lines.append(
            f"Relevant Document {i}, {doc_name}, chunk index {part_index}, source_type={source_type}"
        )

    context = "\n".join(context_lines)
    docs = "\n".join(docs_lines)

    print("Retrieved the following docs:")
    print(docs)

    # 4) Ask the answer model with combined context
    metaprompt = f"""
Answer the following question using ONLY the provided context from both CSV data and PDF reports.
If you can't find the answer, do not pretend you know it; clearly explain why you are unable to answer.

Context:
{context.strip()}

Question:
{question}
"""

    messages = [
        {"role": "user", "content": [{"text": metaprompt}]},
    ]

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    _ = send_request(answer_model, messages, print_prompt=False)


if __name__ == "__main__":
    original_prompt: str = (
        "How many strictly positive return weeks did Apple have in 2024."
    )
    rag(original_prompt, n_points=100)