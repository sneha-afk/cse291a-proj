import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
import datetime

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

REGION = "us-west-2"  # don't change this

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
            # careful with quotes
            print(f"{msg_dict['role']}: {msg_dict['content'][0]['text']}")
    print("\n----\nResponse:")
    text = response["output"]["message"]["content"][-1]["text"]
    print(text)
    return text


# ---------- CSV & PDF rewriters ----------

def rewrite_prompt_csv(original_question: str) -> str:
    sys_prompt = """
You are generating a dense-retrieval query for CSV stock data.

The CSV files have headers like:
- ticker, date, date_ts, open, high, low, close, volume, year, quarter

Rewrite the user's question into a short keyword query that:
- explicitly mentions tickers, years, and quarters
- uses words like: ticker, date_ts, close price, performance, quarter
- is a SINGLE line of plain text, no JSON, no punctuation except spaces and colons if helpful.

Example output for:
"Find the performance of Apple stock in the second quarter 2023 and compare it with Microsoft in second quarter 2023"

Possible query:
"ticker:AAPL ticker:MSFT year:2023 quarter:Q2 close price performance comparison"
"""

    messages = [
        {"role": "user", "content": [{"text": sys_prompt}]},
        {"role": "user", "content": [{"text": original_question.strip()}]},
    ]

    return send_request(rewriter_model, messages)



def rewrite_prompt_pdf(original_question: str) -> str:
    """
    Rewrite user query for PDF-based retrieval (10-Ks, annual reports, MD&A, etc.).
    """
    sys_prompt = """
You are generating a retrieval query for PDF financial reports
such as 10-Ks, 10-Qs, and annual reports.

Rewrite the user's question into a short, dense query that:
- Mentions the company names, segments (e.g., "Google Search"), and years.
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

    return send_request(rewriter_model, messages)

# ---------- RAG pipeline ---------
# ----- -- ----          ---- -----
# -------------          ----------
# ---------------------------------

def date_sort_key(point):
        # Use strptime() to parse the string based on its format
        # %d for day, %m for month, %Y for four-digit year

        date_dict= point.payload.get("date_range", {'start': "2000-01-01"})
        date_str= date_dict.get("start", "2000-01-01")

        return datetime.datetime.strptime(date_str, "%Y-%m-%d")


def rag(question: str, n_points: int = 10):
    # 1) Get two optimized queries: one for CSV data, one for PDF docs
    csv_query = rewrite_prompt_csv(question)
    pdf_query = rewrite_prompt_pdf(question)

    all_points = []

    # 2) Query Qdrant for CSV chunks
    try:
        csv_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Document(text=csv_query, model=embedding_model_name),
            limit=n_points,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchValue(value="csv"),
                    )
                ]
            ),
        )

        # Sorting by date
        sorted_csv_results = sorted(csv_results.points, key = date_sort_key)
        all_points.extend(sorted_csv_results)
    except Exception as e:
        print(f"[WARN] CSV query failed: {e}")

    # 3) Query Qdrant for PDF chunks
    try:
        pdf_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Document(text=pdf_query, model=embedding_model_name),
            limit=n_points,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchValue(value="pdf"),
                    )
                ]
            ),
        )
        all_points.extend(pdf_results.points)
    except Exception as e:
        print(f"[WARN] PDF query failed: {e}")

    if not all_points:
        print("[WARN] No points retrieved from Qdrant.")
        return

    # Optionally: re-sort by score and trim to top N total
    # all_points = sorted(all_points, key=lambda p: p.score, reverse=True)[:n_points]

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

    _ = send_request(answer_model, messages, print_prompt=False)


if __name__ == "__main__":
    original_prompt: str = (
        "Compare the year-end revenues for Google Search in the past two years "
        "and provide insight into what factors contribute to the figures."
    )
    rag(original_prompt, n_points=10)
