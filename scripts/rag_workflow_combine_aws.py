import boto3
from botocore.exceptions import ClientError
from qdrant_client import QdrantClient, models
import datetime
from utils import setup_qdrant
from typing import Any

qdrant_client, collection_name, embedding_model_name = setup_qdrant()

REGION = "us-west-2"  # don't change this

rewriter_model: str = "openai.gpt-oss-20b-1:0"
answer_model: str = "openai.gpt-oss-120b-1:0"

bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

# don't exhaustively retry
MAX_RETRIES = 10

def send_request(model: str, messages: list[dict], print_prompt: bool = True) -> str:
    retry_count = 0
    while True:
        try:
            response = bedrock_client.converse(
                modelId=model,
                messages=messages,
                inferenceConfig={
                    "maxTokens": 16384,
                }
            )
            break
        except ClientError as e:
            print(f"[ERROR] Bedrock request failed: {e}")
            retry_count += 1
            print(f"\tRetrying Bedrock request (retry count: {retry_count})")
            if retry_count >= MAX_RETRIES:
                print("[ERROR] Max retries reached, exiting.")
                exit(1)
            continue

    print(f"Using model: {model}\n\n")
    if print_prompt:
        print("Input messages:")
        for msg_dict in messages:
            # careful with quotes
            print(f"{msg_dict['role']}: {msg_dict['content'][0]['text']}")
    print("\n----\nResponse:")
    print("Usage:", response["usage"])
    print("Metrics:", response["metrics"])
    text = (
        response["output"]["message"]["content"][-1]["reasoningContent"][
            "reasoningText"
        ]["text"]
        if "reasoningContent" in response["output"]["message"]["content"][-1].keys()
        else response["output"]["message"]["content"][-1]["text"]
    )
    return text


# ---------- CSV & PDF rewriters ----------


def rewrite_prompt_csv(original_question: str) -> dict[str, Any]:
    sys_prompt = """
You are an assistant that rewrites a natural language financial query into a Qdrant search query as a Python dictionary
for better retrieval over CSV stocks/company share data.

The CSV files have headers like:
- ticker, date, date_ts, open, high, low, close, volume, year, quarter

Given a user query of:
"Find the performance of Apple stock in the second quarter 2023 and compare it with Microsoft in second quarter 2023"

Possible query:
"company:AAPL company:MSFT year:2023 quarter:Q2 close price performance comparison"

If you are unsure of the ticker symbol of a company, guess multiple possible symbols within an "any" match.
When specifying one value for matches, use "match": {"value": ...}.
When specifying multiple values for matches, use "match": {"any": [...]}.

Here's an example of what a Qdrant query would look like that follows python dictionary formatting:
You're only given two metadata keys: company and date_range
chunk_count is the number of business days within the date range for a given company
{
    "query": "company:AAPL company:MSFT year:2023 quarter:Q2 close price performance comparison",
    "filters: [
        {
            "must": [
                    {"key": "company", "match": {"value": "AAPL"}},
                    {"key": "date_range.start", "range": { "lte": "2023-06-30" }},
                    {"key": "date_range.end", "range": { "gte": "2023-04-01" }}
                ],
            "chunk_count": 81,
        },
        {
            "must": [
                {"key": "company", "match": {"value": "MSFT"}},
                {"key": "date_range.start", "range": { "lte": "2023-06-30" }},
                {"key": "date_range.end", "range": { "gte": "2023-04-01" }}
            ],
            "chunk_count": 81
        }
    ]
}
Reasoning: high


Your task:
- Read the user's question.
- Infer the correct tickers, date ranges, and any other needed filters.
- You will first rewrite the user query then add it to a given valid Python dict literal.
- Return ONLY a valid JSON object in this exact qdrant query format.
- Do NOT include backticks, explanations, or any surrounding text. Just a valid python dictionary.
- If the prompt does not require financial data retrieval, (e.g. questions about company strategies,
  products, or other non-fiscal topics), return an empty query with zero chunk count:
{
    "query": "<optimized query text>",
    "filters": [],
    "chunk_count": 0
}
Remember, questions with numerical or date references often imply a need for financial data retrieval. Be sure to include relevant filters for such queries and a non-zero chunk count.
"""

    messages = [
        {"role": "user", "content": [{"text": sys_prompt}]},
        {"role": "user", "content": [{"text": original_question.strip()}]},
    ]

    # run in loop, eval err if something goes wrong
    retry_count = 0
    while True:
        try:
            qdrant_query = eval(send_request(rewriter_model, messages))
            break
        except Exception as e:
            retry_count += 1
            print(f"[ERROR][CSVAgent] Failed to parse Qdrant query: {e}")
            print(f"\tRetrying CSVAgent (retry count: {retry_count})")
            if retry_count >= MAX_RETRIES:
                print("[ERROR] Max retries reached, exiting.")
                exit(1)
            continue

    return qdrant_query


def rewrite_prompt_pdf(original_question: str) -> dict[str, Any]:
    """
    Rewrite user query for PDF-based retrieval (10-Ks, annual reports, MD&A, etc.).
    """
    sys_prompt = """
You are an assistant that rewrites a natural language financial query into a Qdrant search query as a Python dictionary
for better retrieval over PDF financial reports such as 10-Ks, 10-Qs, and annual reports.

Rewrite the user's question into a short, dense query that:
- Mentions the company names, segments (e.g., "Google Search"), and years.
- Targets sections like "Results of Operations", "Segment Results", "Revenue", and
  commentary on growth, drivers, and profitability.
- Uses phrases that are likely to appear in those reports, like
  "year-over-year", "segment revenue", "results of operations", "management discussion".
- Targets keywords relevant to the question and financial analysis.

You may also use filters on the "company" (ticker symbol) and "year" keys if specific companies or years are mentioned.
When specifying one value for matches, use "match": {"value": ...}.
When specifying multiple values for matches, use "match": {"any": [...]}.

The company key is based of the stock ticker symbol, e.g., "AAPL" for Apple, "MSFT" for Microsoft, "GOOGL" for Google, etc.
If you are unsure of the ticker symbol, guess multiple possible symbols.

For example, given the user question, "What were key revenue figures for Amazon and Oracle in 2021.",
a optimized query would be:
{
    "query": <optimized query text>,
    "filters": [
        {
            "must": [
                {"key": "year", "match": {"value": "2021"}},
                {"key": "company", "match": {"any": ["AMZN", "ORCL"]}}
            ]
        }
    ]
}

Be concise. No explanations, just the optimized query.
"""

    messages = [
        {"role": "user", "content": [{"text": sys_prompt}]},
        {"role": "user", "content": [{"text": original_question.strip()}]},
    ]

    retry_count = 0
    while True:
        try:
            qdrant_query = eval(send_request(rewriter_model, messages))
            break
        except Exception as e:
            retry_count += 1
            print(f"[ERROR][PDFAgent] Failed to parse Qdrant query: {e}")
            print(f"\tRetrying PDFAgent (retry count: {retry_count})")
            if retry_count >= MAX_RETRIES:
                print("[ERROR] Max retries reached, exiting.")
                exit(1)
            continue

    return qdrant_query


def qdrant_filter_from_dict(d: dict, datatype_cond: str = "") -> models.Filter:
    groups = d["filters"]  # list of {"must": [ ... ]}

    all_filters: list[models.Filter | models.Condition] = []

    for group in groups:
        must_conditions: list[models.Condition] = []
        # should_conditions: list[models.Condition] = [] # not used, lets stick with must conditions

        for cond in group.get("must", []):
            key = cond["key"]

            if datatype_cond: # restrict to each group, NOT overall otherwise conditions are too broad
                must_conditions.append(
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_type",
                                match=models.MatchValue(value=datatype_cond),
                            )
                        ]
                    )
                )

            # Match condition
            if "match" in cond:
                # any is used to match multiple values, value for one
                if "any" in cond["match"]:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=cond["match"]["any"]),
                        )
                    )
                elif "value" in cond["match"]:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=cond["match"]["value"]),
                        )
                    )

                if datatype_cond == "csv":
                    must_conditions.append(
                        models.FieldCondition(
                            key="chunk_type",
                            match=models.MatchValue(value="daily")
                        )
                    )

            # Range condition
            if "range" in cond and datatype_cond == "csv":
                r = cond["range"]
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        range=models.DatetimeRange(
                            gte=r.get("gte"),
                            lte=r.get("lte"),
                            lt=r.get("lt"),
                            gt=r.get("gt"),
                        ),
                    )
                )

        all_filters.append(models.Filter(must=must_conditions))

    return models.Filter(should=all_filters)


# ---------- RAG pipeline ---------
# ----- -- ----          ---- -----
# -------------          ----------
# ---------------------------------


def date_sort_key(point) -> datetime.datetime:
    # Use strptime() to parse the string based on its format
    # %d for day, %m for month, %Y for four-digit year

    date_dict = point.payload.get("date_range", {"start": "2000-01-01"})
    date_str = date_dict.get("start", "2000-01-01")

    return datetime.datetime.strptime(date_str, "%Y-%m-%d")


def rag(question: str, n_points: int = 10) -> str:
    # 1) Get two optimized queries: one for CSV data, one for PDF docs
    csv_query = rewrite_prompt_csv(question)
    pdf_query = rewrite_prompt_pdf(question)

    # print("Must filters")
    # print(
    #     [
    #         {"value": f["must"]["match"]}
    #         for f in csv_query["filters"]
    #         if "must" in f.keys() f["must"]
    #     ]
    # )
    #
    # print("Range filters")
    # print(
    #     [
    #         {"gte": f["must"]["range"]["gte"], "lt": f["must"]["range"]["lt"]}
    #         for f in csv_query["filters"]
    #         if "must" in f.keys() and "range" in f["range"].keys()
    #     ]
    # )

    all_points = []

    # 2) Query Qdrant for CSV chunks
    try:
        filtered_csv_query = qdrant_filter_from_dict(csv_query, "csv")
        csv_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Document(text=csv_query["query"], model=embedding_model_name),
            limit=sum([f["chunk_count"] for f in csv_query["filters"]]),
            query_filter=filtered_csv_query,
        )

        # Sorting by date
        sorted_csv_results = sorted(csv_results.points, key=date_sort_key)
        all_points.extend(sorted_csv_results)
        print("CSV CHUNK COUNTS: ", len(all_points))
    except Exception as e:
        print(f"[WARN] CSV query failed: {e}")

    # 3) Query Qdrant for PDF chunks
    try:
        filtered_pdf_query = qdrant_filter_from_dict(pdf_query, "pdf")
        pdf_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Document(text=pdf_query["query"], model=embedding_model_name),
            limit=n_points,
            query_filter=filtered_pdf_query,
        )
        all_points.extend(pdf_results.points)
    except Exception as e:
        print(f"[WARN] PDF query failed: {e}")

    if not all_points:
        print("[WARN] No points retrieved from Qdrant.")
        exit(1)

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
        date_range = payload.get("date_range", "N/A")
        year = payload.get("year", "N/A")

        context_lines.append(
            f"[{source_type.upper()}] Relevant Document {i}, {doc_name}: {content}"
        )
        docs_lines.append(
            f"Relevant Document {i}, {doc_name}, chunk index {part_index}, source_type={source_type}, date range={date_range}"
            if date_range != "N/A"
            else f"Relevant Document {i}, {doc_name}, chunk index {part_index}, source_type={source_type}, year={year}"
        )

    context = "\n".join(context_lines)
    docs = "\n".join(docs_lines)

    print("Retrieved the following docs:")
    print(docs)

    # 4) Ask the answer model with combined context
    metaprompt = f"""
Answer the following question using ONLY the provided context from both CSV data and PDF reports.
If the context seems insufficint, consider running simple calculations, inferring trends, and cross-document analysis.
If you can't find the answer, do not pretend you know it; clearly explain why you are unable to answer.

Context:
{context.strip()}

Question:
{question}
"""

    messages = [
        {"role": "user", "content": [{"text": metaprompt}]},
    ]

    return send_request(answer_model, messages, print_prompt=False)


if __name__ == "__main__":
    original_prompt: str = (
        "Did Microsoft's stock performance reflect its reported cash from operations trends from 2024?"
    )
    print(rag(original_prompt))
