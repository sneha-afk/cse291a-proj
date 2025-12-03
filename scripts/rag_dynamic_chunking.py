"""
Enhanced RAG pipeline with hybrid dynamic retrieval
"""
import re
import json
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

collection_name = "knowledge_base"
embedding_model_name = "BAAI/bge-small-en-v1.5"

if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
    qdrant_client = QdrantClient(url=QDRANT_URL)
else:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

REGION = "us-west-2"
rewriter_model: str = "openai.gpt-oss-20b-1:0"
answer_model: str = "openai.gpt-oss-120b-1:0"
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

def extract_metadata_from_query(question: str, llm_model: str = rewriter_model) -> Dict[str, Any]:
    """
    Extract structured metadata from the query to guide retrieval
    """
    metadata_prompt = f"""Extract financial metadata from this query:

    Query: {question}

    Return JSON with:
    {{
        "companies": ["company1", "company2", ...],  # normalized ticker/names
        "time_periods": ["2023", "Q4 2023", "2022-2023", ...],
        "metrics": ["revenue", "growth", "profitability", ...],
        "data_types_needed": ["csv", "pdf", "both"],
        "analysis_type": "fact_lookup|comparison|trend|forecast|strategy"
    }}

    Normalize companies to common names/tickers.
    If no specific time mentioned, include current/latest year.
    For data_types_needed:
      - "csv" if query mentions numbers, prices, performance, metrics
      - "pdf" if query mentions strategies, evaluations, discussions, analysis
      - "both" for mixed queries
    """
    
    messages = [
        {"role": "user", "content": [{"text": metadata_prompt}]},
        {"role": "user", "content": [{"text": question.strip()}]},
    ]
    
    response = bedrock_client.converse(
        modelId=llm_model,
        messages=messages,
    )
    
    text = response["output"]["message"]["content"][-1]["text"]
    
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            metadata = json.loads(json_match.group())
            # Ensure lists and defaults
            metadata.setdefault("companies", [])
            metadata.setdefault("time_periods", [])
            metadata.setdefault("metrics", [])
            metadata.setdefault("data_types_needed", ["both"])
            metadata.setdefault("analysis_type", "fact_lookup")
            return metadata
    except Exception as e:
        print(f"[WARN] Metadata extraction failed: {e}")
    
    # Fallback
    return {
        "companies": [],
        "time_periods": [],
        "metrics": [],
        "data_types_needed": ["both"],
        "analysis_type": "fact_lookup"
    }

def build_qdrant_filter(metadata: Dict[str, Any], source_type: str = None) -> Optional[Filter]:
    """
    Build Qdrant filter based on extracted metadata
    """
    must_conditions = []
    
    # Filter by source type if specified
    if source_type:
        must_conditions.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchValue(value=source_type)
            )
        )
    
    # Filter by companies if specified
    if metadata["companies"]:
        company_conditions = []
        for company in metadata["companies"]:
            # Try different field names
            for field in ["company", "ticker", "document"]:
                company_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchText(text=company)
                    )
                )
        must_conditions.append(models.Filter(should=company_conditions))
    
    # Filter by years if specified
    if metadata["time_periods"]:
        year_conditions = []
        for period in metadata["time_periods"]:
            # Extract year numbers
            years = re.findall(r'20\d{2}', period)
            for year in years:
                year_conditions.append(
                    models.FieldCondition(
                        key="year",
                        match=models.MatchValue(value=int(year))
                    )
                )
        if year_conditions:
            must_conditions.append(models.Filter(should=year_conditions))
    
    # Filter by quarters if mentioned
    quarter_pattern = r'Q[1-4]'
    for period in metadata["time_periods"]:
        if re.search(quarter_pattern, period, re.IGNORECASE):
            must_conditions.append(
                models.FieldCondition(
                    key="quarter",
                    match=models.MatchText(text=re.search(quarter_pattern, period, re.IGNORECASE).group().upper())
                )
            )
            break
    
    if not must_conditions:
        return None
    
    return models.Filter(must=must_conditions)

def determine_chunk_counts(metadata: Dict[str, Any]) -> Dict[str, int]:
    """
    Determine how many chunks to retrieve based on metadata analysis
    """
    analysis_type = metadata["analysis_type"]
    data_types = metadata["data_types_needed"]
    num_companies = len(metadata["companies"])
    num_time_periods = len(metadata["time_periods"])
    
    # Base counts
    base_counts = {
        "csv": 0,
        "pdf": 0,
        "total": 0
    }
    
    # Adjust based on analysis type
    if analysis_type == "fact_lookup":
        base_per_source = 3
    elif analysis_type == "comparison":
        base_per_source = 5
    elif analysis_type == "trend":
        base_per_source = 7
    elif analysis_type == "strategy" or analysis_type == "forecast":
        base_per_source = 6
    else:
        base_per_source = 4
    
    # Multiply by number of companies (minimum 1)
    company_multiplier = max(1, num_companies)
    
    # Multiply by time periods (capped)
    time_multiplier = min(3, max(1, num_time_periods))
    
    # Calculate per source
    if "csv" in data_types or "both" in data_types:
        base_counts["csv"] = base_per_source * company_multiplier * time_multiplier
        # Ensure we get enough for comparisons
        if analysis_type == "comparison" and num_companies > 1:
            base_counts["csv"] += 3
    
    if "pdf" in data_types or "both" in data_types:
        base_counts["pdf"] = base_per_source * company_multiplier
        # PDFs often need more context for strategy/analysis
        if analysis_type in ["strategy", "forecast"]:
            base_counts["pdf"] += 4
    
    # Apply caps
    base_counts["csv"] = min(max(base_counts["csv"], 2), 15)
    base_counts["pdf"] = min(max(base_counts["pdf"], 2), 15)
    base_counts["total"] = base_counts["csv"] + base_counts["pdf"]
    
    return base_counts

def retrieve_metadata_first_pass(metadata: Dict[str, Any], query_text: str, 
                                source_type: str, limit: int) -> List[Any]:
    """
    First pass: Retrieve chunks matching metadata filters
    """
    qdrant_filter = build_qdrant_filter(metadata, source_type)
    
    try:
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=models.Document(text=query_text, model=embedding_model_name),
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False
        )
        return results.points
    except Exception as e:
        print(f"[WARN] Metadata-first retrieval failed for {source_type}: {e}")
        return []

def retrieve_similarity_second_pass(query_text: str, source_type: str, 
                                   limit: int, exclude_ids: List[str] = None) -> List[Any]:
    """
    Second pass: Pure similarity search, optionally excluding already retrieved chunks
    """
    try:
        query_params = {
            "collection_name": collection_name,
            "query": models.Document(text=query_text, model=embedding_model_name),
            "limit": limit,
            "with_payload": True,
            "with_vectors": False
        }
        print(1)
        
        # Add filter for source type
        query_params["query_filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value=source_type)
                )
            ]
        )
        print(2)
        
        # Exclude already retrieved chunks if needed
        # if exclude_ids:
        #     query_params["query_filter"].must.append(
        #         models.FieldCondition(
        #             key="id",
        #             match=models.MatchExcept(except_ids=exclude_ids)
        #         )
        #     )
        print(3)
        results = qdrant_client.query_points(**query_params)
        print(4)
        return results.points
    except Exception as e:
        print(f"[WARN] Similarity retrieval failed for {source_type}: {e}")
        return []

def check_coverage_gaps(retrieved_chunks: List[Any], metadata: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check if retrieved chunks have coverage gaps
    """
    coverage = {
        "has_all_companies": True,
        "has_time_coverage": True,
        "has_source_diversity": True,
        "has_recent_data": True
    }
    
    if not retrieved_chunks:
        return {k: False for k in coverage.keys()}
    
    # Check company coverage
    found_companies = set()
    for chunk in retrieved_chunks:
        payload = chunk.payload
        company = payload.get("company") or payload.get("ticker")
        if company:
            found_companies.add(str(company).lower())
    
    requested_companies = {c.lower() for c in metadata["companies"]}
    if requested_companies:
        coverage["has_all_companies"] = requested_companies.issubset(found_companies)
    
    # Check time coverage
    years_found = set()
    for chunk in retrieved_chunks:
        year = chunk.payload.get("year")
        if year:
            years_found.add(str(year))
    
    requested_years = set()
    for period in metadata["time_periods"]:
        years = re.findall(r'20\d{2}', period)
        requested_years.update(years)
    
    if requested_years:
        coverage["has_time_coverage"] = requested_years.issubset(years_found)
    
    # Check source diversity
    source_types = set(chunk.payload.get("source_type", "") for chunk in retrieved_chunks)
    requested_sources = set(metadata["data_types_needed"])
    if "both" in requested_sources:
        requested_sources = {"csv", "pdf"}
    
    coverage["has_source_diversity"] = requested_sources.issubset(source_types)
    
    # Check for recent data (within 2 years)
    current_year = 2024  # Update as needed
    recent_years = {str(current_year), str(current_year - 1)}
    coverage["has_recent_data"] = bool(recent_years.intersection(years_found))
    
    return coverage

def hybrid_retrieval(question: str, use_hybrid: bool = True) -> List[Any]:
    """
    Main hybrid retrieval function
    """
    # Step 1: Extract metadata from query
    metadata = extract_metadata_from_query(question)
    print(f"\n📊 Extracted Metadata: {json.dumps(metadata, indent=2)}")
    
    # Step 2: Determine chunk counts based on metadata
    chunk_counts = determine_chunk_counts(metadata)
    print(f"\n🎯 Target Retrieval: CSV={chunk_counts['csv']}, PDF={chunk_counts['pdf']}")
    
    # Step 3: Rewrite queries for each source type
    csv_query = rewrite_prompt_csv(question) if chunk_counts["csv"] > 0 else ""
    pdf_query = rewrite_prompt_pdf(question) if chunk_counts["pdf"] > 0 else ""
    
    all_chunks = []
    retrieved_ids = set()
    
    # Process CSV data if needed
    if chunk_counts["csv"] > 0 and csv_query:
        print(f"\n📈 Retrieving CSV chunks...")
        
        # First pass: Metadata-filtered
        metadata_chunks = retrieve_metadata_first_pass(
            metadata, csv_query, "csv", 
            limit=max(3, chunk_counts["csv"] // 2)
        )
        
        # Track retrieved IDs
        for chunk in metadata_chunks:
            retrieved_ids.add(chunk.id)
        
        # Second pass: Similarity search to fill quota
        remaining = chunk_counts["csv"] - len(metadata_chunks)
        if remaining > 0:
            similarity_chunks = retrieve_similarity_second_pass(
                csv_query, "csv", remaining, 
                exclude_ids=list(retrieved_ids)
            )
            for chunk in similarity_chunks:
                retrieved_ids.add(chunk.id)
            metadata_chunks.extend(similarity_chunks)
        
        print(f"   Retrieved {len(metadata_chunks)} CSV chunks")
        all_chunks.extend(metadata_chunks)
    
    # Process PDF data if needed
    if chunk_counts["pdf"] > 0 and pdf_query:
        print(f"\n📄 Retrieving PDF chunks...")
        
        # First pass: Metadata-filtered
        metadata_chunks = retrieve_metadata_first_pass(
            metadata, pdf_query, "pdf", 
            limit=max(3, chunk_counts["pdf"] // 2)
        )
        
        # Track retrieved IDs
        for chunk in metadata_chunks:
            retrieved_ids.add(chunk.id)
        
        # Second pass: Similarity search to fill quota
        remaining = chunk_counts["pdf"] - len(metadata_chunks)
        if remaining > 0:
            similarity_chunks = retrieve_similarity_second_pass(
                pdf_query, "pdf", remaining, 
                exclude_ids=list(retrieved_ids)
            )
            for chunk in similarity_chunks:
                retrieved_ids.add(chunk.id)
            metadata_chunks.extend(similarity_chunks)
        
        print(f"   Retrieved {len(metadata_chunks)} PDF chunks")
        all_chunks.extend(metadata_chunks)
    
    # Step 4: Check coverage and fill gaps if needed
    coverage = check_coverage_gaps(all_chunks, metadata)
    print(f"\n📋 Coverage Analysis: {coverage}")
    
    # Fill critical gaps
    if not coverage["has_all_companies"] and metadata["companies"]:
        print("   ⚠️  Missing company data, attempting to fill...")
        missing_companies = set(c.lower() for c in metadata["companies"])
        found_companies = set()
        for chunk in all_chunks:
            company = chunk.payload.get("company") or chunk.payload.get("ticker")
            if company:
                found_companies.add(str(company).lower())
        
        missing = missing_companies - found_companies
        for company in missing:
            # Quick search for missing company
            try:
                fill_results = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=models.Document(text=company, model=embedding_model_name),
                    limit=2,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="company",
                                match=models.MatchText(text=company)
                            )
                        ]
                    )
                )
                if fill_results.points:
                    all_chunks.extend(fill_results.points)
                    print(f"     Added {len(fill_results.points)} chunks for {company}")
            except Exception as e:
                print(f"     Could not fill data for {company}: {e}")
    
    # Optional: Re-rank chunks by relevance
    if len(all_chunks) > 1:
        all_chunks.sort(key=lambda x: x.score, reverse=True)
    
    print(f"\n✅ Total chunks retrieved: {len(all_chunks)}")
    return all_chunks

def enhanced_rag(question: str, use_hybrid: bool = True):
    """
    Enhanced RAG pipeline with hybrid retrieval
    """
    # Use hybrid retrieval
    all_chunks = hybrid_retrieval(question, use_hybrid)
    
    if not all_chunks:
        print("[ERROR] No chunks retrieved")
        return
    
    # Build context from chunks (your existing format)
    context_lines = []
    docs_lines = []
    
    for i, chunk in enumerate(all_chunks):
        payload = chunk.payload
        doc_name = payload.get("document", "unknown_document")
        content = payload.get("content", "")
        part_index = payload.get("part_index", "NA")
        source_type = payload.get("source_type", "unknown")
        company = payload.get("company", "unknown")
        year = payload.get("year", "unknown")
        
        context_lines.append(
            f"[{source_type.upper()}] {company} {year} | {doc_name}: {content[:300]}..."
        )
        docs_lines.append(
            f"Chunk {i} | Score: {chunk.score:.3f} | {doc_name} | {company} {year} | source={source_type}"
        )
    
    context = "\n".join(context_lines)
    
    print("\n📚 Retrieved Chunks Summary:")
    for line in docs_lines[:10]:  # Show top 10
        print(f"   {line}")
    
    if len(docs_lines) > 10:
        print(f"   ... and {len(docs_lines) - 10} more")
    
    # Generate answer (your existing code)
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
    
    # Use your existing send_request function
    response = bedrock_client.converse(
        modelId=answer_model,
        messages=messages,
    )
    
    answer = response["output"]["message"]["content"][-1]["text"]
    print(f"\n🤖 Answer:\n{answer}\n")
    
    return answer

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

if __name__ == "__main__":
    # Test queries
    # test_queries = [
    #     "What was Tesla's revenue in Q4 2023?",
    #     "Compare Tesla and Apple's stock performance in 2023",
    #     "Analyze Google's revenue growth strategy over the past 3 years",
    #     "How does NVIDIA plan to maintain its AI chip market leadership?"
    # ]

    test_queries = [
        "Compare Tesla and Apple's stock performance in 2023",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        enhanced_rag(query, use_hybrid=True)