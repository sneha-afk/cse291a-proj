import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Initialize Qdrant client
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL, timeout=60)
else:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collection_name = "knowledge_base"

# Can dump in string directly from RAG script
def parse_document_references(dump_string):
    references = []

    for line in dump_string.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Remove the "Relevant Document N, " prefix
        if line.startswith("Relevant Document"):
            first_comma = line.find(',')
            if first_comma != -1:
                line = line[first_comma + 1:].strip()

        # Now parse "NASDAQ_GOOGL_2021.txt, chunk index 131"
        parts = line.split(', chunk index ')
        if len(parts) == 2:
            doc_name = parts[0].strip()
            chunk_idx = int(parts[1].strip())
            references.append((doc_name, chunk_idx))

    return references


def fetch_chunk(doc_name, chunk_idx):
    results = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "document", "match": {"value": doc_name}},
                {"key": "part_index", "match": {"value": chunk_idx}}
            ]
        },
        limit=1
    )

    if results:
        return results[0][0]
    return None


def calculate_metrics(relevance_judgments):
    n = len(relevance_judgments)
    total_relevant = sum(relevance_judgments)
    total_num_docs = None

    while not total_num_docs:
        try:
            total_num_docs = int(input("Total amount of possible chunks to fetch from (guess of how many were relevant, sum from embed.py): "))
        except:
            print("Should be a number")

    if total_relevant == 0:
        return {"error": "No relevant documents found"}

    metrics = []
    cumulative_ap = 0.0

    for k in range(1, n + 1):
        # Documents retrieved up to position k
        retrieved_at_k = relevance_judgments[:k]

        # True positives: was fetched, is relevant
        tp = sum(retrieved_at_k)
        # False positive: was fetched, not relevant
        fp = k - tp

        # Precision and Recall at k
        precision_at_k = tp / k if k > 0 else 0
        recall_at_k = tp / total_relevant if total_relevant > 0 else 0
        recall_grounded = tp / total_num_docs if total_num_docs > 0 else 0

        # F1 at k
        if precision_at_k + recall_at_k > 0:
            f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        else:
            f1_at_k = 0

        if precision_at_k + recall_grounded > 0:
            f1_at_k_grounded = 2 * precision_at_k * recall_grounded / (precision_at_k + recall_grounded)
        else:
            f1_at_k_grounded = 0


        # Add to AP calculation: depends on the order and how many have been seen thus far
        if relevance_judgments[k-1]:  # rel(k) = 1
            cumulative_ap += precision_at_k

        metrics.append({
            'k': k,
            'precision': precision_at_k,
            'recall': recall_at_k,
            'recall_grounded': recall_grounded,
            'f1': f1_at_k,
            'f1_grounded': f1_at_k_grounded,
        })

    # Average Precision
    ap = cumulative_ap / total_relevant if total_relevant > 0 else 0

    return {
        'metrics_at_k': metrics,
        'average_precision': ap,
        'total_relevant': total_relevant,
        'total_relevant_whole': total_num_docs,
        'total_retrieved': n
    }


def evaluate_retrieval(query, document_references):
    print(f"Query: {query}")

    relevance_judgments = []

    parsed_doc_refs = parse_document_references(document_references)

    for i, (doc_name, chunk_idx) in enumerate(parsed_doc_refs):
        # doc_name, chunk_idx = parse_document_references(doc_ref)

        chunk = fetch_chunk(doc_name, chunk_idx)
        if chunk is None:
            print(f"\n[{i}/{len(document_references)}] WARNING: Could not find chunk")
            print(f"  Document: {doc_name}, Chunk: {chunk_idx}")
            continue

        print(f"\n{'='*80}")
        print(f"[{i}/{len(document_references)}] Doc: {doc_name}, Chunk (part_index): {chunk_idx}")
        print(f"{'='*80}")
        print(chunk.payload.get('content', 'didnt fetch'))
        print(f"{'='*80}")

        # Ask for relevance judgment
        while True:
            response = input(f"\nRelevant to the query? (y/n): ").strip().lower()
            if response in ['y', 'n']:
                relevance_judgments.append(response == 'y')
                break
            print("only enter 'y' or 'n'")

    print(f"\n\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")

    results = calculate_metrics(relevance_judgments)

    if "error" in results:
        print(f"Error: {results['error']}")
        print("Probably zero for all")
        return

    print(f"Total Retrieved: {results['total_retrieved']}")
    print(f"Total Relevant: {results['total_relevant']}")
    print(f"Total Relevant (considering all chunks): {results['total_relevant_whole']}")
    print(f"\nAverage Precision (AP): {results['average_precision']:.4f}\n")

    print(f"{'k':<5} {'Precision':<16} {'Recall':<16} {'~Recall (all chunks)':<20} {'F1':<16} {'F1 (all chunks)':<16}")
    print("-" * 70)
    for m in results['metrics_at_k']:
        print(f"{m['k']:<5} {m['precision']:<16.4f} {m['recall']:<16.4f} {m['recall_grounded']:<20.4f} {m['f1']:<16.4f} {m['f1_grounded']:<16.4f}")

    return results


if __name__ == "__main__":
    # what was the question asked, doesn't have to be filled out
    query = ""

    document_refs = """
Relevant Document 0, NASDAQ_META_2024.txt, chunk index 1
Relevant Document 1, NASDAQ_MSFT_2021.txt, chunk index 13
Relevant Document 2, NASDAQ_MSFT_2022.txt, chunk index 7
Relevant Document 3, NASDAQ_META_2023.txt, chunk index 264
Relevant Document 4, NASDAQ_MSFT_2022.txt, chunk index 14
Relevant Document 5, NASDAQ_MSFT_2023.txt, chunk index 12
Relevant Document 6, NASDAQ_MSFT_2023.txt, chunk index 10
Relevant Document 7, NASDAQ_META_2023.txt, chunk index 6
Relevant Document 8, NASDAQ_META_2024.txt, chunk index 277
Relevant Document 9, NASDAQ_MSFT_2023.txt, chunk index 5


        """

    results = evaluate_retrieval(query, document_refs)
