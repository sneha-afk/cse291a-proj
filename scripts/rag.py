"""
Generates response based on queries
"""
from qdrant_client import QdrantClient, models
from ollama import chat
from ollama import ChatResponse
from our_utils import get_qdrant_client, get_qdrant_config

collection_name, model_name = get_qdrant_config()
client: QdrantClient = get_qdrant_client()

def rag(question: str, n_points: int = 10):
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document(text=question, model=model_name),
        limit=n_points,
    )

    context = "\n".join(f"Relevant Document {i}, {r.payload["document"]}: {r.payload["content"]}" for i, r in enumerate(results.points))
    docs = "\n".join(f"Relevant Document {i}, {r.payload["document"]}, chunk index {r.payload["part_index"]}" for i, r in enumerate(results.points))
    print(docs)

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

    print(f"User: {question.strip()}")

    try:
        # Receive the chunks from the streaming reponse, print as they arrive
        for chunk in response:
            print(chunk.message.content, end='', flush=True)
    except KeyboardInterrupt:
        print("(QUIT)")
        return None


# Example use
rag("How does Tesla evaluate its energy segment growth and what strategies are they working on to increase its profitability?")
