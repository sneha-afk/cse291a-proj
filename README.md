# cse291a

To install libraries using uv, there is a `pyproject.toml` at the root:

```bash
uv sync
```

To start working, always do:

```bash
uv sync
./.venv/Scripts/activate
```

Quickstart:

1. Start Qdrant`docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant`
2. Start ollama with gpt oss
3. Generate embeddings with `embed.py` to generate embeddings
4. Run rag with`rag.py` with `rag("<Question>")`

To test retieval only use `retrieval.py`
