# cse291a

To install libraries using uv, there is a `pyproject.toml` at the root:

```bash
uv sync
```

To start working, always do:

```bash
uv sync
./.venv/bin/activate    # bin vs. Scripts depends on OS
```


Quickstart:

1. Start Qdrant: ensure Docker Desktop is running and do `docker pull qdrant/qdrant` (one-time).
```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```
2. Start [Ollama](https://ollama.com/) with `gpt-oss:20b`: starting the GUI client or the following command
```bash
ollama run gpt-oss:20b
```

3. Generate embeddings with `embed.py` to generate embeddings: run from root of this repo
4. Run rag with`rag.py` with `rag("<Question>")`:

On terminal, to run a file within `scripts`, do:
```bash
python -m scripts.rag
```
so the `our_utils` module gets recognized on run.

To test retrieval (i.e see which documents are fetched by Qdrant), only run `python -m scripts.retrieval`
