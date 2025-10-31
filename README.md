# cse291a

To install libraries using [`uv`](https://github.com/astral-sh/uv), there is a `pyproject.toml` at the root:

```bash
uv sync
```

To start working, always do:

```bash
uv sync
source ./.venv/bin/activate     # or ./.venv/Scripts/activate
```

## Quickstarts
### Running locally

1. Start Qdrant: ensure Docker Desktop is running
```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```
2. Start [Ollama](https://ollama.com/) with `gpt-oss:20b`
3. Generate embeddings with `embed.py` to generate embeddings: run from root of this repo
4. Run rag with`rag.py` with `rag("<Question>")`

To test retrieval only use `retrieval.py`

### Running AWS Bedrock
See [`boto3` documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) (Python client for Bedrock).

Generate API keys from the login page. You can set these as environment variables, or:
```bash
aws configure
```

**Make sure you set your region to `us-west-2` ONLY.**

To check what models are available on this region:
```bash
aws bedrock list-foundation-models \
  --region us-west-2 \
  --query "modelSummaries[].{id:modelId, name:modelName, provider:providerName, types:inferenceTypesSupported}"
```

Running the above command *should also* validate your credentials, i.e. if your credential or region setup is wrong, the above
would fail as well.

Run `scripts/bedrock_quickstart.py` to run **a prompt request to a model** (obviously, don't spam this).
