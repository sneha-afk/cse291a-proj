# cse291a

## Development

To install libraries using [`uv`](https://github.com/astral-sh/uv), there is a `pyproject.toml` at the root:

```bash
uv sync
```

To start working, always do:

```bash
uv sync
source ./.venv/bin/activate     # or ./.venv/Scripts/activate
```

> Recommended to set encoding to UTF-8 if encountering errors related to Unicode
> that may not render correctly when dumping into UTF-8 text files:
> ```
> export PYTHONIOENCODING="utf-8"
> # or
> $env:PYTHONIOENCODING="utf-8"
> ```

## Quickstarts
### Setup
1. Start Qdrant: ensure [Docker Desktop](https://docs.docker.com/get-started/introduction/get-docker-desktop/) is running

To create the storage folder in the local filesystem (i.e, PWD):
```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

To instead use a Docker volume to run from any directory or prevent potential file corruption:
```bash
docker volume create qdrant_storage
# Removing the $(pwd) from above to instead use the Docker volume
docker run -d -p 6333:6333 -p 6334:6334 -v "qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

2. Generate embeddings for PDF documents by running `embed.py` from the root of this repo
3. Generate embeddigns for CSV files by running `embed_csv.py` from the root of this repo
4. Set up inference source: locally with [Ollama](https://ollama.com/) or with [AWS Bedrock](https://aws.amazon.com/bedrock/) with the [instructions below](#running-aws-bedrock)
5. Run [`rag_workflow_combine_aws.py`](scripts/rag_workflow_combine_aws.py) from the root of this repo.

> `rag_local.py` and `rag_aws.py` are legacy scripts without the most up-to-date chunking methods.

To test Qdrant retrievals (i.e which documents or latency), run `retrieval.py`.

### Running AWS Bedrock
See [`boto3` documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) (Python client for Bedrock).

Ensure AWS CLI is installed: see [aws/README.md](./aws/README.md) for manual installation on Linux, else with package managers:
```bash
# Windows via winget
winget install Amazon.AWSCLI
scoop install aws             # or via Scoop
```

```bash
# macOS via brew
brew install awscli
```

```bash
# Linux, globally via pip if preferred
# Can add --user for just your user
sudo python -m pip install awscli
```

Generate API keys from the login page. You can set these as environment variables, or:
```bash
aws configure
```

**Make sure you set your region to `us-west-2` ONLY.**

To check what models are available on this region: (The models that we can use are the ones' listed ON-DEMAND)
```bash
aws bedrock list-foundation-models \
  --region us-west-2 \
  --query "modelSummaries[].{id:modelId, name:modelName, provider:providerName, types:inferenceTypesSupported}"
```

Running the above command *should also* validate your credentials, i.e. if your credential or region setup is wrong, the above
would fail as well.

Run `scripts/bedrock_quickstart.py` to run **a prompt request to a model** (obviously, don't spam this).
