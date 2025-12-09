from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from utils import setup_qdrant

client, collection_name, embedding_model_name = setup_qdrant()
# ========================================================================

# See documentation for filters:
# https://qdrant.tech/documentation/concepts/filtering/
# "source_type" is either "pdf" or "csv"
# "company" is usually the uppercase stock symbol

# Example: deleting csvs from ORCL
delete_filter = rest.Filter(
    must=[
        rest.FieldCondition(
            key="company",
            match=rest.MatchValue(value="ORCL")
        ),
        rest.FieldCondition(
            key="source_type",
            match=rest.MatchValue(value="csv")
        ),
    ]
)

result = client.delete(
    collection_name=collection_name,
    points_selector=rest.FilterSelector(filter=delete_filter)
)

print("Deletion completed:", result)
