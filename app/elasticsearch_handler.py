"""Lightweight Elasticsearch wrapper for index and search operations."""

from elasticsearch import Elasticsearch


class ElasticsearchHandler:
    """Basic Elasticsearch client for document indexing and vector search."""

    def __init__(self, host="localhost", port=9200):
        self.es = Elasticsearch([f"http://{host}:{port}"])

    def create_index(self, index_name):
        """Create an Elasticsearch index if it doesn't exist."""
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name)

    def index_document(self, index_name, doc_id, text, embedding):
        """Index a document with text and embedding vector."""
        body = {
            "text": text,
            "embedding": embedding.tolist(),
        }
        self.es.index(index=index_name, id=doc_id, body=body)

    def search(self, index_name, query_vector, top_k=5):
        """Search using cosine similarity on embedding vectors."""
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector.tolist()},
                },
            }
        }
        response = self.es.search(
            index=index_name,
            body={
                "size": top_k,
                "query": script_query,
                "_source": {"includes": ["text"]},
            },
        )
        return [hit["_source"]["text"] for hit in response["hits"]["hits"]]
