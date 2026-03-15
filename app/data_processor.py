"""Data processing, embedding generation, and search functionality."""

import logging
import os
import re

import numpy as np
from elasticsearch import Elasticsearch
from minsearch import Index
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default field configuration
DEFAULT_TEXT_FIELDS = ["content", "title", "description"]
DEFAULT_KEYWORD_FIELDS = ["video_id", "author", "upload_date"]


def clean_text(text):
    """Remove special characters and normalize whitespace."""
    if not isinstance(text, str):
        logger.warning(f"Non-string input to clean_text: {type(text)}")
        return ""
    cleaned = re.sub(r"[^\w\s.,!?]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class DataProcessor:
    """Handles transcript processing, embedding generation, and search."""

    def __init__(
        self,
        text_fields=None,
        keyword_fields=None,
        embedding_model="multi-qa-MiniLM-L6-cos-v1",
    ):
        self.text_fields = text_fields or DEFAULT_TEXT_FIELDS
        self.keyword_fields = keyword_fields or DEFAULT_KEYWORD_FIELDS
        self.all_fields = self.text_fields + self.keyword_fields
        self.text_index = Index(
            text_fields=self.text_fields,
            keyword_fields=self.keyword_fields,
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = []
        self.index_built = False
        self.current_index_name = None

        es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
        es_port = int(os.getenv("ELASTICSEARCH_PORT", 9200))
        self.es = Elasticsearch([f"http://{es_host}:{es_port}"])
        logger.info(f"DataProcessor initialized (ES: {es_host}:{es_port}).")

    def process_transcript(self, video_id, transcript_data):
        """Process a video transcript into a searchable document."""
        if not transcript_data:
            logger.error(f"No transcript data for video {video_id}.")
            return None

        if "metadata" not in transcript_data or "transcript" not in transcript_data:
            logger.error(f"Invalid transcript structure for video {video_id}.")
            return None

        metadata = transcript_data["metadata"]
        transcript = transcript_data["transcript"]

        full_transcript = " ".join(
            segment.get("text", "") for segment in transcript
        )
        cleaned_transcript = clean_text(full_transcript)

        if not cleaned_transcript:
            logger.warning(f"Empty transcript after cleaning for video {video_id}.")
            return None

        doc = {
            "video_id": video_id,
            "content": cleaned_transcript,
            "title": clean_text(metadata.get("title", "")),
            "description": clean_text(metadata.get("description", "Not Available")),
            "author": metadata.get("author", ""),
            "upload_date": metadata.get("upload_date", ""),
            "segment_id": f"{video_id}_full",
            "view_count": metadata.get("view_count", 0),
            "like_count": metadata.get("like_count", 0),
            "comment_count": metadata.get("comment_count", 0),
            "video_duration": metadata.get("duration", ""),
        }

        self.documents.append(doc)
        embedding = self.embedding_model.encode(
            cleaned_transcript + " " + metadata.get("title", "")
        )
        self.embeddings.append(embedding)
        logger.info(f"Processed transcript for video {video_id}.")

        return {
            "content": cleaned_transcript,
            "metadata": metadata,
            "index_name": (
                f"video_{video_id}_"
                f"{self.embedding_model.get_sentence_embedding_dimension()}"
            ),
        }

    def build_index(self, index_name):
        """Build text and Elasticsearch indices from processed documents."""
        if not self.documents:
            logger.error("No documents to index.")
            return None

        logger.info(f"Building index '{index_name}' with {len(self.documents)} documents.")

        index_fields = self.text_fields + self.keyword_fields
        docs_to_index = []
        for doc in self.documents:
            indexed_doc = {field: doc.get(field, "") for field in index_fields}
            if all(indexed_doc.values()):
                docs_to_index.append(indexed_doc)
            else:
                missing = [f for f, v in indexed_doc.items() if not v]
                logger.warning(
                    f"Document {doc.get('video_id', 'unknown')} missing fields: {missing}"
                )

        if not docs_to_index:
            logger.error("No valid documents to index.")
            return None

        # Build text index
        try:
            self.text_index.fit(docs_to_index)
            self.index_built = True
            logger.info("Text index built successfully.")
        except Exception as e:
            logger.error(f"Error building text index: {e}")
            raise

        # Build Elasticsearch index
        try:
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(
                    index=index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": len(self.embeddings[0]),
                                    "index": True,
                                    "similarity": "cosine",
                                },
                                "content": {"type": "text"},
                                "title": {"type": "text"},
                                "description": {"type": "text"},
                                "video_id": {"type": "keyword"},
                                "author": {"type": "keyword"},
                                "upload_date": {"type": "date"},
                                "segment_id": {"type": "keyword"},
                                "view_count": {"type": "integer"},
                                "like_count": {"type": "integer"},
                                "comment_count": {"type": "integer"},
                                "video_duration": {"type": "text"},
                            }
                        }
                    },
                )
                logger.info(f"Created Elasticsearch index: {index_name}")

            for doc, embedding in zip(self.documents, self.embeddings):
                doc_with_embedding = doc.copy()
                doc_with_embedding["embedding"] = embedding.tolist()
                self.es.index(
                    index=index_name,
                    body=doc_with_embedding,
                    id=doc["segment_id"],
                )

            logger.info(f"Indexed {len(self.documents)} documents in Elasticsearch.")
            self.current_index_name = index_name
            return index_name
        except Exception as e:
            logger.error(f"Error building Elasticsearch index: {e}")
            raise

    def compute_rrf(self, rank, k=60):
        """Compute Reciprocal Rank Fusion score."""
        return 1 / (k + rank)

    def hybrid_search(self, query, index_name, num_results=5):
        """Perform hybrid search combining vector and keyword search with RRF."""
        if not index_name:
            raise ValueError("No index name provided for hybrid search.")

        vector = self.embedding_model.encode(query)

        knn_query = {
            "field": "embedding",
            "query_vector": vector.tolist(),
            "k": 10,
            "num_candidates": 100,
        }
        keyword_query = {
            "multi_match": {"query": query, "fields": self.text_fields}
        }

        try:
            knn_results = self.es.search(
                index=index_name,
                body={"knn": knn_query, "size": 10},
            )["hits"]["hits"]

            keyword_results = self.es.search(
                index=index_name,
                body={"query": keyword_query, "size": 10},
            )["hits"]["hits"]

            # Compute RRF scores
            rrf_scores = {}
            for rank, hit in enumerate(knn_results):
                rrf_scores[hit["_id"]] = self.compute_rrf(rank + 1)

            for rank, hit in enumerate(keyword_results):
                doc_id = hit["_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.compute_rrf(rank + 1)

            reranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            results = []
            for doc_id, _score in reranked[:num_results]:
                doc = self.es.get(index=index_name, id=doc_id)
                results.append(doc["_source"])

            return results
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise

    def search(
        self,
        query,
        filter_dict=None,
        boost_dict=None,
        num_results=10,
        method="hybrid",
        index_name=None,
    ):
        """Dispatch search to the appropriate method."""
        if not index_name:
            raise ValueError("No index name provided for search.")
        if not self.es.indices.exists(index=index_name):
            raise ValueError(f"Index '{index_name}' does not exist.")

        filter_dict = filter_dict or {}
        boost_dict = boost_dict or {}

        logger.info(f"Searching ({method}) in '{index_name}': {query[:80]}...")

        try:
            if method == "text":
                return self.text_search(query, filter_dict, boost_dict, num_results, index_name)
            elif method == "embedding":
                return self.embedding_search(query, num_results, index_name)
            else:
                return self.hybrid_search(query, index_name, num_results)
        except Exception as e:
            logger.error(f"Error in {method} search: {e}")
            raise

    def text_search(self, query, filter_dict=None, boost_dict=None, num_results=10, index_name=None):
        """Perform keyword-based text search."""
        if not index_name:
            raise ValueError("No index name provided for text search.")

        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {
                        "multi_match": {"query": query, "fields": self.text_fields}
                    },
                    "size": num_results,
                },
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            raise

    def embedding_search(self, query, num_results=10, index_name=None):
        """Perform vector similarity search."""
        if not index_name:
            raise ValueError("No index name provided for embedding search.")

        try:
            query_vector = self.embedding_model.encode(query).tolist()
            response = self.es.search(
                index=index_name,
                body={
                    "size": num_results,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_vector},
                            },
                        }
                    },
                    "_source": {"excludes": ["embedding"]},
                },
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error in embedding search: {e}")
            raise

    def set_embedding_model(self, model_name):
        """Switch the embedding model."""
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Embedding model set to: {model_name}")
