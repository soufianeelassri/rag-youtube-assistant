"""Lightweight search index using TF-IDF and cosine similarity."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Index:
    """A simple search index using TF-IDF for text and exact matching for keywords.

    Attributes:
        text_fields: List of text field names to index with TF-IDF.
        keyword_fields: List of keyword field names for exact matching.
    """

    def __init__(self, text_fields, keyword_fields, vectorizer_params=None):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.vectorizers = {
            field: TfidfVectorizer(**(vectorizer_params or {}))
            for field in text_fields
        }
        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    def fit(self, docs):
        """Build the index from a list of document dictionaries."""
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, "") for doc in docs]
            self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ""))

        self.keyword_df = pd.DataFrame(keyword_data)
        return self

    def search(self, query, filter_dict=None, boost_dict=None, num_results=10):
        """Search the index with optional keyword filters and field boosting.

        Args:
            query: Search query string.
            filter_dict: Keyword field filters (field_name -> value).
            boost_dict: Boost multipliers for text fields (field_name -> weight).
            num_results: Maximum number of results to return.

        Returns:
            List of matching documents ranked by relevance.
        """
        filter_dict = filter_dict or {}
        boost_dict = boost_dict or {}

        query_vecs = {
            field: self.vectorizers[field].transform([query])
            for field in self.text_fields
        }
        scores = np.zeros(len(self.docs))

        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [self.docs[i] for i in top_indices if scores[i] > 0]
