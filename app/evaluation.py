"""RAG evaluation system with metrics, LLM-as-Judge, and search optimization."""

import csv
import json
import logging
import sqlite3

import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvaluationSystem:
    """Evaluates RAG system performance using multiple metrics."""

    def __init__(self, data_processor, database_handler):
        self.data_processor = data_processor
        self.db_handler = database_handler

    def relevance_scoring(self, query, retrieved_docs, top_k=5):
        """Compute average cosine similarity between query and top-k documents."""
        query_embedding = self.data_processor.embedding_model.encode(query)
        doc_embeddings = [
            self.data_processor.embedding_model.encode(doc["content"])
            for doc in retrieved_docs
        ]
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        return np.mean(sorted(similarities, reverse=True)[:top_k])

    def answer_similarity(self, generated_answer, reference_answer):
        """Compute cosine similarity between generated and reference answers."""
        gen_emb = self.data_processor.embedding_model.encode(generated_answer)
        ref_emb = self.data_processor.embedding_model.encode(reference_answer)
        return cosine_similarity([gen_emb], [ref_emb])[0][0]

    def human_evaluation(self, video_id, query):
        """Get average user feedback score for a query."""
        with sqlite3.connect(self.db_handler.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT AVG(feedback) FROM user_feedback WHERE video_id = ? AND query = ?",
                (video_id, query),
            )
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0

    def llm_as_judge(self, question, generated_answer, prompt_template):
        """Use an LLM to evaluate answer relevance."""
        prompt = prompt_template.format(
            question=question, answer_llm=generated_answer
        )
        try:
            response = ollama.chat(
                model="phi3.5",
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(response["message"]["content"])
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            return None

    def evaluate_rag(self, rag_system, ground_truth_file, prompt_template=None):
        """Evaluate RAG system against ground truth questions."""
        try:
            ground_truth = pd.read_csv(ground_truth_file)
        except FileNotFoundError:
            logger.error("Ground truth file not found.")
            return None

        evaluations = []

        for _, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
            question = row["question"]
            video_id = row["video_id"]

            index_name = self.db_handler.get_elasticsearch_index_by_youtube_id(video_id)
            if not index_name:
                logger.warning(f"No index for video {video_id}. Skipping.")
                continue

            try:
                answer_llm, _ = rag_system.query(
                    question, search_method="hybrid", index_name=index_name
                )
            except ValueError as e:
                logger.error(f"RAG query error: {e}")
                continue

            if prompt_template:
                evaluation = self.llm_as_judge(question, answer_llm, prompt_template)
                if evaluation:
                    evaluations.append({
                        "video_id": str(video_id),
                        "question": str(question),
                        "answer": str(answer_llm),
                        "relevance": str(evaluation.get("Relevance", "UNKNOWN")),
                        "explanation": str(
                            evaluation.get("Explanation", "No explanation provided")
                        ),
                    })
            else:
                similarity = self.answer_similarity(
                    answer_llm, row.get("reference_answer", "")
                )
                evaluations.append({
                    "video_id": str(video_id),
                    "question": str(question),
                    "answer": str(answer_llm),
                    "relevance": f"Similarity: {similarity:.4f}",
                    "explanation": "Cosine similarity evaluation",
                })

        # Save results to CSV
        csv_path = "data/evaluation_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["video_id", "question", "answer", "relevance", "explanation"],
            )
            writer.writeheader()
            writer.writerows(evaluations)
        logger.info(f"Evaluation results saved to {csv_path}")

        # Save to database
        self._save_evaluations_to_db(evaluations)

        return evaluations

    def _save_evaluations_to_db(self, evaluations):
        """Persist evaluation results to the database."""
        for eval_data in evaluations:
            self.db_handler.save_rag_evaluation(eval_data)
        logger.info(f"Saved {len(evaluations)} evaluations to database.")

    def run_full_evaluation(self, rag_system, ground_truth_file, prompt_template=None):
        """Run complete evaluation: RAG + search performance + optimization."""
        ground_truth = pd.read_csv(ground_truth_file)

        # Evaluate RAG answers
        rag_evaluations = self.evaluate_rag(
            rag_system, ground_truth_file, prompt_template
        )

        # Evaluate search performance
        def search_function(query, video_id):
            index_name = self.db_handler.get_elasticsearch_index_by_youtube_id(video_id)
            if index_name:
                return rag_system.data_processor.search(
                    query, num_results=10, method="hybrid", index_name=index_name
                )
            return []

        search_performance = self.evaluate_search(ground_truth, search_function)

        # Optimize search parameters
        param_ranges = {"content": (0.0, 3.0)}

        def objective_function(params):
            def parameterized_search(query, video_id):
                index_name = self.db_handler.get_elasticsearch_index_by_youtube_id(
                    video_id
                )
                if index_name:
                    return rag_system.data_processor.search(
                        query,
                        num_results=10,
                        method="hybrid",
                        index_name=index_name,
                        boost_dict=params,
                    )
                return []
            return self.evaluate_search(ground_truth, parameterized_search)["mrr"]

        best_params, best_score = self.simple_optimize(
            param_ranges, objective_function
        )

        return {
            "rag_evaluations": rag_evaluations,
            "search_performance": search_performance,
            "best_params": best_params,
            "best_score": best_score,
        }

    @staticmethod
    def hit_rate(relevance_total):
        """Calculate the fraction of queries with at least one relevant result."""
        return sum(any(line) for line in relevance_total) / len(relevance_total)

    @staticmethod
    def mrr(relevance_total):
        """Calculate Mean Reciprocal Rank."""
        scores = []
        for line in relevance_total:
            for rank, relevant in enumerate(line, 1):
                if relevant:
                    scores.append(1 / rank)
                    break
            else:
                scores.append(0)
        return sum(scores) / len(scores)

    @staticmethod
    def simple_optimize(param_ranges, objective_function, n_iterations=10):
        """Random search optimization over parameter ranges."""
        best_params = None
        best_score = float("-inf")
        for _ in range(n_iterations):
            current_params = {
                param: np.random.uniform(min_val, max_val)
                for param, (min_val, max_val) in param_ranges.items()
            }
            current_score = objective_function(current_params)
            if current_score > best_score:
                best_score = current_score
                best_params = current_params
        return best_params, best_score

    def evaluate_search(self, ground_truth, search_function):
        """Evaluate search quality using hit rate and MRR."""
        relevance_total = []
        for _, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):
            video_id = row["video_id"]
            results = search_function(row["question"], video_id)
            relevance = [d["video_id"] == video_id for d in results]
            relevance_total.append(relevance)
        return {
            "hit_rate": self.hit_rate(relevance_total),
            "mrr": self.mrr(relevance_total),
        }
