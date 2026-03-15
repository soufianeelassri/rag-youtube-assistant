"""RAG (Retrieval-Augmented Generation) system for video transcript Q&A."""

import logging
import os
import time

import ollama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """
You are an AI assistant analyzing YouTube video transcripts. Answer questions
based on the provided transcript context.

Context from transcript:
{context}

User Question: {question}

Guidelines:
1. Use only information from the provided context
2. Be specific and direct in your answer
3. If context is insufficient, say so
4. Maintain accuracy and avoid speculation
5. Use natural, conversational language
""".strip()


class RAGSystem:
    """Retrieval-Augmented Generation system using Ollama and Elasticsearch."""

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = os.getenv("OLLAMA_MODEL", "phi3")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", 240))
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", 3))
        self._check_ollama_service()

    def _check_ollama_service(self):
        """Verify Ollama is accessible and pull the model."""
        try:
            ollama.list()
            logger.info("Ollama service is accessible.")
            self._pull_model()
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_host}: {e}")

    def _pull_model(self):
        """Download the specified Ollama model if not already available."""
        try:
            ollama.pull(self.model)
            logger.info(f"Model '{self.model}' is ready.")
        except Exception as e:
            logger.error(f"Error pulling model '{self.model}': {e}")

    def generate(self, prompt):
        """Generate a response from the LLM with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response["message"]["content"]
            except Exception as e:
                logger.error(
                    f"Generation attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error("All retries exhausted.")
                    return None
                time.sleep(2 ** attempt)

    def get_prompt(self, user_query, relevant_docs):
        """Build a RAG prompt from the query and retrieved documents."""
        context = "\n".join(doc["content"] for doc in relevant_docs)
        return RAG_PROMPT_TEMPLATE.format(context=context, question=user_query)

    def query(self, user_query, search_method="hybrid", index_name=None):
        """Execute the full RAG pipeline: search -> prompt -> generate."""
        try:
            if not index_name:
                raise ValueError(
                    "No index name provided. Select a video and ensure it has been processed."
                )

            relevant_docs = self.data_processor.search(
                user_query, num_results=3, method=search_method, index_name=index_name
            )

            if not relevant_docs:
                logger.warning("No relevant documents found.")
                return (
                    "I couldn't find any relevant information to answer your query.",
                    "",
                )

            prompt = self.get_prompt(user_query, relevant_docs)
            answer = self.generate(prompt)

            if answer is None:
                return "Failed to generate a response. Please try again.", prompt

            return answer, prompt
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return f"An error occurred: {e}", ""

    def rewrite_cot(self, query):
        """Rewrite a query using Chain-of-Thought reasoning."""
        prompt = (
            f"Rewrite the following query using chain-of-thought reasoning:\n\n"
            f"Query: {query}\n\nRewritten query:"
        )
        response = self.generate(prompt)
        return (response or query, prompt)

    def rewrite_react(self, query):
        """Rewrite a query using the ReAct framework."""
        prompt = (
            f"Rewrite the following query using ReAct (Reasoning and Acting) approach:\n\n"
            f"Query: {query}\n\nRewritten query:"
        )
        response = self.generate(prompt)
        return (response or query, prompt)
