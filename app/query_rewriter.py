"""Query rewriting using Chain-of-Thought and ReAct frameworks."""

import logging
import os

import ollama

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites user queries to improve retrieval quality."""

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "phi3")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")

    def generate(self, prompt):
        """Generate a response from the LLM."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

    def rewrite_cot(self, query):
        """Rewrite a query using Chain-of-Thought reasoning."""
        prompt = (
            "Rewrite the following query using Chain-of-Thought reasoning:\n"
            f"Query: {query}\n\n"
            "Rewritten query:"
        )
        result = self.generate(prompt)
        if result is None:
            return query, prompt
        return result, prompt

    def rewrite_react(self, query):
        """Rewrite a query using the ReAct framework."""
        prompt = (
            "Rewrite the following query using the ReAct framework "
            "(Reasoning and Acting):\n"
            f"Query: {query}\n\n"
            "Thought 1:\nAction 1:\nObservation 1:\n\n"
            "Thought 2:\nAction 2:\nObservation 2:\n\n"
            "Final rewritten query:"
        )
        result = self.generate(prompt)
        if result is None:
            return query, prompt
        return result, prompt
