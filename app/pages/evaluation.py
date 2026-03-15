"""RAG evaluation dashboard page."""

import streamlit as st

st.set_page_config(
    page_title="04_Evaluation",
    page_icon="📊",
    layout="wide",
)

import logging

import pandas as pd
from data_processor import DataProcessor
from database import DatabaseHandler
from evaluation import EvaluationSystem
from generate_ground_truth import get_evaluation_display_data
from rag import RAGSystem

logger = logging.getLogger(__name__)

EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for a Youtube transcript assistant.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in the following JSON format:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "Your explanation for the relevance classification"
}}

Requirements:
1. Relevance must be one of the three exact values
2. Provide clear reasoning in the explanation
3. Consider accuracy and completeness of the answer
4. Return valid JSON only
""".strip()


@st.cache_resource
def init_components():
    """Initialize all evaluation components."""
    db_handler = DatabaseHandler()
    data_processor = DataProcessor()
    rag_system = RAGSystem(data_processor)
    evaluation_system = EvaluationSystem(data_processor, db_handler)
    return db_handler, data_processor, rag_system, evaluation_system


def main():
    st.title("RAG Evaluation")

    db_handler, data_processor, rag_system, evaluation_system = init_components()

    try:
        pd.read_csv("data/ground-truth-retrieval.csv")
        ground_truth_available = True
    except FileNotFoundError:
        ground_truth_available = False

    if not ground_truth_available:
        st.warning(
            "No ground truth data available. "
            "Generate it in the Ground Truth Generation page first."
        )
        if st.button("Go to Ground Truth Generation"):
            st.switch_page("pages/ground_truth.py")
        return

    # Display existing evaluations
    existing_evaluations = get_evaluation_display_data()
    if not existing_evaluations.empty:
        st.subheader("Existing Evaluation Results")
        st.dataframe(existing_evaluations)
        csv = existing_evaluations.to_csv(index=False)
        st.download_button(
            label="Download Evaluation Results",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv",
        )

    # Run evaluation
    if st.button("Run Full Evaluation"):
        with st.spinner("Running evaluation (this may take a while)..."):
            try:
                results = evaluation_system.run_full_evaluation(
                    rag_system,
                    "data/ground-truth-retrieval.csv",
                    EVALUATION_PROMPT_TEMPLATE,
                )

                if results:
                    st.subheader("RAG Evaluations")
                    rag_eval_df = pd.DataFrame(results["rag_evaluations"])
                    st.dataframe(rag_eval_df)

                    st.subheader("Search Performance")
                    search_perf_df = pd.DataFrame([results["search_performance"]])
                    st.dataframe(search_perf_df)

                    st.subheader("Optimized Search Parameters")
                    params_df = pd.DataFrame([
                        {
                            "parameter": k,
                            "value": v,
                            "score": results["best_score"],
                        }
                        for k, v in results["best_params"].items()
                    ])
                    st.dataframe(params_df)

                    # Save results to database
                    for video_id in rag_eval_df["video_id"].unique():
                        db_handler.save_search_performance(
                            video_id,
                            results["search_performance"]["hit_rate"],
                            results["search_performance"]["mrr"],
                        )
                        db_handler.save_search_parameters(
                            video_id,
                            results["best_params"],
                            results["best_score"],
                        )

                    st.success("Evaluation complete. Results saved.")
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                logger.error(f"Evaluation error: {e}")


if __name__ == "__main__":
    main()
