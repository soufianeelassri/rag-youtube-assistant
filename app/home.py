"""Main entry point for the YouTube Transcript RAG System."""

import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide",
)

import logging
import os
import sys

from transcript_extractor import test_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    st.title("YouTube Transcript RAG System")
    st.write("Welcome to the YouTube Transcript RAG System!")

    if not test_api_key():
        st.error(
            "YouTube API key is invalid or not set. "
            "Please check your configuration."
        )
        new_api_key = st.text_input("Enter your YouTube API key:")
        if new_api_key:
            os.environ["YOUTUBE_API_KEY"] = new_api_key
            if test_api_key():
                st.success("API key validated successfully!")
                st.rerun()
            else:
                st.error("Invalid API key. Please try again.")
        return

    st.success("System is ready! Use the sidebar to navigate between pages.")

    st.header("System Overview")
    st.markdown(
        """
        **1. Data Ingestion** - Process YouTube videos and transcripts
        (single videos or entire channels).

        **2. Chat Interface** - Interactive chat with processed videos,
        with multiple query rewriting and search strategies.

        **3. Ground Truth Generation** - Generate and manage ground truth
        questions for evaluation.

        **4. RAG Evaluation** - Evaluate system performance with
        detailed metrics and analytics.
        """
    )


if __name__ == "__main__":
    main()
