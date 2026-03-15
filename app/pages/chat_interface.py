"""Chat interface page for querying video transcripts."""

import streamlit as st

st.set_page_config(
    page_title="02_Chat_Interface",
    page_icon="💬",
    layout="wide",
)

import logging
import sqlite3
from datetime import datetime

import pandas as pd
from data_processor import DataProcessor
from database import DatabaseHandler
from query_rewriter import QueryRewriter
from rag import RAGSystem
from utils import process_single_video

logger = logging.getLogger(__name__)

SEARCH_METHOD_MAP = {
    "Hybrid": "hybrid",
    "Text-only": "text",
    "Embedding-only": "embedding",
}


@st.cache_resource
def init_components():
    """Initialize system components."""
    try:
        db_handler = DatabaseHandler()
        data_processor = DataProcessor()
        rag_system = RAGSystem(data_processor)
        query_rewriter = QueryRewriter()
        return db_handler, data_processor, rag_system, query_rewriter
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        st.error(f"Error initializing components: {e}")
        return None, None, None, None


def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()


def handle_feedback(db_handler, video_id, chat_id, query, response, feedback_value, message_key):
    """Record user feedback and update session state."""
    db_handler.add_user_feedback(
        video_id=video_id,
        chat_id=chat_id,
        query=query,
        response=response,
        feedback=feedback_value,
    )
    st.session_state.feedback_given.add(message_key)
    label = "positive" if feedback_value == 1 else "negative"
    st.success(f"Thank you for your {label} feedback!")
    st.rerun()


def render_feedback_buttons(db_handler, video_id, message, message_key):
    """Render thumbs up/down buttons for a message."""
    if message_key in st.session_state.feedback_given:
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍", key=f"like_{message_key}"):
            handle_feedback(
                db_handler, video_id, message["id"],
                message["user"], message["assistant"], 1, message_key,
            )
    with col2:
        if st.button("👎", key=f"dislike_{message_key}"):
            handle_feedback(
                db_handler, video_id, message["id"],
                message["user"], message["assistant"], -1, message_key,
            )


def create_chat_interface(db_handler, rag_system, video_id, index_name, rewrite_method, search_method):
    """Create the chat interface with feedback functionality."""
    # Reload chat history when video changes
    if st.session_state.current_video_id != video_id:
        st.session_state.chat_history = []
        db_history = db_handler.get_chat_history(video_id)
        for chat_id, user_msg, asst_msg, timestamp in db_history:
            st.session_state.chat_history.append({
                "id": chat_id,
                "user": user_msg,
                "assistant": asst_msg,
                "timestamp": timestamp,
            })
        st.session_state.current_video_id = video_id

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["assistant"])
            render_feedback_buttons(
                db_handler, video_id, message, str(message["id"])
            )

    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Apply query rewriting if selected
                    rewritten_query = prompt
                    if rewrite_method == "Chain of Thought":
                        rewritten_query, _ = rag_system.rewrite_cot(prompt)
                        st.caption(f"Rewritten query: {rewritten_query}")
                    elif rewrite_method == "ReAct":
                        rewritten_query, _ = rag_system.rewrite_react(prompt)
                        st.caption(f"Rewritten query: {rewritten_query}")

                    # Get response
                    response, _ = rag_system.query(
                        rewritten_query,
                        search_method=SEARCH_METHOD_MAP[search_method],
                        index_name=index_name,
                    )
                    st.markdown(response)

                    # Save to database and session state
                    chat_id = db_handler.add_chat_message(video_id, prompt, response)
                    message = {
                        "id": chat_id,
                        "user": prompt,
                        "assistant": response,
                        "timestamp": datetime.now(),
                    }
                    st.session_state.chat_history.append(message)

                    render_feedback_buttons(
                        db_handler, video_id, message, str(chat_id)
                    )

                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.error(f"Error in chat interface: {e}")


def get_system_status(db_handler, selected_video_id=None):
    """Get system status information for the sidebar."""
    try:
        with sqlite3.connect(db_handler.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT index_name) FROM elasticsearch_indices")
            total_indices = cursor.fetchone()[0]

            cursor.execute("SELECT model_name FROM embedding_models")
            models = [row[0] for row in cursor.fetchall()]

            video_details = None
            if selected_video_id:
                cursor.execute(
                    """
                    SELECT v.id, v.title, v.channel_name, v.processed_date,
                           ei.index_name, em.model_name
                    FROM videos v
                    LEFT JOIN elasticsearch_indices ei ON v.id = ei.video_id
                    LEFT JOIN embedding_models em ON ei.embedding_model_id = em.id
                    WHERE v.youtube_id = ?
                    """,
                    (selected_video_id,),
                )
                video_details = cursor.fetchall()

            return {
                "total_videos": total_videos,
                "total_indices": total_indices,
                "models": models,
                "video_details": video_details,
            }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return None


def display_system_status(status, selected_video_id=None):
    """Display system status in the sidebar."""
    if not status:
        st.sidebar.error("Unable to fetch system status.")
        return

    st.sidebar.header("System Status")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Videos", status["total_videos"])
    with col2:
        st.metric("Total Indices", status["total_indices"])

    if status["models"]:
        st.sidebar.markdown("**Available Models:**")
        for model in status["models"]:
            st.sidebar.markdown(f"- {model}")

    if selected_video_id and status.get("video_details"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Selected Video:**")
        for _vid, title, channel, processed, index, model in status["video_details"]:
            st.sidebar.markdown(
                f"- **Title:** {title}\n"
                f"- **Channel:** {channel}\n"
                f"- **Processed:** {processed}\n"
                f"- **Index:** {index or 'Not indexed'}\n"
                f"- **Model:** {model or 'N/A'}"
            )


def main():
    st.title("Chat Interface")

    components = init_components()
    if not all(components):
        st.error("Failed to initialize components. Check the logs.")
        return

    db_handler, data_processor, rag_system, query_rewriter = components
    init_session_state()

    system_status = get_system_status(db_handler)

    # Video selection
    st.sidebar.header("Video Selection")

    with sqlite3.connect(db_handler.db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT v.youtube_id, v.title, v.channel_name, v.upload_date,
                   GROUP_CONCAT(ei.index_name) as indices
            FROM videos v
            LEFT JOIN elasticsearch_indices ei ON v.id = ei.video_id
            GROUP BY v.youtube_id
            ORDER BY v.upload_date DESC
            """,
            conn,
        )

    if df.empty:
        st.info("No videos available. Process videos in the Data Ingestion page first.")
        display_system_status(system_status)
        return

    st.sidebar.markdown(f"**Available Videos:** {len(df)}")

    channels = sorted(df["channel_name"].unique())
    selected_channel = st.sidebar.selectbox(
        "Filter by Channel", ["All"] + channels, key="channel_filter"
    )

    filtered_df = df if selected_channel == "All" else df[df["channel_name"] == selected_channel]

    selected_video_id = st.sidebar.selectbox(
        "Select a Video",
        filtered_df["youtube_id"].tolist(),
        format_func=lambda x: filtered_df[filtered_df["youtube_id"] == x]["title"].iloc[0],
        key="video_select",
    )

    if selected_video_id:
        system_status = get_system_status(db_handler, selected_video_id)
        display_system_status(system_status, selected_video_id)

        index_name = db_handler.get_elasticsearch_index_by_youtube_id(selected_video_id)

        if not index_name:
            st.warning("This video hasn't been indexed yet.")
            if st.button("Process Now"):
                with st.spinner("Processing video..."):
                    try:
                        embedding_model = "multi-qa-MiniLM-L6-cos-v1"
                        index_name = process_single_video(
                            db_handler, data_processor, selected_video_id, embedding_model
                        )
                        if index_name:
                            st.success("Video processed successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
        else:
            st.sidebar.header("Chat Settings")
            rewrite_method = st.sidebar.radio(
                "Query Rewriting Method",
                ["None", "Chain of Thought", "ReAct"],
                key="rewrite_method",
            )
            search_method = st.sidebar.radio(
                "Search Method",
                ["Hybrid", "Text-only", "Embedding-only"],
                key="search_method",
            )

            create_chat_interface(
                db_handler, rag_system, selected_video_id,
                index_name, rewrite_method, search_method,
            )


if __name__ == "__main__":
    main()
