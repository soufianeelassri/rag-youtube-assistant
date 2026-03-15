"""Data ingestion page for processing YouTube videos."""

import streamlit as st

st.set_page_config(
    page_title="01_Data_Ingestion",
    page_icon="📥",
    layout="wide",
)

import logging

import pandas as pd
from data_processor import DataProcessor
from database import DatabaseHandler
from transcript_extractor import extract_video_id, get_channel_videos, get_transcript

logger = logging.getLogger(__name__)


@st.cache_resource
def init_components():
    """Initialize database and data processor."""
    return DatabaseHandler(), DataProcessor()


def process_video(db_handler, data_processor, video_id, embedding_model):
    """Process a single video: extract, index, and save."""
    try:
        existing_index = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
        if existing_index:
            st.info(f"Video {video_id} already processed.")
            return existing_index

        transcript_data = get_transcript(video_id)
        if not transcript_data:
            st.error("Failed to retrieve transcript.")
            return None

        processed_data = data_processor.process_transcript(video_id, transcript_data)
        if not processed_data:
            st.error("Failed to process transcript.")
            return None

        metadata = transcript_data["metadata"]
        video_data = {
            "video_id": video_id,
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "upload_date": metadata.get("upload_date", ""),
            "view_count": int(metadata.get("view_count", 0)),
            "like_count": int(metadata.get("like_count", 0)),
            "comment_count": int(metadata.get("comment_count", 0)),
            "video_duration": metadata.get("duration", ""),
            "transcript_content": processed_data["content"],
        }

        db_handler.add_video(video_data)

        index_name = f"video_{video_id}_{embedding_model}".lower()
        index_name = data_processor.build_index(index_name)

        if index_name:
            embedding_model_id = db_handler.add_embedding_model(
                embedding_model, f"Embedding model: {embedding_model}"
            )
            video_record = db_handler.get_video_by_youtube_id(video_id)
            if video_record:
                db_handler.add_elasticsearch_index(
                    video_record[0], index_name, embedding_model_id
                )
            st.success(f"Successfully processed: {video_data['title']}")
            return index_name
    except Exception as e:
        st.error(f"Error processing video: {e}")
        logger.error(f"Error processing video {video_id}: {e}")
    return None


def process_batch(db_handler, data_processor, video_ids, embedding_model):
    """Process multiple videos with a progress bar."""
    progress_bar = st.progress(0)
    processed = 0
    total = len(video_ids)

    for video_id in video_ids:
        if process_video(db_handler, data_processor, video_id, embedding_model):
            processed += 1
        progress_bar.progress((processed) / total if total > 0 else 1)

    st.success(f"Processed {processed} out of {total} videos.")


def main():
    st.title("Data Ingestion")

    db_handler, data_processor = init_components()

    embedding_model = st.selectbox(
        "Select embedding model:",
        ["multi-qa-MiniLM-L6-cos-v1", "all-mpnet-base-v2"],
    )

    # Display existing videos
    st.header("Processed Videos")
    videos = db_handler.get_all_videos()
    if videos:
        video_df = pd.DataFrame(
            videos, columns=["youtube_id", "title", "channel_name", "upload_date"]
        )
        channels = sorted(video_df["channel_name"].unique())
        selected_channel = st.selectbox("Filter by Channel", ["All"] + channels)
        if selected_channel != "All":
            video_df = video_df[video_df["channel_name"] == selected_channel]
        st.dataframe(video_df)
    else:
        st.info("No videos processed yet. Use the form below to add videos.")

    # Process new videos
    st.header("Process New Video")
    with st.form("process_video_form"):
        input_type = st.radio(
            "Select input type:", ["Video URL", "Channel URL", "YouTube ID"]
        )
        input_value = st.text_input("Enter the URL or ID:")
        submit_button = st.form_submit_button("Process")

        if submit_button and input_value:
            data_processor.set_embedding_model(embedding_model)

            with st.spinner("Processing..."):
                if input_type == "Video URL":
                    video_id = extract_video_id(input_value)
                    if video_id:
                        process_video(
                            db_handler, data_processor, video_id, embedding_model
                        )
                    else:
                        st.error("Could not extract video ID from URL.")

                elif input_type == "Channel URL":
                    channel_videos = get_channel_videos(input_value)
                    if channel_videos:
                        video_ids = [v["video_id"] for v in channel_videos]
                        process_batch(
                            db_handler, data_processor, video_ids, embedding_model
                        )
                    else:
                        st.error("Failed to retrieve videos from the channel.")

                else:  # YouTube ID
                    process_video(
                        db_handler, data_processor, input_value, embedding_model
                    )


if __name__ == "__main__":
    main()
