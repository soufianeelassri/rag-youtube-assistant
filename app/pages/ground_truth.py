"""Ground truth generation and management page."""

import streamlit as st

st.set_page_config(
    page_title="03_Ground_Truth",
    page_icon="📝",
    layout="wide",
)

import logging

import pandas as pd
from data_processor import DataProcessor
from database import DatabaseHandler
from generate_ground_truth import generate_ground_truth, get_ground_truth_display_data

logger = logging.getLogger(__name__)


@st.cache_resource
def init_components():
    """Initialize database and data processor."""
    return DatabaseHandler(), DataProcessor()


def main():
    st.title("Ground Truth Generation")

    db_handler, data_processor = init_components()

    videos = db_handler.get_all_videos()
    if not videos:
        st.warning("No videos available. Process videos in the Data Ingestion page first.")
        return

    video_df = pd.DataFrame(
        videos, columns=["youtube_id", "title", "channel_name", "upload_date"]
    )

    # Channel filter
    channels = sorted(video_df["channel_name"].unique())
    selected_channel = st.selectbox("Filter by Channel", ["All"] + channels)

    if selected_channel != "All":
        video_df = video_df[video_df["channel_name"] == selected_channel]
        gt_data = get_ground_truth_display_data(
            db_handler, channel_name=selected_channel
        )
        if not gt_data.empty:
            st.subheader("Existing Ground Truth Questions for Channel")
            st.dataframe(gt_data)
            csv = gt_data.to_csv(index=False)
            st.download_button(
                label="Download Channel Ground Truth CSV",
                data=csv,
                file_name=f"ground_truth_{selected_channel}.csv",
                mime="text/csv",
            )

    st.subheader("Available Videos")
    st.dataframe(video_df)

    # Video selection
    selected_video_id = st.selectbox(
        "Select a Video",
        video_df["youtube_id"].tolist(),
        format_func=lambda x: video_df[video_df["youtube_id"] == x]["title"].iloc[0],
    )

    if selected_video_id:
        if st.button("Generate Ground Truth Questions"):
            with st.spinner("Generating questions..."):
                try:
                    questions_df = generate_ground_truth(
                        db_handler, data_processor, selected_video_id
                    )
                    if questions_df is not None and not questions_df.empty:
                        st.success("Successfully generated ground truth questions.")
                        st.dataframe(questions_df)
                    else:
                        st.error("Failed to generate ground truth questions.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Error in ground truth generation: {e}")

        # Display existing ground truth
        gt_data = get_ground_truth_display_data(
            db_handler, video_id=selected_video_id
        )
        if not gt_data.empty:
            st.subheader("Existing Ground Truth Questions")
            st.dataframe(gt_data)
            csv = gt_data.to_csv(index=False)
            st.download_button(
                label="Download Ground Truth CSV",
                data=csv,
                file_name=f"ground_truth_{selected_video_id}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
