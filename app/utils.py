"""Utility functions for video processing pipeline."""

import logging

from transcript_extractor import get_transcript

logger = logging.getLogger(__name__)


def process_single_video(db_handler, data_processor, video_id, embedding_model):
    """Process a single video: extract transcript, build index, save to DB.

    Returns the Elasticsearch index name on success, or None on failure.
    """
    try:
        # Skip if already processed
        existing_index = db_handler.get_elasticsearch_index_by_youtube_id(video_id)
        if existing_index:
            logger.info(f"Video {video_id} already processed. Using existing index.")
            return existing_index

        # Extract transcript
        transcript_data = get_transcript(video_id)
        if not transcript_data:
            logger.error(f"Failed to retrieve transcript for video {video_id}.")
            return None

        # Process transcript
        processed_data = data_processor.process_transcript(video_id, transcript_data)
        if not processed_data:
            logger.error(f"Failed to process transcript for video {video_id}.")
            return None

        # Save video metadata to database
        metadata = transcript_data["metadata"]
        video_data = {
            "video_id": video_id,
            "title": metadata.get("title", "Unknown Title"),
            "author": metadata.get("author", "Unknown Author"),
            "upload_date": metadata.get("upload_date", "Unknown Date"),
            "view_count": int(metadata.get("view_count", 0)),
            "like_count": int(metadata.get("like_count", 0)),
            "comment_count": int(metadata.get("comment_count", 0)),
            "video_duration": metadata.get("duration", "Unknown Duration"),
            "transcript_content": processed_data["content"],
        }
        db_handler.add_video(video_data)

        # Build search index
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
                logger.info(f"Successfully processed video: {video_data['title']}")
                return index_name

        logger.error(f"Failed to build index for video {video_id}.")
        return None

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        return None
