"""Ground truth question generation from video transcripts."""

import json
import logging
import os
import sqlite3

import ollama
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm

logger = logging.getLogger(__name__)

QUESTION_GENERATION_PROMPT = """
You are an AI assistant tasked with generating questions based on a YouTube video transcript.
Formulate EXACTLY 10 questions that a user might ask based on the provided transcript.
Make the questions specific to the content of the transcript.
The questions should be complete and not too short. Use as few words as possible from the transcript.
Ensure that all 10 questions are unique and not repetitive.

The transcript:

{transcript}

Provide the output in parsable JSON without using code blocks:

{{"questions": ["question1", "question2", ..., "question10"]}}
""".strip()


def _get_transcript_from_elasticsearch(es, index_name, video_id):
    """Retrieve transcript content from an Elasticsearch index."""
    try:
        result = es.search(
            index=index_name,
            body={"query": {"match": {"video_id": video_id}}},
        )
        if result["hits"]["hits"]:
            return result["hits"]["hits"][0]["_source"]["content"]
    except Exception as e:
        logger.error(f"Error retrieving transcript from Elasticsearch: {e}")
    return None


def _get_transcript_from_sqlite(db_path, video_id):
    """Retrieve transcript content from the SQLite database."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT transcript_content FROM videos WHERE youtube_id = ?",
                (video_id,),
            )
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"Error retrieving transcript from SQLite: {e}")
    return None


def generate_questions(transcript, max_retries=3):
    """Generate test questions from a transcript using an LLM."""
    all_questions = set()
    retries = 0

    while len(all_questions) < 10 and retries < max_retries:
        prompt = QUESTION_GENERATION_PROMPT.format(transcript=transcript)
        try:
            response = ollama.chat(
                model="phi3.5",
                messages=[{"role": "user", "content": prompt}],
            )
            questions = json.loads(response["message"]["content"])["questions"]
            all_questions.update(questions)
        except Exception as e:
            logger.error(f"Error generating questions (attempt {retries + 1}): {e}")
        retries += 1

    if len(all_questions) < 10:
        logger.warning(
            f"Generated only {len(all_questions)} unique questions "
            f"after {max_retries} attempts."
        )

    return {"questions": list(all_questions)[:10]}


def generate_ground_truth(db_handler, data_processor, video_id):
    """Generate ground truth questions for a video and save to DB and CSV."""
    es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
    es_port = os.getenv("ELASTICSEARCH_PORT", "9200")
    es = Elasticsearch([f"http://{es_host}:{es_port}"])

    # Get existing questions to avoid duplicates
    existing_questions = {
        q[2] for q in db_handler.get_ground_truth_by_video(video_id)
    }

    # Try to get transcript from ES first, then SQLite
    transcript = None
    index_name = db_handler.get_elasticsearch_index_by_youtube_id(video_id)

    if index_name:
        transcript = _get_transcript_from_elasticsearch(es, index_name, video_id)

    if not transcript:
        transcript = _get_transcript_from_sqlite(db_handler.db_path, video_id)

    if not transcript:
        logger.error(f"No transcript found for video {video_id}.")
        return None

    # Generate unique questions
    all_questions = set()
    max_attempts = 3
    for attempt in range(max_attempts):
        result = generate_questions(transcript)
        if result and "questions" in result:
            new_questions = set(result["questions"]) - existing_questions
            all_questions.update(new_questions)
        if len(all_questions) >= 10:
            break

    if not all_questions:
        logger.error("Failed to generate any unique questions.")
        return None

    # Save to database
    db_handler.add_ground_truth_questions(video_id, all_questions)

    # Save to CSV (append if exists)
    df = pd.DataFrame(
        [(video_id, q) for q in all_questions], columns=["video_id", "question"]
    )
    csv_path = "data/ground-truth-retrieval.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    logger.info(f"Ground truth saved to {csv_path} ({len(all_questions)} questions).")
    return df


def get_ground_truth_display_data(db_handler, video_id=None, channel_name=None):
    """Retrieve ground truth data from database and CSV for display."""
    # Get database data
    if video_id:
        data = db_handler.get_ground_truth_by_video(video_id)
    elif channel_name:
        data = db_handler.get_ground_truth_by_channel(channel_name)
    else:
        data = []

    db_df = (
        pd.DataFrame(
            data, columns=["id", "video_id", "question", "generation_date", "channel_name"]
        )
        if data
        else pd.DataFrame()
    )

    # Get CSV data
    try:
        csv_df = pd.read_csv("data/ground-truth-retrieval.csv")
        if video_id:
            csv_df = csv_df[csv_df["video_id"] == video_id]
        elif channel_name:
            videos_df = pd.DataFrame(
                db_handler.get_all_videos(),
                columns=["youtube_id", "title", "channel_name", "upload_date"],
            )
            csv_df = csv_df.merge(videos_df, left_on="video_id", right_on="youtube_id")
            csv_df = csv_df[csv_df["channel_name"] == channel_name]
    except FileNotFoundError:
        csv_df = pd.DataFrame()

    # Combine sources
    if not db_df.empty and not csv_df.empty:
        return pd.concat([db_df, csv_df]).drop_duplicates(
            subset=["video_id", "question"]
        )
    return db_df if not db_df.empty else csv_df


def generate_ground_truth_for_all_videos(db_handler, data_processor):
    """Generate ground truth for all processed videos."""
    videos = db_handler.get_all_videos()
    all_questions = []

    for video in tqdm(videos, desc="Generating ground truth"):
        video_id = video[0]
        df = generate_ground_truth(db_handler, data_processor, video_id)
        if df is not None:
            all_questions.extend(df.values.tolist())

    if not all_questions:
        logger.error("Failed to generate questions for any video.")
        return None

    df = pd.DataFrame(all_questions, columns=["video_id", "question"])
    csv_path = "data/ground-truth-retrieval.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"All ground truth saved to {csv_path}.")
    return df


def get_evaluation_display_data(video_id=None):
    """Load evaluation results from CSV for display."""
    try:
        df = pd.read_csv("data/evaluation_results.csv")
        if video_id:
            df = df[df["video_id"] == video_id]
        return df
    except FileNotFoundError:
        return pd.DataFrame()
