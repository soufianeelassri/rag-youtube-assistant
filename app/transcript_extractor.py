"""YouTube transcript extraction and API integration."""

import logging
import os
import re

import certifi
import googleapiclient.http
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(os.path.dirname(current_dir), ".env")
load_dotenv(dotenv_path)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError(
        "YouTube API key not found. "
        "Set YOUTUBE_API_KEY in your .env file."
    )

logger.info("YouTube API key loaded successfully.")


def get_youtube_client():
    """Initialize and return a YouTube API client."""
    try:
        session = requests.Session()
        session.verify = certifi.where()

        http = googleapiclient.http.build_http()
        http.verify = session.verify

        youtube = build("youtube", "v3", developerKey=API_KEY, http=http)
        logger.info("YouTube API client initialized successfully.")
        return youtube
    except Exception as e:
        logger.error(f"Error initializing YouTube API client: {e}")
        raise


def extract_video_id(url):
    """Extract video ID from a YouTube URL."""
    if not url:
        return None
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None


def get_video_metadata(video_id):
    """Retrieve metadata for a YouTube video."""
    youtube = get_youtube_client()
    try:
        response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id,
        ).execute()

        if not response.get("items"):
            logger.error(f"No video found with id: {video_id}")
            return None

        video = response["items"][0]
        snippet = video["snippet"]
        stats = video["statistics"]
        description = snippet.get("description", "").strip() or "Not Available"

        return {
            "title": snippet["title"],
            "author": snippet["channelTitle"],
            "upload_date": snippet["publishedAt"],
            "view_count": stats.get("viewCount", "0"),
            "like_count": stats.get("likeCount", "0"),
            "comment_count": stats.get("commentCount", "0"),
            "duration": video["contentDetails"]["duration"],
            "description": description,
        }
    except Exception as e:
        logger.error(f"Error fetching metadata for video {video_id}: {e}")
        return None


def get_transcript(video_id):
    """Fetch transcript and metadata for a YouTube video."""
    if not video_id:
        return None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        metadata = get_video_metadata(video_id)

        if not metadata:
            return None

        logger.info(
            f"Retrieved transcript for video {video_id} "
            f"({len(transcript)} segments)."
        )
        return {"transcript": transcript, "metadata": metadata}
    except Exception as e:
        logger.error(f"Error extracting transcript for video {video_id}: {e}")
        return None


def extract_channel_id(url):
    """Extract channel ID from a YouTube channel URL."""
    match = re.search(r"(?:channel\/|c\/|@)([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None


def get_channel_videos(channel_url):
    """List videos from a YouTube channel."""
    youtube = get_youtube_client()
    channel_id = extract_channel_id(channel_url)
    if not channel_id:
        logger.error(f"Invalid channel URL: {channel_url}")
        return []

    try:
        response = youtube.search().list(
            part="id,snippet",
            channelId=channel_id,
            type="video",
            maxResults=50,
        ).execute()

        return [
            {
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "published_at": item["snippet"]["publishedAt"],
            }
            for item in response["items"]
        ]
    except HttpError as e:
        logger.error(f"HTTP error {e.resp.status}: {e.content}")
        return []
    except Exception as e:
        logger.error(f"Error fetching channel videos: {e}")
        return []


def test_api_key():
    """Validate the YouTube API key."""
    try:
        youtube = get_youtube_client()
        response = youtube.videos().list(
            part="snippet", id="dQw4w9WgXcQ"
        ).execute()
        if "items" in response:
            logger.info("API key is valid.")
            return True
        logger.error("API key test returned unexpected response.")
        return False
    except Exception as e:
        logger.error(f"API key test failed: {e}")
        return False


def initialize_youtube_api():
    """Initialize and test the YouTube API connection."""
    if test_api_key():
        logger.info("YouTube API initialized successfully.")
        return True
    logger.error("Failed to initialize YouTube API.")
    return False
