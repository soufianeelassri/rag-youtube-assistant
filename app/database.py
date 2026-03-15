"""SQLite database handler for video metadata, chat history, and evaluations."""

import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Manages all SQLite database operations."""

    def __init__(self, db_path="data/sqlite.db"):
        self.db_path = db_path
        self.conn = None
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._create_tables()
        self._update_schema()
        self._migrate_database()

    def _get_connection(self):
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        """Create all required database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    youtube_id TEXT UNIQUE,
                    title TEXT,
                    channel_name TEXT,
                    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    upload_date TEXT,
                    view_count INTEGER,
                    like_count INTEGER,
                    comment_count INTEGER,
                    video_duration TEXT,
                    transcript_content TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    chat_id INTEGER,
                    query TEXT,
                    response TEXT,
                    feedback INTEGER CHECK (feedback IN (-1, 1)),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id),
                    FOREIGN KEY (chat_id) REFERENCES chat_history (id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE,
                    description TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS elasticsearch_indices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    index_name TEXT,
                    embedding_model_id INTEGER,
                    FOREIGN KEY (video_id) REFERENCES videos (id),
                    FOREIGN KEY (embedding_model_id) REFERENCES embedding_models (id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ground_truth (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    question TEXT,
                    generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(video_id, question),
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    hit_rate REAL,
                    mrr REAL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    parameter_name TEXT,
                    parameter_value REAL,
                    score REAL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    question TEXT,
                    answer TEXT,
                    relevance TEXT,
                    explanation TEXT,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (youtube_id)
                )
            """)

            conn.commit()

    def _update_schema(self):
        """Add missing columns to existing tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(videos)")
            existing_columns = {col[1] for col in cursor.fetchall()}

            new_columns = [
                ("upload_date", "TEXT"),
                ("view_count", "INTEGER"),
                ("like_count", "INTEGER"),
                ("comment_count", "INTEGER"),
                ("video_duration", "TEXT"),
                ("transcript_content", "TEXT"),
            ]

            for col_name, col_type in new_columns:
                if col_name not in existing_columns:
                    cursor.execute(
                        f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}"
                    )

            conn.commit()

    def _migrate_database(self):
        """Run database migrations for schema changes."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(user_feedback)")
                columns = {col[1] for col in cursor.fetchall()}

                if "chat_id" not in columns:
                    logger.info("Migrating user_feedback table to add chat_id.")
                    cursor.execute("""
                        CREATE TABLE user_feedback_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            video_id TEXT,
                            query TEXT,
                            response TEXT,
                            feedback INTEGER CHECK (feedback IN (-1, 1)),
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            chat_id INTEGER,
                            FOREIGN KEY (video_id) REFERENCES videos (youtube_id),
                            FOREIGN KEY (chat_id) REFERENCES chat_history (id)
                        )
                    """)
                    cursor.execute("""
                        INSERT INTO user_feedback_new
                            (video_id, query, response, feedback, timestamp)
                        SELECT video_id, query, response, feedback, timestamp
                        FROM user_feedback
                    """)
                    cursor.execute("DROP TABLE user_feedback")
                    cursor.execute(
                        "ALTER TABLE user_feedback_new RENAME TO user_feedback"
                    )
                    logger.info("Migration completed successfully.")

                conn.commit()
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            raise

    # --- Video Management ---

    def add_video(self, video_data):
        """Insert or update a video record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO videos
                    (youtube_id, title, channel_name, upload_date, view_count,
                     like_count, comment_count, video_duration, transcript_content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        video_data["video_id"],
                        video_data["title"],
                        video_data["author"],
                        video_data["upload_date"],
                        video_data["view_count"],
                        video_data["like_count"],
                        video_data["comment_count"],
                        video_data["video_duration"],
                        video_data["transcript_content"],
                    ),
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error adding video: {e}")
            raise

    def get_video_by_youtube_id(self, youtube_id):
        """Retrieve a video record by YouTube ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM videos WHERE youtube_id = ?", (youtube_id,)
            )
            return cursor.fetchone()

    def get_all_videos(self):
        """Retrieve all videos ordered by upload date."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT youtube_id, title, channel_name, upload_date
                FROM videos ORDER BY upload_date DESC
            """)
            return cursor.fetchall()

    # --- Chat & Feedback ---

    def add_chat_message(self, video_id, user_message, assistant_message):
        """Save a chat message pair to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_history (video_id, user_message, assistant_message)
                VALUES (?, ?, ?)
                """,
                (video_id, user_message, assistant_message),
            )
            conn.commit()
            return cursor.lastrowid

    def get_chat_history(self, video_id):
        """Retrieve chat history for a video."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_message, assistant_message, timestamp
                FROM chat_history WHERE video_id = ?
                ORDER BY timestamp ASC
                """,
                (video_id,),
            )
            return cursor.fetchall()

    def add_user_feedback(self, video_id, chat_id, query, response, feedback):
        """Record user feedback for a chat response."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT id FROM videos WHERE youtube_id = ?", (video_id,)
                )
                if not cursor.fetchone():
                    raise ValueError(f"Video {video_id} not found.")

                if chat_id:
                    cursor.execute(
                        "SELECT id FROM chat_history WHERE id = ?", (chat_id,)
                    )
                    if not cursor.fetchone():
                        raise ValueError(f"Chat message {chat_id} not found.")

                cursor.execute(
                    """
                    INSERT INTO user_feedback
                    (video_id, chat_id, query, response, feedback)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (video_id, chat_id, query, response, feedback),
                )
                conn.commit()
                logger.info(f"Feedback saved for video {video_id}, chat {chat_id}.")
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise

    def get_user_feedback_stats(self, video_id):
        """Get positive/negative feedback counts for a video."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        COUNT(CASE WHEN feedback = 1 THEN 1 END),
                        COUNT(CASE WHEN feedback = -1 THEN 1 END)
                    FROM user_feedback WHERE video_id = ?
                    """,
                    (video_id,),
                )
                return cursor.fetchone() or (0, 0)
        except sqlite3.Error as e:
            logger.error(f"Error getting feedback stats: {e}")
            return (0, 0)

    # --- Embedding & Index Management ---

    def add_embedding_model(self, model_name, description):
        """Register an embedding model."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO embedding_models (model_name, description) VALUES (?, ?)",
                (model_name, description),
            )
            conn.commit()
            return cursor.lastrowid

    def add_elasticsearch_index(self, video_id, index_name, embedding_model_id):
        """Record an Elasticsearch index mapping."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO elasticsearch_indices
                (video_id, index_name, embedding_model_id) VALUES (?, ?, ?)
                """,
                (video_id, index_name, embedding_model_id),
            )
            conn.commit()

    def get_elasticsearch_index(self, video_id, embedding_model):
        """Get the Elasticsearch index name for a video and model."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ei.index_name
                FROM elasticsearch_indices ei
                JOIN embedding_models em ON ei.embedding_model_id = em.id
                JOIN videos v ON ei.video_id = v.id
                WHERE v.youtube_id = ? AND em.model_name = ?
                """,
                (video_id, embedding_model),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_elasticsearch_index_by_youtube_id(self, youtube_id):
        """Get the Elasticsearch index name for a video."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ei.index_name
                FROM elasticsearch_indices ei
                JOIN videos v ON ei.video_id = v.id
                WHERE v.youtube_id = ?
                """,
                (youtube_id,),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    # --- Ground Truth ---

    def add_ground_truth_questions(self, video_id, questions):
        """Save generated ground truth questions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for question in questions:
                try:
                    cursor.execute(
                        "INSERT OR IGNORE INTO ground_truth (video_id, question) VALUES (?, ?)",
                        (video_id, question),
                    )
                except sqlite3.IntegrityError:
                    continue
            conn.commit()

    def get_ground_truth_by_video(self, video_id):
        """Retrieve ground truth questions for a specific video."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT gt.*, v.channel_name
                FROM ground_truth gt
                JOIN videos v ON gt.video_id = v.youtube_id
                WHERE gt.video_id = ?
                ORDER BY gt.generation_date DESC
                """,
                (video_id,),
            )
            return cursor.fetchall()

    def get_ground_truth_by_channel(self, channel_name):
        """Retrieve ground truth questions for a channel."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT gt.*, v.channel_name
                FROM ground_truth gt
                JOIN videos v ON gt.video_id = v.youtube_id
                WHERE v.channel_name = ?
                ORDER BY gt.generation_date DESC
                """,
                (channel_name,),
            )
            return cursor.fetchall()

    def get_all_ground_truth(self):
        """Retrieve all ground truth questions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT gt.*, v.channel_name
                FROM ground_truth gt
                JOIN videos v ON gt.video_id = v.youtube_id
                ORDER BY gt.generation_date DESC
            """)
            return cursor.fetchall()

    # --- Evaluation ---

    def save_search_performance(self, video_id, hit_rate, mrr):
        """Save search performance metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO search_performance (video_id, hit_rate, mrr) VALUES (?, ?, ?)",
                (video_id, hit_rate, mrr),
            )
            conn.commit()

    def save_search_parameters(self, video_id, parameters, score):
        """Save optimized search parameters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for param_name, param_value in parameters.items():
                cursor.execute(
                    """
                    INSERT INTO search_parameters
                    (video_id, parameter_name, parameter_value, score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (video_id, param_name, param_value, score),
                )
            conn.commit()

    def save_rag_evaluation(self, evaluation_data):
        """Save a single RAG evaluation result."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO rag_evaluations
                (video_id, question, answer, relevance, explanation)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    evaluation_data["video_id"],
                    evaluation_data["question"],
                    evaluation_data["answer"],
                    evaluation_data["relevance"],
                    evaluation_data["explanation"],
                ),
            )
            conn.commit()

    def get_latest_evaluation_results(self, video_id=None):
        """Retrieve RAG evaluation results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if video_id:
                cursor.execute(
                    "SELECT * FROM rag_evaluations WHERE video_id = ? ORDER BY evaluation_date DESC",
                    (video_id,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM rag_evaluations ORDER BY evaluation_date DESC"
                )
            return cursor.fetchall()

    def get_latest_search_performance(self, video_id=None):
        """Retrieve search performance metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if video_id:
                cursor.execute(
                    """
                    SELECT * FROM search_performance
                    WHERE video_id = ?
                    ORDER BY evaluation_date DESC LIMIT 1
                    """,
                    (video_id,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM search_performance ORDER BY evaluation_date DESC"
                )
            return cursor.fetchall()
