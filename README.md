# RAG YouTube Assistant

A Retrieval-Augmented Generation (RAG) application that lets you chat with YouTube video transcripts. Ask questions about any YouTube video and get accurate, context-aware answers powered by local LLMs.

## Features

- **Video Ingestion** - Process single videos or entire channels via YouTube Data API v3
- **Interactive Chat** - Ask questions about video content with context-aware responses
- **Multiple Search Strategies** - Hybrid (vector + keyword), text-only, or embedding-only search
- **Query Rewriting** - Chain-of-Thought and ReAct query optimization methods
- **Ground Truth Generation** - Auto-generate test questions from transcripts for evaluation
- **RAG Evaluation** - Measure system performance with Hit Rate, MRR, and LLM-as-Judge metrics
- **User Feedback** - Thumbs up/down feedback loop for continuous improvement
- **Monitoring** - Grafana dashboards for metrics, feedback, and search performance

## Architecture

```
YouTube Video URL
    |
    v
Transcript Extraction (YouTube Data API v3)
    |
    v
Text Processing & Embedding (Sentence Transformers)
    |
    +---> Elasticsearch (vector + full-text index)
    +---> SQLite (metadata + transcripts)
    |
    v
RAG Query Pipeline
    |
    +---> Query Rewriting (optional: CoT / ReAct)
    +---> Hybrid Search (KNN + BM25 + RRF fusion)
    +---> LLM Response Generation (Ollama)
    |
    v
Streamlit UI with Feedback & Evaluation
```

## Tech Stack

| Component        | Technology                |
|------------------|---------------------------|
| UI               | Streamlit                 |
| LLM              | Ollama (local inference)  |
| Embeddings       | Sentence Transformers     |
| Vector Store     | Elasticsearch 8.9         |
| Metadata Store   | SQLite                    |
| Monitoring       | Grafana                   |
| Containerization | Docker & Docker Compose   |

## Project Structure

```
rag-youtube-assistant/
├── app/
│   ├── home.py                  # Main entry point (Streamlit)
│   ├── data_processor.py        # Text processing, embeddings, search
│   ├── transcript_extractor.py  # YouTube API integration
│   ├── database.py              # SQLite database handler
│   ├── rag.py                   # RAG pipeline & LLM interaction
│   ├── query_rewriter.py        # Query optimization (CoT, ReAct)
│   ├── evaluation.py            # Evaluation metrics & LLM-as-Judge
│   ├── generate_ground_truth.py # Ground truth question generation
│   ├── elasticsearch_handler.py # Elasticsearch utilities
│   ├── minsearch.py             # Lightweight TF-IDF search index
│   ├── utils.py                 # Video processing helpers
│   └── pages/
│       ├── data_ingestion.py    # Data ingestion UI
│       ├── chat_interface.py    # Chat UI with feedback
│       ├── ground_truth.py      # Ground truth management UI
│       └── evaluation.py        # Evaluation dashboard UI
├── config/
│   └── config.yaml
├── data/                        # SQLite DB & generated CSVs
├── grafana/                     # Grafana provisioning & dashboards
├── .streamlit/config.toml       # Streamlit theme & settings
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── .env_template
└── run-docker-compose.sh        # Startup script (Linux/Mac)
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with WSL2 on Windows)
- [YouTube Data API v3 key](https://console.cloud.google.com/apis/credentials)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/ganesh3/rag-youtube-assistant.git
   cd rag-youtube-assistant
   ```

2. **Configure environment variables**
   ```bash
   cp .env_template .env
   # Edit .env and add your YOUTUBE_API_KEY
   ```

3. **Build and start services**
   ```bash
   docker-compose build app
   docker-compose up -d
   ```

4. **Open the application**

   Navigate to [http://localhost:8501](http://localhost:8501)

## Environment Variables

| Variable             | Description                     | Default              |
|----------------------|---------------------------------|----------------------|
| `YOUTUBE_API_KEY`    | YouTube Data API v3 key         | *(required)*         |
| `HF_TOKEN`          | Hugging Face token (optional)   | -                    |
| `OLLAMA_MODEL`       | Ollama model to use             | `phi3`               |
| `OLLAMA_HOST`        | Ollama service URL              | `http://ollama:11434`|
| `OLLAMA_TIMEOUT`     | Request timeout (seconds)       | `240`                |
| `OLLAMA_MAX_RETRIES` | Max retry attempts              | `3`                  |

## Evaluation Results

| Metric   | Score |
|----------|-------|
| Hit Rate | 1.0   |
| MRR      | 1.0   |
| RELEVANT | 100%  |

## Monitoring

Grafana is available at [http://localhost:3000](http://localhost:3000) (default credentials: `admin`/`admin`).

Dashboards include:
- User feedback tracking
- Search performance metrics
- RAG evaluation results

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
