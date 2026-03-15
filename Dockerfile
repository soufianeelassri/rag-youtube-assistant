# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p app/pages config data grafana logs /root/.streamlit

# Set Python path and Streamlit configs
ENV PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create empty __init__.py files
RUN touch app/__init__.py app/pages/__init__.py

# Copy the application code and other files into the container
COPY app/ ./app/
COPY config/ ./config/
COPY data/ ./data/
COPY grafana/ ./grafana/
COPY .env ./
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create a healthcheck script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8501/_stcore/health' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["/healthcheck.sh"]

# Run Streamlit
CMD ["streamlit", "run", "app/home.py", "--server.port=8501", "--server.address=0.0.0.0"]