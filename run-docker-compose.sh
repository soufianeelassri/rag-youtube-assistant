#!/bin/bash

# Define the path to the .env file
ENV_PATH="./.env"

# Check if the .env file exists
if [ -f "$ENV_PATH" ]; then
    # Read the .env file and set environment variables
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        if [[ $line =~ ^[[:space:]]*$ ]] || [[ $line =~ ^# ]]; then
            continue
        fi
        
        # Extract variable name and value
        if [[ $line =~ ^([^=]+)=(.*)$ ]]; then
            name="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            export "$name"="$value"
            echo "Loaded environment variable: $name"
        fi
    done < "$ENV_PATH"
    
    # Stop existing containers
    echo "Stopping existing containers..."
    docker-compose down
    
    # Rebuild the container
    echo "Rebuilding Docker containers..."
    docker-compose build --no-cache app
    
    # Start the services
    echo "Starting Docker services..."
    docker-compose up -d
    
    # Wait for services to be ready
    echo "Waiting for services to start up..."
    sleep 20
    
    # Run the Streamlit app
    echo "Starting Streamlit app..."
    docker-compose exec -T app sh -c "cd /app/app && streamlit run main.py"
else
    echo "Error: The .env file was not found at $ENV_PATH" >&2
    exit 1
fi