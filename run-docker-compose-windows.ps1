# Define the path to the .env file
$envPath = ".\.env"

# Check if the .env file exists
if (Test-Path $envPath) {
    # Read the .env file
    $envContent = Get-Content $envPath
    # Parse the environment variables
    foreach ($line in $envContent) {
        if ($line -match '^([^=]+)=(.*)$') {
            $name = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "Loaded environment variable: $name"
        }
    }
    
    # Stop existing containers
    Write-Host "Stopping existing containers..."
    docker-compose down

    # Rebuild the container
    Write-Host "Rebuilding Docker containers..."
    docker-compose build --no-cache app

    # Start the services
    Write-Host "Starting Docker services..."
    docker-compose up -d

    # Wait for services to be ready
    Write-Host "Waiting for services to start up..."
    Start-Sleep -Seconds 20

    # Run the Streamlit app
    Write-Host "Starting Streamlit app..."
    docker-compose exec -T app sh -c "cd /app/app && streamlit run main.py"
}
else {
    Write-Error "The .env file was not found at $envPath"
}