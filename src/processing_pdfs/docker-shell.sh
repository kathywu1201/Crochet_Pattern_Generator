#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="processing_pdf"
export BASE_DIR=$(pwd)

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
echo "Building the Docker image..."
docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run the container
echo "Running the Docker container..."
docker run --rm --name $IMAGE_NAME -ti $IMAGE_NAME