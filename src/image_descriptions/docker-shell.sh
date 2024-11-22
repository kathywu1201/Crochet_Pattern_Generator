#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

BUILD="True" 

# Read the settings file
source ../env.dev

export IMAGE_NAME="gemini-prompts-generator"

if [ "$BUILD" == "True" ]; then 
    # Build the docker image
    echo "Building the Docker image..."
    docker build -t $IMAGE_NAME -f Dockerfile .

    # Run Container
    echo "Running the Docker container..."
    docker run --rm --name $IMAGE_NAME -ti \
    -v "$BASE_DIR":/app \
    -v "$SECRETS_DIR":/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
    -e GCP_PROJECT=$GCP_PROJECT \
    -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
    $IMAGE_NAME
fi

if [ "$BUILD" != "True" ]; then 
    echo "Using prebuilt image..."
    # Run the container
    echo "Running the Docker container..."
    docker run --rm --name $IMAGE_NAME -ti \
    -v "$BASE_DIR":/app \
    -v "$SECRETS_DIR":/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
    -e GCP_PROJECT=$GCP_PROJECT \
    -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
    $IMAGE_NAME
fi

