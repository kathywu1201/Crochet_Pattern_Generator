#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set vairables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="crochetai-438515" # CHANGE TO YOUR PROJECT ID
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/crochetai-438515-63ef835b2ccc.json"
export IMAGE_NAME="data-collecting"

# Create the network if we don't have it yet
# docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network
docker network inspect data-collecting-network >/dev/null 2>&1 || docker network create data-collecting-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports $IMAGE_NAME






# # Build the image based on the Dockerfile
# docker build -t $IMAGE_NAME -f Dockerfile .

# # Run Container
# docker run --rm --name $IMAGE_NAME -ti \
# -v "$BASE_DIR":/app \
# -v "$SECRETS_DIR":/secrets \
# -v "$PERSISTENT_DIR":/persistent \
# -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
# -e GCP_PROJECT=$GCP_PROJECT \
# $IMAGE_NAME