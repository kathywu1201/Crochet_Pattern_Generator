#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="crochet-app-api-service"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export GCP_PROJECT="crochetai-new" # CHANGE TO YOUR PROJECT ID
export GCS_BUCKET_NAME="crochet-app-models"
export CHROMADB_HOST="crochet-app-vector-db"
export CHROMADB_PORT=8000

# Create the network if we don't have it yet
docker network inspect crochet-app-network >/dev/null 2>&1 || docker network create crochet-app-network

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$PERSISTENT_DIR":/persistent \
-p 9000:9000 \
-e DEV=1 \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/crochetai_new.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e CHROMADB_HOST=$CHROMADB_HOST \
-e CHROMADB_PORT=$CHROMADB_PORT \
--network crochet-app-network \
$IMAGE_NAME