#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set variables
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="crochetai-new"
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/crochetai_new.json"
export IMAGE_NAME="crochet-app-vector-db-cli"

# Create the network if we don't have it yet
docker network inspect crochet-app-network >/dev/null 2>&1 || docker network create crochet-app-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports $IMAGE_NAME