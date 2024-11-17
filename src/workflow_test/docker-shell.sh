#!/bin/bash

# Load environment variables from env.dev
source ../env.dev

# Define the Docker image name and container name
export IMAGE_NAME="crochet-app-workflow"
export CONTAINER_NAME="${IMAGE_NAME}-$(date +%s)"

# Resolve the base directory to an absolute path
export BASE_DIR=$(cd "$(dirname "$0")" && pwd)

# Correctly set the secrets directory path
if [ -d "$BASE_DIR/../secrets" ]; then
    export SECRETS_DIR=$(cd "$BASE_DIR/../secrets" && pwd)
else
    echo "Error: Secrets directory not found at $BASE_DIR/../secrets"
    exit 1
fi

# Debugging output
echo "Secrets Directory: $SECRETS_DIR"
echo "Base Directory: $BASE_DIR"
echo "Google Credentials: $GOOGLE_APPLICATION_CREDENTIALS"

# Build the Docker image based on the Dockerfile
docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run the Docker container
docker run --rm --name $CONTAINER_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$BASE_DIR/../src/image_2_vector":/image_2_vector \
-v "$BASE_DIR/../src/llm-rag":/llm-rag \
-v "$BASE_DIR/../src/llm-finetuning":/llm-finetuning \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCP_SERVICE_ACCOUNT=$GCP_SERVICE_ACCOUNT \
-e LOCATION=$LOCATION \
$IMAGE_NAME
