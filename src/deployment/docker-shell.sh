#!/bin/bash

# exit immediately if a command exits with a non-zero status
#set -e

# Define some environment variables
export IMAGE_NAME="crochet-app-deployment"
export BASE_DIR=$(pwd)
# export SECRETS_DIR=$(pwd)/../../../secrets/
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="crochetai-new" # Change to your GCP Project
export GCP_ZONE="us-central1-a"
export GOOGLE_APPLICATION_CREDENTIALS=/../secrets/deployment.json
export GCS_BUCKET_NAME="crochet-patterns-bucket-v2"
export GCS_SERVICE_ACCOUNT="crochet-service@crochetai-new.iam.gserviceaccount.com"
export GCS_PACKAGE_URI="gs://crochet-app-trainer-code"

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$HOME/.ssh":/home/app/.ssh \
-v "$BASE_DIR/../api-service":/api-service \
-v "$BASE_DIR/../frontend-react":/frontend-react \
-v "$BASE_DIR/../frontend-react":/pdf_processor \
-v "$BASE_DIR/../frontend-react":/image_descriptions \
-v "$BASE_DIR/../vector-db":/vector-db \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e USE_GKE_GCLOUD_AUTH_PLUGIN=True \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCP_ZONE=$GCP_ZONE \
-e GCP_REGION=$GCP_REGION \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \
-e WANDB_KEY=$WANDB_KEY \
$IMAGE_NAME

