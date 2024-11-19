set -e

# Set variables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="crochetai-438515" # CHANGE TO YOUR PROJECT ID
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/crochet_prompts_generate.json"
export IMAGE_NAME="seleniarm/standalone-chromium:latest"
export GCS_BUCKET_NAME="crochet-patterns-bucket"

# Pull the latest image from Docker Hub
docker pull $IMAGE_NAME

# Run the container as root, install Python, pip3, create a virtual environment, and install Selenium
docker run --rm -ti \
    --user root \
    --name chromium-container \
    -p 4444:4444 \
    -p 5900:5900 \
    -p 7900:7900 \
    -v "$BASE_DIR":/app \
    -v "$SECRETS_DIR":/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
    -e GCP_PROJECT=$GCP_PROJECT \
    -e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
    $IMAGE_NAME /bin/bash -c "\
      apt-get update && \
      apt-get install -y python3 python3-pip python3-venv && \
      python3 -m venv /app/venv && \
      source /app/venv/bin/activate && \
      pip install selenium && \
      pip install requests && \
      pip install google-cloud-storage && \ 
      cd /app && \
      bash"