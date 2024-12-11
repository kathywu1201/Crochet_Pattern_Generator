# Use the official lightweight Python image
FROM python:3.8-slim-buster

# Set the shell environment
ENV PYENV_SHELL=/bin/bash

# Set the Google service account credentials path inside the container
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/crochetai-438515-7948fa08ef91.json

# Ensure we have an up-to-date baseline and install dependencies
RUN set -ex; \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential git ffmpeg && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv

WORKDIR /app

# Copy your source code and secrets into the container
COPY . /app/
COPY secrets/crochetai-438515-7948fa08ef91.json /secrets/

# Install dependencies directly inside the container using Pipenv
RUN pipenv install google-cloud-vision google-cloud-storage pdfplumber torch transformers Pillow opencv-python-headless

# Set the entry point to bash
ENTRYPOINT ["/bin/bash"]

# Open a pipenv shell on container startup
CMD ["-c", "pipenv shell"]
