# Use the official lightweight Python image
FROM python:3.9-slim-buster

# Set the shell environment
ENV PYENV_SHELL=/bin/bash

# # Set the Google service account credentials path inside the container
# ENV GOOGLE_APPLICATION_CREDENTIALS=../secrets/crochet_secret.json

# Ensure we have an up-to-date baseline and install dependencies
RUN set -ex; \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential git ffmpeg && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv

WORKDIR /app

# Add Pipfile, Pipfile.lock
ADD Pipfile Pipfile.lock /app/

RUN pipenv sync

# Source code
ADD . /app

# Entry point
ENTRYPOINT ["/bin/bash"]

# Get into the pipenv shell
CMD ["-c", "pipenv shell"]
