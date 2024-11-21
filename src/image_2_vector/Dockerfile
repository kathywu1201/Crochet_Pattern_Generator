# Use the official lightweight Python image
FROM python:3.9-slim-buster

# Set the shell environment
ENV PYENV_SHELL=/bin/bash

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

# Install additional Python dependencies for FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn

# Expose the port the app will run on
EXPOSE 5000

# Entry point for interactive pipenv shell
ENTRYPOINT ["/bin/bash"]

# Command to run the FastAPI application
CMD ["pipenv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

# Get into the pipenv shell
# CMD ["-c", "pipenv shell"]
