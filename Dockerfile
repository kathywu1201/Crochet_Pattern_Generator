# Use an official Python runtime as a base image
# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GCP_PROJECT="our-shield-436522-f2"
ENV GCS_BUCKET_NAME="dummy-bucket-name"
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/testing.json"

# Set the working directory
WORKDIR /app

# Copy the service account key to the container
# Consider passing secrets via environment variables or Docker secrets for better security
COPY ../secrets/testing.json /app/testing.json

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy the application code into the container
COPY . /app/

# Default entrypoint to allow interactive sessions
ENTRYPOINT ["bash"]

# CMD ["pipenv", "run", "pytest", "tests/test_rag.py"]
# PYTHONPATH=./src pipenv run pytest tests/test_pdfProcessor.py

#  PYTHONPATH=./src pipenv run pytest tests/test_rag.py
