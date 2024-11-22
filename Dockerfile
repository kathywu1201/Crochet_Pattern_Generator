# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GCP_PROJECT="our-shield-436522-f2"
ENV GCS_BUCKET_NAME="dummy-bucket-name"

# Copy the service account key to the container
COPY ../secrets/testing.json /app/testing.json
# data-service-account.json
# llm-service-account-rag.json
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/testing.json"

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock first
COPY Pipfile Pipfile.lock /app/

# Install dependencies from Pipfile.lock
RUN pipenv install --deploy

# Copy the entire project into the container
COPY . /app/

# Set entrypoint to start bash
ENTRYPOINT ["bash"]
# CMD ["pipenv", "run", "pytest", "tests/test_rag.py"]
# PYTHONPATH=./src pipenv run pytest tests/test_pdfProcessor.py

#  PYTHONPATH=./src pipenv run pytest tests/test_rag.py
