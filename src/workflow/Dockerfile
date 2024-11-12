# Use the official Debian-based Python image
FROM python:3.9-slim-buster

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYENV_SHELL=/bin/bash

# Install necessary packages and Google Cloud SDK
RUN set -ex; \
    # Create man directories to avoid warnings
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ffmpeg \
        apt-transport-https \
        ca-certificates \
        gnupg \
        screen \
        vim && \
    # Install Google Cloud SDK
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
        tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    # Install pip and pipenv
    pip install --no-cache-dir --upgrade pip pipenv && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
RUN useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app /persistent && \
    chown app:app /app /persistent

# Switch to the new user
USER app
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY --chown=app:app Pipfile Pipfile.lock /app/

# Install Python dependencies
RUN pipenv sync

# Copy the rest of the application code
COPY --chown=app:app . /app

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]
