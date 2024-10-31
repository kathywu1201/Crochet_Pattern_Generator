# Use the official Debian-hosted Python image
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential git screen vim"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set UTF-8 locale for Python
ENV LANG=C.UTF-8

# Define shell for pipenv
ENV PYENV_SHELL=/bin/bash

# Disable Python buffering to ensure log visibility
ENV PYTHONUNBUFFERED=1

# Ensure up-to-date system, install dependencies, create a user
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app && \
    mkdir -p /persistent && \
    chown app:app /persistent && \
    chown app:app /app

# Switch to the new user
USER app
WORKDIR /app

# Copy Pipfile and Pipfile.lock for dependency installation
ADD --chown=app:app Pipfile Pipfile.lock /app/

# Install Python packages with pipenv, ensuring unsloth is included
RUN pipenv sync --dev

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Add the rest of the source code
ADD --chown=app:app . /app

# Define entrypoint
ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]
