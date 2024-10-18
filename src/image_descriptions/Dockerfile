# Use the official Debian-hosted Python image
FROM python:3.9-slim-buster

ARG DEBIAN_PACKAGES="build-essential git curl"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

# Tell pipenv where the shell is.
ENV PYENV_SHELL=/bin/bash

# Set Google Cloud credentials environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/crochet_prompts_generate.json

# Ensure we have an up-to-date baseline and install dependencies
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES apt-transport-https ca-certificates gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    pip install --no-cache-dir --upgrade pip pipenv && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app /persistent && chown app:app /app /persistent

# Switch to the new user
USER app
WORKDIR /app

# Install Python packages
ADD --chown=app:app Pipfile Pipfile.lock /app/
RUN pipenv sync

# Add the rest of the source code
ADD --chown=app:app . /app

# Entry point
ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]




