# # Use the official lightweight Python image
# FROM python:3.9-slim-buster

# # Set the shell environment
# ENV PYENV_SHELL=/bin/bash

# # Ensure we have an up-to-date baseline and install dependencies
# RUN set -ex; \
#     apt-get update && \
#     apt-get install -y --no-install-recommends build-essential git ffmpeg && \
#     pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir pipenv==2023.7.11

# WORKDIR /app

# # Add Pipfile, Pipfile.lock
# ADD Pipfile Pipfile.lock /app/

# RUN pipenv sync

# # Source code
# ADD . /app

# # Install additional Python dependencies for FastAPI and Uvicorn
# RUN pip install --no-cache-dir fastapi uvicorn

# # Expose the port the app will run on
# EXPOSE 5000

# # Entry point for interactive pipenv shell
# ENTRYPOINT ["/bin/bash"]

# # Command to run the FastAPI application
# CMD ["python", "-m", "pipenv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

# # Get into the pipenv shell
# # CMD ["-c", "pipenv shell"]

# Use the same base image as llm_rag
FROM python:3.12-slim-bookworm

ARG DEBIAN_PACKAGES="build-essential git ffmpeg"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell pipenv where the shell is
ENV PYENV_SHELL=/bin/bash

# Tell Python to disable buffering
ENV PYTHONUNBUFFERED=1

# Update and install dependencies
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
    chown app:app /app

# Switch to the new user
USER app
WORKDIR /app

# Install python packages
ADD --chown=app:app Pipfile Pipfile.lock /app/

RUN pipenv sync

# Add the source code
ADD --chown=app:app . /app

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn

# Expose the port
EXPOSE 5000

# Create an entrypoint script
RUN echo '#!/bin/bash\necho "Container is running!!!"\nexec "$@"' > /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh

# Entry point
ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]

# Command to run the FastAPI app
CMD ["python", "-m", "pipenv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]