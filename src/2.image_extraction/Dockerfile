# base image
FROM python:3.11-slim-buster


# up to date baseline and OS dependencies
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app
    
# add user app
#RUN useradd -m app

# set working directory to app
# USER app
WORKDIR /app

COPY . /app/

# install necessary packages
RUN pipenv --python 3.11 && \
    pipenv install numpy scikit-learn

# running pipenv sync
RUN pipenv sync

# set entry point
ENTRYPOINT ["/bin/bash"]

# get into the pipenv shell
CMD ["-c", "pipenv shell"]