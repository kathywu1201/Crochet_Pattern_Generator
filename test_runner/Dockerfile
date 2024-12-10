# Use Python as the base image
FROM python:3.9

# Install pipenv
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy only the Pipfile and Pipfile.lock first to leverage Docker layer caching for dependencies
COPY Pipfile Pipfile.lock /app/

# Install dependencies from Pipfile
RUN pipenv install 

# Copy the entire project into the container
COPY . /app