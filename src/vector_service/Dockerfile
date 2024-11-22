# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the application code to the container
COPY app.py .

# Install necessary dependencies
RUN pip install --no-cache-dir fastapi uvicorn numpy

# Expose the port the app will run on
EXPOSE 5050

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"]
