# Use official Python 3.13 slim base image
FROM python:3.13-slim

# Set working directory
WORKDIR /usr/src/app

# Copy requirements and install dependencies without caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn for production deployment
RUN pip install --no-cache-dir gunicorn

# Copy all application code into the image
COPY . .

# Expose port 5000
EXPOSE 5000

# Define default command to launch API via Gunicorn with 2 workers
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"] 