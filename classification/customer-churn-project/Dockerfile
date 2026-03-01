# Base image - use official Python slim (smaller than full)
# "slim" = no extra system packages, keeps image small 
FROM python:3.11-slim

# who maintains this
LABEL maintainer="hiruniliyanage4@gmail.com"

# Set working directory inside the container
# All subsequent commands run from/app
WORKDIR /app

# Copy requirements FIRST(before source code)
# Docker caches layers - if requirements don't change
# it won't reinstall packages on every buil (big speed)
COPY requirements.txt .


# Install python dependencies
# -- no-caches dir keeps image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Now copy source code (after package - for cache efficiency)
COPY src/ ./src
COPY configs/ ./configs/

# Create output directories inside container
RUN mkdir -p /app/models/artifacts /app/data/processed

# Default command when container starts
# We'll overrride this per use case
CMD ["python, "src/train.py"]

