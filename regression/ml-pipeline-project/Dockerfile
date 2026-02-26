# Base image - use official Python slim (smaller than full)
# "slim" = no extra system packages, keeps image small
FROM python:3.11-slim

# Who maintains this (optionl but good practice)
LABEL  maintainer = "hiruniliyanage4@gamil.com"

# Set working directory inside the container
# All subsequent commans run from/app
WORKDIR /app


# Copy requirements FIRST (before source code)
# Docker cahes layers - if requirements don't change,
# it won't reinstall packages packages on every build (big speedup)
COPY requirements.txt .

# Install python dependencies
# -- no-cahces dir keeps image smaller 
RUN pip install --no-cache-dir -r requirements.txt

# Now copy source code (after package - for cache efficiency)
COPY src/ ./src
COPY configs/ ./configs/

# Create output directoiries inside container
RUN mkdir -p /app/models/artifacts /app/data/processed 


# Default command when container starts 
# We'll override this per use case
CMD ["python", "src/train.py"]

