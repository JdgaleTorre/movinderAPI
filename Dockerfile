# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HF_HOME=/tmp/huggingface
ENV HF_HUB_CACHE=/tmp/huggingface/hub
ENV HF_MODULES_CACHE=/tmp/huggingface/modules
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers



# Create a working directory
WORKDIR /app

# Install system dependencies
# (build-essential + libpq-dev for psycopg2 + curl)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy requirement files first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Collect static files (optional)
RUN python manage.py collectstatic --noinput || true

# Expose port 7860 (Hugging Face Spaces default) or 8000
EXPOSE 7860

# Run Django with gunicorn on port 7860
CMD gunicorn movinderAPI.wsgi:application --bind 0.0.0.0:7860 --workers 2
