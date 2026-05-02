FROM python:3.11-slim

WORKDIR /app

# Install git for repo cloning
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy application code
COPY app/ app/
COPY build_index.py .
COPY templates/ templates/
COPY static/ static/

# Copy pre-built index (if available)
COPY Vector_Store/ Vector_Store/

# Copy data directory (docs will be embedded at build time)
COPY data/ data/

# Environment
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Start the server
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
