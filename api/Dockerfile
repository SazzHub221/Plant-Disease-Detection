FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model directory and other files
COPY . .

# Make directory for models
RUN mkdir -p saved_models

# Expose port
EXPOSE 8080

# Run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]