# Dockerfile for the API running back end
FROM python:3.11-slim

WORKDIR /app

# Download dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend file structure

COPY backend/ ./backend/


#Specify the port
EXPOSE 8000

# Command for running the API

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
