FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/

RUN mkdir -p data ml/artifacts ml/plots

CMD ["sh", "-c", "python ml/generate_data.py && python ml/train.py"]
