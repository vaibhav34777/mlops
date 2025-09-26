FROM python:3.11-slim

WORKDIR /batch

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip && pip install -r requirements.txt

COPY batch.py .

ENTRYPOINT ["python", "batch.py"]