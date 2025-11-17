FROM python:3.10-slim

# Create a non‑root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    mkdir -p /home/appuser/app

WORKDIR /home/appuser/app
COPY . /home/appuser/app

# Install system dependencies for matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libfreetype6-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]