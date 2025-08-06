FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=180 -r requirements.txt

# Pre-download models
COPY download_models.py .
RUN python download_models.py

COPY . .

RUN apt-get update && apt-get install -y curl

EXPOSE 5001

CMD ["python", "app.py"]