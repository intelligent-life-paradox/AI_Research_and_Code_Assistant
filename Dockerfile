FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# indexa os dados durante o build da imagem
RUN python rag/pipeline_manager.py --mode all

EXPOSE 7860

CMD ["python", "app.py"]