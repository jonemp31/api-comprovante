FROM python:3.12-slim

# Install tesseract + Portuguese language pack
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-por \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Variáveis de validação (valores padrão — sobrescrever via docker-compose/stack)
ENV EXPECTED_NOME_RECEBEDOR="Maria Eduarda Cavaton"
ENV EXPECTED_CPF_RECEBEDOR_PARTIAL="824.458"
ENV EXPECTED_INSTITUICAO_RECEBEDOR="Banco Inter"
ENV EXPECTED_CHAVE_PIX="16991500219"
ENV MAX_HORAS_VALIDADE="24"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
