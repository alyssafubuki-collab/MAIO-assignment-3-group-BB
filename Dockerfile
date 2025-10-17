FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
