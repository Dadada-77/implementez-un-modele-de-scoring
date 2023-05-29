# backend/Dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements_backend.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--timeout", "10000"]
