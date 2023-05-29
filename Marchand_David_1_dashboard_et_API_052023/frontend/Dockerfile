# frontend/Dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements_frontend.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "frontend.py"]
