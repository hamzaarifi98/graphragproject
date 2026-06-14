FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

COPY requirements-frontend.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-compile -r requirements-frontend.txt

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py"]
