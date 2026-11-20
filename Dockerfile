FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# face_recognition is installed last, without dependencies, so pip uses the
# prebuilt dlib-bin wheel instead of compiling dlib from source.
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt \
 && pip install face_recognition --no-deps

COPY . .

# Hosts such as Render provide the PORT variable; fall back to 8000 elsewhere.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]