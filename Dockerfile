FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose port for HF Spaces
EXPOSE 7860

CMD uvicorn releaseops_arena.server:app --host 0.0.0.0 --port ${PORT:-7860}
