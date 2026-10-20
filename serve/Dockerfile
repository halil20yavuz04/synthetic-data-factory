# Scratch inspection API: ONNX Runtime + FastAPI, no PyTorch (small image).
# Build and run from the repo root:
#   docker build -t scratch-api -f serve/Dockerfile serve
#   docker run --rm -p 8000:8000 scratch-api
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py app.py model.onnx ./
ENV SCRATCH_THRESHOLD=0.59
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
