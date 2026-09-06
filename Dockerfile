# Base image pinned by digest; Dependabot keeps the tag and digest in sync.
FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for OCR / pdf rendering
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    libgl1 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python deps
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Copy source
COPY . .

# Create non-root user and switch
RUN useradd -m appuser
USER appuser

# Default command: launch the viewer UI
CMD ["python", "pdf_viewer_app.py"]
