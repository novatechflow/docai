# Base image pinned by digest; Dependabot keeps the tag and digest in sync.
FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

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
