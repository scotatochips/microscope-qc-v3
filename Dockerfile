# Dockerfile for HuggingFace Spaces deployment
# Spaces runs as user 1000 by default and expects port 7860.

FROM python:3.10-slim

# System deps required by OpenCV and Cellpose
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces requires non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install Python deps as user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Cellpose cyto3 model so first request isn't slow
RUN python -c "from cellpose import models; models.CellposeModel(gpu=False, pretrained_model='cyto3')"

# Copy application code
COPY --chown=user . .

EXPOSE 7860

CMD ["python", "server.py"]
