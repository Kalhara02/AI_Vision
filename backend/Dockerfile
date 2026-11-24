FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf TORCH_HOME=/data/torch
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU build by default; the GPU compose file passes a CUDA index instead
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install -r requirements.txt

# SAM 2 (Meta, Apache-2.0), official package, pure-Python build
RUN SAM2_BUILD_CUDA=0 pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/sam2.git"

COPY app ./app
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
