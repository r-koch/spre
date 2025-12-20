# Base image — slim, Lambda-compatible
FROM public.ecr.aws/lambda/python:3.14

# Set Hugging Face home
ENV HF_HOME=/tmp/hf_home

# Install minimal system dependencies
RUN dnf install -y git && dnf clean all

# Upgrade pip, setuptools, wheel to avoid build issues
RUN pip install --upgrade pip setuptools wheel

# Copy application files
WORKDIR ${LAMBDA_TASK_ROOT}
COPY python/news_preproc.py .
COPY python/shared.py .
COPY docker/news-preproc/requirements.txt .

# Install Python dependencies (with pinned versions)
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

# download the model and move to a stable folder
RUN mkdir -p /tmp/finbert_model && \
    mkdir -p ${LAMBDA_TASK_ROOT}/finbert_model && \
    python - <<'PY'
from huggingface_hub import snapshot_download
import os, shutil

CACHE_DIR = "/tmp/finbert_model"
TARGET_DIR = f"{os.environ['LAMBDA_TASK_ROOT']}/finbert_model"

repo_path = snapshot_download("ProsusAI/finbert", cache_dir=CACHE_DIR)

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
shutil.copytree(repo_path, TARGET_DIR)
shutil.rmtree(CACHE_DIR, ignore_errors=True)
PY

ENV MODEL_DIR=${LAMBDA_TASK_ROOT}/finbert_model

# Set Lambda handler
CMD ["news_preproc.lambda_handler"]
