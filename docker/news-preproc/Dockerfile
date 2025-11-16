# Base image — slim, Lambda-compatible
FROM public.ecr.aws/lambda/python:3.11

# Set Hugging Face home
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache

# Install minimal system dependencies
RUN yum install -y git && yum clean all

# Upgrade pip, setuptools, wheel to avoid build issues
RUN pip install --upgrade pip setuptools wheel

# Copy application files
COPY app/* ${LAMBDA_TASK_ROOT}/
COPY requirements.txt .

# Install Python dependencies (with pinned versions)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

# download the model into /opt/finbert_model and move to a stable folder
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
CMD ["preproc.lambda_handler"]
