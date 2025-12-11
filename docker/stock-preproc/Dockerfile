# Base image — slim, Lambda-compatible
FROM public.ecr.aws/lambda/python:3.14

# Install minimal system dependencies
RUN dnf install -y git && dnf clean all

# Upgrade pip, setuptools, wheel to avoid build issues
RUN pip install --upgrade pip setuptools wheel

# Copy application files
WORKDIR ${LAMBDA_TASK_ROOT}
COPY python/stock_preproc.py .
COPY python/shared.py .
COPY docker/stock-preproc/requirements.txt .

# Install Python dependencies (with pinned versions)
RUN pip install --no-cache-dir -r requirements.txt

# Set Lambda handler
CMD ["stock_preproc.lambda_handler"]
