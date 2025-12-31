# Base image — slim, Lambda-compatible
FROM public.ecr.aws/lambda/python:3.13

# Upgrade pip, setuptools, wheel to avoid build issues
RUN pip install --upgrade pip setuptools wheel

# Copy application files
WORKDIR ${LAMBDA_TASK_ROOT}
COPY python/inference.py .
COPY python/shared.py .
COPY python/shared_model.py .
COPY docker/inference/requirements.txt .

# Install Python dependencies (with pinned versions)
RUN pip install --no-cache-dir -r requirements.txt

# Set Lambda handler
CMD ["inference.lambda_handler"]
