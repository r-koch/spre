# Base image — slim, Lambda-compatible
FROM public.ecr.aws/lambda/python:3.11

# Install minimal system dependencies
RUN yum install -y git && yum clean all

# Upgrade pip, setuptools, wheel to avoid build issues
RUN pip install --upgrade pip setuptools wheel

# Copy application files
WORKDIR ${LAMBDA_TASK_ROOT}
COPY app/* .
COPY requirements.txt .

# Install Python dependencies (with pinned versions)
RUN pip install --no-cache-dir -r requirements.txt

# Set Lambda handler
CMD ["main.lambda_handler"]
