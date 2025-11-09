#!/bin/bash

# Variables
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
REGION=eu-west-1
REPO_NAME=spre-news-preproc

# 1. Authenticate Docker with ECR
aws ecr get-login-password --region $REGION | \
docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# 2. Create ECR repo (if not yet created)
aws ecr create-repository --repository-name $REPO_NAME --region $REGION || true

# 3. Build and tag image
docker build -t $REPO_NAME .

# 4. Tag for ECR
docker tag $REPO_NAME:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

# 5. Push image
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest
