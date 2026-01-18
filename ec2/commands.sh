# in wsl window 1 - connect to instance
ssh -i .ssh/spre-training.pem ubuntu@18.201.12.105

source /opt/tensorflow/bin/activate

pip install pyarrow boto3 botocore numpy

# in wsl window 2 - copy code
rsync -avz --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  -e "ssh -i ~/.ssh/spre-training.pem" \
  ~/spre/python/ \
  ubuntu@18.201.12.105:~/python/

# in wsl window 1 - start training
cd ~/python

python training.py
