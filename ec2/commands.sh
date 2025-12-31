ssh -i .ssh/spre-training.pem ubuntu@34.250.200.236

source /opt/tensorflow/bin/activate

pip install pyarrow boto3 botocore numpy

rsync -avz --delete \
  -e "ssh -i ~/.ssh/spre-training.pem" \
  ~/spre/python/ \
  ubuntu@34.250.200.236:~/python/

cd ~/python

python training.py
