# in <name>-layer

docker run -it --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint bash \
  -v "$PWD":/build \
  public.ecr.aws/lambda/python:3.14

# should see bash-5.2#

cd /build
mkdir -p python/lib/python3.14/site-packages

pip install <name> \
  --target python/lib/python3.14/site-packages \
  --no-cache-dir

exit

# optional find python | head -20

zip -r <name>-layer.zip python

aws lambda publish-layer-version \
  --layer-name <name> \
  --zip-file fileb://<name>-layer.zip \
  --compatible-runtimes python3.14 \
  --region eu-west-1

# delete after

sudo rm -rf python
