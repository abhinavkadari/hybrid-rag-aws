set "account_id=%~1"
set "folder=%~2"
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin %account_id%.dkr.ecr.us-east-1.amazonaws.com
docker build -f Dockerfile.%folder% --provenance=false -t hybrid-rag/%folder% .
docker tag hybrid-rag/%folder%:latest %account_id%.dkr.ecr.us-east-1.amazonaws.com/hybrid-rag/%folder%:latest
docker push %account_id%.dkr.ecr.us-east-1.amazonaws.com/hybrid-rag/%folder%:latest