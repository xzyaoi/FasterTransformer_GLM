#!/bin/bash

TARGET=$1
COMMIT=$(git show --format="%H" --no-patch)
COMMIT_AUTHOR=$(git show --format="%an" --no-patch)
COMMIT_TIME=$(git show --format="%cI" --no-patch)
echo "$COMMIT" > COMMIT_INFO
echo "$COMMIT_AUTHOR" >> COMMIT_INFO
echo "$COMMIT_TIME" >> COMMIT_INFO

python3 ./docker/check_cuda.py > SM_NUMBER

if [ "$1" = "debug" ]; then
    echo "Build in DEBUG mode with git files"
    echo "RUN apt install -y vim git" >> ./docker/Dockerfile
fi
echo "Building CUDA Docker Image with tag ftglm:latest"
docker build -f ./docker/Dockerfile -t ftglm .