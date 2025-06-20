#!/bin/bash
container_name="MASTER"
image_name="master"
script_path=$(realpath "$0")
docker_path=$(dirname "$script_path")
workspace_path=$(dirname "$docker_path")
echo "Script path: $script_path"
echo "Workspace path: $workspace_path"

# Check if the image is built
if [ -z "$(docker images -q $image_name)" ]; then
    echo "Image $image_name not found. Building the image..."
    docker build -t $image_name $docker_path
    # docker build -t "$image_name" -f "$docker_path/Dockerfile_deploy" "$docker_path"
else
    echo "Image $image_name found."
fi

if [ -z "$(docker ps -a -q -f name=$container_name)" ]; then
    echo 'container not exist'
    docker run -it -d --init \
        --name $container_name \
        --network host \
        --privileged \
        -v "$workspace_path":"/MASTER" \
        -w /MASTER \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /dev:/dev \
        -e DISPLAY=$DISPLAY \
        --gpus all \
        $image_name
else
    echo 'container exist'
    docker start $container_name
fi

# Unset variables
unset script_path
unset docker_path
unset workspace_path
unset container_name
unset image_name
