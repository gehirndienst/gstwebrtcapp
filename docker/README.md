# Docker installation
To overcome the installation and dependency hell, please deploy the Docker environment. You can use the Dockerfile-cpu (cpu version) or Dockerfile-cuda (cuda version with nv* plugins for hardware acceleration) to build the image. The Dockerfile-cuda builds the image with CUDA 12.1 (so far the version is fixed) support. You have two options of how to build and run.

## Compose
Install `docker-compose-plugin` plugin (>=v2) if not installed. Then go to the `docker` folder use cuda/cpu profiles to build and run the image with docker compose. E.g., for CUDA:
```bash
docker compose --profile cuda build
```
and for cpu:
```bash
docker compose --profile cpu build
```
Then run the container with:
```bash
docker compose --profile cuda up -d
```

## Manual
Go to the `docker` folder and build the CUDA image with:
```bash
docker build -f Dockerfile-cuda -t nsmirnov/gstwebrtcapp:latest .
```
To build the image without CUDA support, use:
```bash
docker build -f Dockerfile-cpu -t nsmirnov/gstwebrtcapp-cpu:latest .
```
Then run the container with:
```bash
docker run -d --name gstwebrtcapp-container --network=bridge -P --privileged --cap-add=NET_ADMIN {..} nsmirnov/gstwebrtcap-cpu:latest bash
```
to run the CUDA container, use:
```bash 
docker run --gpus all -d --name gstwebrtcapp-container --network=bridge -P --privileged --cap-add=NET_ADMIN {..} nsmirnov/gstwebrtcapp:latest bash
```	
where {..} are the OPTIONAL display options that could be skipped. On Linux:
```bash
-e DISPLAY=$YOUR_IPV4_ADDRESS:0 -v /tmp/.X11-unix:/tmp/.X11-unix
```
on Windows:
```bash
-e DISPLAY=host.docker.internal:0.0
```