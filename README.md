# GStreamer WebRTC application
This repo contains the python application to stream video from the given source via GStreamer pipeline to the WebRTC client with the possibility of controlling the video quality on the fly. AhoyApp streams the source to the AhoyMedia WebRTC client maintained by ADDIX GmbH. The application is based on the [GStreamer](https://gstreamer.freedesktop.org/) framework and uses its python bindings.

AhoyApp uses a low-level webrtcbin plugin for streaming, which is a part of GStreamer WebRTC stack. It also provides the custom signaling to connect to AhoyMedia engine and to handle the WebRTC streaming properly. It requires GStreamer >= 1.22 as well as its python GI bindings and it requires aiortc library for the WebRTC connection with AhoyMedia itself. It also heavily uses asyncio with uvloop backend to handle the asynchronous nature of the main coroutines.

## Features
The main features of the application are:
* Stream video through the given GStreamer pipeline to AhoyMedia in a fast manner without any extra steps. One needs a pipeline in a string format, a video source (like rtsp://...) and Ahoy address/api key parameters to stream.
* Support stop/resume streaming with efficient resource re-allocation and self-restarting without any additional calls.
* Flexibly configure the GStreamer pipeline elements. By default, the pipeline is tuned towards the lowest latency.
* Control resolution, framerate and bitrate of the video on the fly via setters provided by the application.
* Control video encoder and RTP payloader parameters of the pipeline.
* Receive WebRTCBin statistics from the viewer's browser.


## Installation
### Docker
The preferred way to install it for a quick test is to use Docker. Go to the docker folder and build the image with:
```bash
docker build -f Dockerfile -t gstwebrtcapp:latest .
```
Then run the container with:
```bash
docker run -it --name gstwebrtcapp_c0 --network=host --cap-add=NET_ADMIN {..} gstwebrtcapp:latest bash
```
where {..} are the display options. On Linux:
```bash
-e DISPLAY=$YOUR_IPV4_ADDRESS:0 -v /tmp/.X11-unix:/tmp/.X11-unix
```
on Windows:
```bash
-e DISPLAY=host.docker.internal:0.0
```

#### Bug with gstreamer python bindings
It could be that gi bindings are installed by a system in the wrong place so that they are invisible for pyenv. You may check it in VSCode. Then find and copy them like this (for 3.11.5 version in Dockerfile):
```bash
cp -r /usr/lib/python3/dist-packages/gi/ /root/.pyenv/versions/3.11.5/lib/python3.11/site-packages/
```

### Manual
To install manually, you need to do all the instructions written in a Dockerfile, namely:
1. Install sources for video codecs.
2. Install GStreamer and its plugins with version >= 1.22. This Dockefile serves ubuntu 23.04 for this, where you can find the dependencies in the apt repo.
3. Install python >= 3.11 and cpython bindings for gstreamer:
```bash
apt-get install python3-gst-1.0 python-gi-dev python3-gi gstreamer1.0-python3-plugin-loader
```
4. Install aiortc python library.

## Usage
### Run
To run the application, you need to specify the configuration of the GStreamer application and connection parameters. The default tuned pipeline together with the main parameters is provided in app.py. For the connection parameters, one needs to specify the server address and the API key among other optional parameters.

Next, in the main endpoint, one needs to await the two coroutines as provided in `example.py` file. The first `connect_coro` controls the connection and communication with AhoyMedia and the second `webrtc_coro` controls all GStreamer -> WebRTC steps via its sub-coroutines aka handlers. The main.py file contains an example of how to run the application. Note that it won't work unless the valid AhoyDirector URL, API key and video source are specified instead of the dummy strings. 

If one stops streaming in AhoyDirector window, then `webrtc_coro` will turn into a pending state and will wait for the next play event to resume streaming. The pipeline will be stopped and set in a NULL state together with WebRTC stack and restarted on the next play event so that when the video is not played, the resources for GStreamer C objects are properly deallocated. In the meantime, the `connect_coro` coroutine will be still running to keep the connection with AhoyMedia.

To add a new internal handler (e.g., for some specific statistics), one needs to add a new sub-coroutine to the main block of the `webrtc_coro` coroutine. To add a new external handler (e.g., to control the video quality), one needs to add a new coroutine to the main execution as demonstrated in the `example.py` file by a dummy `test_manipulate_video` endpoint.

## License
This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Author
M.Sc. Nikita Smirnov, Intelligent Systems AG, Department of Computer Science, Kiel University, Germany.
Please contact me in case of any questions or bug reports: [Nikita Smirnov](mailto:nsm@informatik.uni-kiel.de)




