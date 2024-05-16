# GStreamer WebRTC application
This repo contains the python application to stream video from the given source via GStreamer pipeline to the WebRTC client with the possibility of controlling the video quality on the fly. AhoyApp streams the source to the AhoyMedia WebRTC client maintained by ADDIX GmbH. SinkApp streams the source to the JS client in Chrome browser. Both applications are based on the [GStreamer](https://gstreamer.freedesktop.org/) framework and uses its python bindings.

AhoyApp uses a low-level webrtcbin plugin for streaming, which is a part of GStreamer WebRTC stack. It also provides the custom signaling to connect to AhoyMedia engine and to handle the WebRTC streaming properly. SinkApp uses a webrtcsink plugin and connects to the provided JS client that one need to run together with the signalling server for webrtcsink. They both require GStreamer >= 1.22 as well as its python GI bindings and ApoyApp also needs aiortc library for the WebRTC connection with AhoyMedia itself. They also heavily use asyncio with uvloop backend to handle the asynchronous nature of the main coroutines.

## Features
The main features of the applications are:
* Stream video through the given GStreamer pipeline (AhoyApp) to AhoyMedia in a fast manner without any extra steps. One needs a pipeline in a string format, a video source (like rtsp://...) and Ahoy address/api key parameters to stream.
* Stream video through the given GStreamer pipeline (SinkApp) to an independent GstWebRTC API (WebRTC JS client in browser), a webpack service run on the localhost. 
* Support stop/resume streaming with efficient resource re-allocation and self-restarting without any additional calls.
* Flexibly configure the GStreamer pipeline elements. By default, the pipeline is tuned towards the lowest latency.
* Control resolution, framerate and bitrate of the video on the fly via setters provided by the application.
* Control video encoder and RTP payloader parameters of the pipeline. Supports H264, H265, VP8 and VP9 codecs with their respective hardware acceleration (NVENC).
* Receive WebRTC statistics from the viewer's browser via GStreamer internal callbacks and provided agents (control API) for processing them.
* Deploy internal or external MQTT broker for the communication between agents or different parts of the application as well as for publishing the statistics.
* Use Google Congestion Control (GCC) algorithm for baseline congestion control and bandwidth estimation delivery for every stream.
* Evaluate/train a Deep Reinforcement Learning (DRL) agent for AI-based video quality control.
* Deploy a safety detector to automatically switch between different control agents (e.g., DRL -> GCC) in case the agent's actions tend to show a negative trend in some statistics, e.g., growing RTT.

## Installation
### Docker
To overcome the dependency hell, please deploy the prepared Docker image in a container. Go to the docker folder and build the image with:
```bash
docker build -f Dockerfile -t gstwebrtcapp:latest .
```
To build docker with CUDA 12.1 (so far fixed) support, use:
```bash
docker build -f Dockerfile-cuda -t gstwebrtcapp:latest .
```
Then run the container with:
```bash
docker run -it --name gstwebrtcapp-container --network=bridge -P --privileged --cap-add=NET_ADMIN {..} gstwebrtcapp:latest bash
```
to run the CUDA container, use:
```bash 
docker run --gpus all -it --name gstwebrtcapp-container --network=bridge -P --privileged --cap-add=NET_ADMIN {..} gstwebrtcapp:latest bash
```	
where {..} are the OPTIONAL display options that could be skipped. On Linux:
```bash
-e DISPLAY=$YOUR_IPV4_ADDRESS:0 -v /tmp/.X11-unix:/tmp/.X11-unix
```
on Windows:
```bash
-e DISPLAY=host.docker.internal:0.0
```

## Usage
### MQTT
You need either an internal or external MQTT broker to run the application. You can deploy an internal broker (installed in the Docker image) with the following command:
```bash
mosquitto -c /etc/mosquitto/mosquitto.conf
```
Then you need to configure a `MqttConfig` object with either the local or external address and port of a broker and pass it to all agents and a connector class. If you want to use an external broker, you may also need to provide the username and password for the connection. The MQTT broker is used for communication between agents and different parts of the application, as well as for publishing statistics.

`MqttConfig` provides 4 default topics via `MqttGstWebrtcAppTopics` dataclass for the communication between agents:
* `actions` - for the actions of the agents
* `state` - for the state of the agents
* `stats` - for the GStreamer statistics
* `gcc` - for the Google Congestion Control bandwidth estimations

### App Config
To run the application, you need to specify the configuration of the GStreamer application in `GstWebRTCAppConfig` object and connection parameters in one of the connector classes, either `AhoyConnector` or `SinkConnector`. Some of the tuned pipelines are provided in `apps/pipelines.py`. Connector objects configure the connection parameters, the agents of the Control API and the additional services like `NetworkController` for bandiwdth limitation. The `example.py` file contains an example of how to run the application.

### AhoyApp Run
To run the AhoyApp, one needs to await the two coroutines as provided in `example.py` file. The first `connect_coro` controls the connection and communication with AhoyMedia and the second `webrtc_coro` controls all GStreamer -> WebRTC steps via its sub-coroutines aka handlers. The `example.py` file contains an example of how to run the application. Note that it won't work unless the valid AhoyDirector URL, API key and video source are specified instead of the dummy strings. 

If one stops streaming in AhoyDirector window, then `webrtc_coro` will turn into a pending state and will wait for the next play event to resume streaming. The pipeline will be stopped and set in a NULL state together with WebRTC stack and restarted on the next play event so that when the video is not played, the resources for GStreamer C objects are properly deallocated. In the meantime, the `connect_coro` coroutine will be still running to keep the connection with AhoyMedia.

### SinkApp Run
To run the SinkApp, you need to specify the configuration of the GStreamer application and the connection parameters in the same way as for the AhoyApp. It just has a different connector and application class that uses the **webrtcsink** plugin instead of **webrtcbin**. It also only needs to wait for the `webrtc_coro` coroutine. The default tuned pipeline along with the main parameters is provided in `example.py` under the `test_sink_app` coroutine. You also need to run the JS client and signaling server for webrtcsink, all instructions are written in the webrtcsink [repo](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc?ref_type=heads#usage). By clicking on the remote stream in the JS client, you accept streaming from the SinkApp. 

### Control API
The application provides the Control API to define the agents that control video quality on the fly. It is implemented in the `control` submodule. The agents are either AI enablers or congestion control algorithms. The `control/drl` submodule contains the first proactive Deep Reinforcement Learning agent, `DrlAgent`, which uses the WebRTC statistics from the viewer's browser to control the video stream. It is based on the stable-baselines3 library, uses the SAC algorithm, and serves as an example of how to implement and use the Control API. There is also an offline DRL agent `DrlOfflineAgent` which is based on the d3rlpy library and expects a Decision Transformer model for bandwidth estimation. Among the agents are also `CsvViewerRecorderAgent` that stores WebRTC statistics in csv, `GccAgent` that uses Google Congestion Control algorithm either in active or passive mode, `SafetyDetectorAgent` that switches between agents in case of negative trend in some statistics.

## License
This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Author
M.Sc. Nikita Smirnov, Intelligent Systems AG, Department of Computer Science, Kiel University, Germany.

Please contact me in case of any questions or bug reports: [Nikita Smirnov](mailto:nsm@informatik.uni-kiel.de)





