# GStreamer WebRTC application
This repo contains `GstWebRTCApp` -- the Python application to stream video feeds from the given RTSP sources via the GStreamer pipeline to two different WebRTC clients with the possibility to control the video quality on the fly in automatic mode using AI/congestion control agents or in manual/rule-based mode via MQTT messaging. AhoyApp streams the source to the AhoyRTC Director WebRTC client maintained by ADDIX GmbH. SinkApp streams to the open-source js WebRTC client maintained by [webrtcsink team](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc?ref_type=heads). Both applications are based on the [GStreamer] framework (https://gstreamer.freedesktop.org/) and use its Python bindings.

AhoyApp uses the low-level **webrtcbin** plugin for streaming, which is a part of the GStreamer WebRTC stack. It also provides custom signaling to connect to the AhoyRTC Director engine and to handle the WebRTC streaming properly. SinkApp uses the **webrtcsink** plugin and connects to the provided JS client that one needs to run together with the signaling server for the webrtcsink plugin. They both require GStreamer >= 1.22 as well as its Python GI bindings (python ^3.11). The application is deployed in a Docker container with all necessary dependencies and configurations (CPU and CUDA environments) to run the application.

## Features
The main features of the applications are:
* Stream videos with the GStreamer pipelines to the proprietary AhoyRTC Director or the independent GstWebRTCAPI (WebRTC JS client in browser) WebRTC browser client.
* Flexibly configure the GStreamer's pipeline elements or use pre-built pipelines. By default, all pipelines are tuned for the lowest latency.
* Run multiple streams and auto-allocate actions for each stream via dynamic importance weights. Control the feeds over the MQTT messages. 
* Control resolution, framerate and bitrate of the video on the fly via feed controller or let the AI/CC agents do it in auto-mode with the option of manual control overtaking.
* Encode with H264, H265, AV1, VP8 and VP9 codecs with their respective hardware acceleration (nv* plugins for H264/5 and AV1 codecs). 
* Deploy the application with the provided CUDA or CPU Docker environment with all necessary dependencies and configurations. All codecs are built from source. The internal MQTT broker is installed. Python AI stack is also installed.
* Publish and collect WebRTC statistics, actions and states of the video streams over the internal or external MQTT broker.
* Use the Google Congestion Control (GCC) algorithm for baseline congestion control and bandwidth estimation delivery for every stream.
* Evaluate/train a Deep Reinforcement Learning (DRL) agent for AI-based video quality control.
* Deploy a safety detector to automatically switch between different control agents (e.g., DRL -> GCC) in case the agent's actions tend to show a negative trend in some statistics, e.g., growing RTT.

## Installation
### Docker
Please follow the instructions in the docker folder: [Docker README](docker/README.md)

### GstWebRTCApp
The source code is copied into the docker image for further development. By default, the project is installed globally with the `install.sh` script. It uses the `poetry` package manager to build the wheel from the source code and install the application with pip for the root user in a container. If you want to install the application into the virtual environment (locally), you can use the following commands after cloning the repo:
```bash
poetry install && poetry run python <your_script.py>
```

You can also install it remotely from the repo with poetry into the virtual environment by uninstalling it globally with pip and later you can delete the source code as well.

The project has not been published on PyPi nor a wheel is attached to the repo's releases. The reason is that there are two environments for the application: CPU and CUDA. It is not possible to build a universal wheel for both environments. The Docker image is the best way to deploy the application with all necessary dependencies and configurations.

## Usage
### Examples
Go to the examples folder: [Examples](examples/README.md)

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
* `controller` - (multiple feeds) for the feed actions performed by the `FeedController` class

### App Config
To run the application, you need to specify the configuration of the GStreamer application in the `GstWebRTCAppConfig` object and connection parameters in one of the connector classes, either `AhoyConnector` or `SinkConnector`. Some of the tuned pipelines are provided in `apps/pipelines.py`. Connector objects configure the connection parameters, the agents of the Control API and the additional services like `NetworkController` for bandwidth limitation (NOTE: works only on the Linux host machine).

### AhoyApp Run
You need to purchase a license for the [AhoyRTC service](https://ahoyrtc.com/) from ADDIX GmbH to run the AhoyApp. By that, you will get the AhoyDirector URL and API key to stream.

To run the AhoyApp, one needs to await the two coroutines. The first `connect_coro` controls the connection and communication with the AhoyRTC Director and the second `webrtc_coro` controls all GStreamer -> WebRTC steps via its sub-coroutines aka handlers. If one stops streaming in the browser UI, then `webrtc_coro` will turn into a pending state and will wait for the next play event to resume streaming. The pipeline will be stopped and set in a NULL state together with the WebRTC stack and restarted on the next play event so that when the video is not played, the resources for GStreamer C objects are properly deallocated. In the meantime, the `connect_coro` coroutine will be still running to keep the connection with the AhoyRTC Director.

### SinkApp Run
This connector uses a free WebRTC client provided by the webrtcsink repo.

To run the SinkApp, you need to specify the configuration of the GStreamer application and the connection parameters in the same way as for the AhoyApp. It just has a different connector and application class that uses the **webrtcsink** plugin instead of the **webrtcbin**. It also only needs to wait for the `webrtc_coro` coroutine. The default example operates with 2 feeds being controlled and viewed by their client. You also need to run the JS client and signaling server for the webrtcsink, all instructions are written in the webrtcsink [repo](https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc?ref_type=heads#usage). By clicking on the remote stream in the JS client, you accept streaming from the SinkApp. 

### Control API
The application provides the Control API to define the agents that control video quality on the fly. It is implemented in the `control` submodule. The agents are either AI enablers or congestion control algorithms. The `control/drl` submodule contains the first proactive Deep Reinforcement Learning agent, `DrlAgent`, which uses the WebRTC statistics from the viewer's browser to control the video stream. It is based on the stable-baselines3 library, uses the SAC algorithm, and serves as an example of how to implement and use the Control API. There is also an offline DRL agent `DrlOfflineAgent` which is based on the d3rlpy library and expects a Decision Transformer model for bandwidth estimation. Among the agents is `RecorderAgent` which stores WebRTC statistics in csv or publishes cooked stats over the MQTT, `GccAgent` that uses the Google Congestion Control algorithm either in active or passive mode, `SafetyDetectorAgent` which switches between agents in case of a negative trend in some statistics.

## License
This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Author
M.Sc. Nikita Smirnov, Intelligent Systems AG, Department of Computer Science, Kiel University, Germany.

Please contact me in case of any questions or bug reports: [Nikita Smirnov](mailto:nsm@informatik.uni-kiel.de)





