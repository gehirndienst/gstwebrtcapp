# Release Notes

## Version 1.0.0 (2023-12-19)

### Features
* Stream video through the given GStreamer pipeline to AhoyMedia in a fast manner without any extra steps. One needs a pipeline in a string format, a video source (like rtsp://...) and Ahoy address/api key parameters to stream.
* Support stop/resume streaming with efficient resource re-allocation and self-restarting without any additional calls.
* Flexibly configure the GStreamer pipeline elements. By default, the pipeline is tuned towards the lowest latency.
* Control resolution, framerate and bitrate of the video on the fly via setters provided by the application.
* Control video encoder and RTP payloader parameters of the pipeline.
* Receive WebRTCBin statistics from the viewer's browser.

## Version 1.0.1 (2024-01-15)

### Features
* New Control API introduced in the `control` submodule. It allows to define the AI-enablers / CC algorithms to control the video stream on the fly via the API and GStreamer app setters.
* First Deep Reinforcement Learning AI-enabler (`control/drl`) that uses the WebRTC stats from the viewer's browser to control the video stream. It is based on the stable-baselines3 library and uses the SAC algorithm. Currently the reward design and hyperparameters are not publicly available.
* Fully isolated Docker environment with CUDA, with a built-in VPN support via openconnect and with tcconfig to tweak the network. Check `docker` folder for the corresponding Dockerfiles.
* Support GStreamer NVENC encoders for CUDA containers (h264, h265, vp8, vp9, av1). Currently no support for VAAPI or Jetson Gstreamer plugins.

## Version 1.1.0 (2024-02-16)

### Features
* New sink application introduced in the `apps/sinkapp` submodule. It allows to stream via high-level webrtcsink rs plugin to the JS client.
* New recorder agent introduced in the `control/recorder` submodule. It saves webrtc stats to the CSV file.
* Refactoring and re-structuring of the codebase. Improving the docker environment. Extended examples for ahoy/sink apps, hardware accelerated encoders, control API, recorder and the full DRL agent.