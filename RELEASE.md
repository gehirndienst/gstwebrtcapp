# Release Notes

## Version 1.0.0 (2023-12-19)

### Features
* Stream video through the given GStreamer pipeline to AhoyMedia in a fast manner without any extra steps. One needs a pipeline in a string format, a video source (like rtsp://...) and Ahoy address/api key parameters to stream.
* Support stop/resume streaming with efficient resource re-allocation and self-restarting without any additional calls.
* Flexibly configure the GStreamer pipeline elements. By default, the pipeline is tuned towards the lowest latency.
* Control resolution, framerate and bitrate of the video on the fly via setters provided by the application.
* Control video encoder and RTP payloader parameters of the pipeline.
* Receive WebRTCBin statistics from the viewer's browser.