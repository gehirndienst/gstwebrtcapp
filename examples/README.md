All examples should be run with the SinkApp since AhoyApp requires a license. To do this, open four terminals in the vscode attached to the container and run the following commands in each terminal:
1. In the first terminal, run the local mosquitto broker:
```bash
mosquitto -c /etc/mosquitto/mosquitto.conf
```
2. In the second terminal, run the signalling server:
```bash
cd /opt/gstreamer/subprojects/gst-plugins-rs/net/webrtc/signalling/ && cargo run --bin gst-webrtc-signalling-server
```
3. In the third terminal, run the JS client:
```bash
cd /opt/gstreamer/subprojects/gst-plugins-rs/net/webrtc/gstwebrtc-api/ && npm install && npm start
```
4. In the fourth terminal, run the example script:
```bash
python <selected_example.py>
```

# Prerequisites
- Valid RTSP stream URL(s). Can be artificially generated with OBS Studio + RTSP server plugin for it from the webcam or any other source.

# Examples
## 1. Multiple feeds with GCC control
run test_multiple.py. It tests running 2 RTSP streams and controlling them with GCC. You should see the allocated actions printed in the console.
