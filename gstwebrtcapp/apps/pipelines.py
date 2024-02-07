"""
pipelines.py

Description: A header with default GStreamer pipelines for the WebRTC applications.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

DEFAULT_BIN_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! queue !
    x264enc name=encoder tune=zerolatency speed-preset=superfast threads=4 key-int-max=2560 b-adapt=false vbv-buf-capacity=120 ! 
    rtph264pay name=payloader auto-header-extension=true aggregate-mode=zero-latency config-interval=1 ! queue !
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126, rtcp-fb-goog-remb=(boolean)true, rtcp-fb-transport-cc=(boolean)true" ! webrtc.
'''

DEFAULT_BIN_AV1_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! queue !
    av1enc name=encoder usage-profile=realtime qos=true cpu-used=4 ! av1parse !
    rtpav1pay name=payloader ! queue !
    capsfilter name=payloader_capsfilter ! webrtc.
'''

DEFAULT_BIN_CUDA_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate ! cudaupload ! cudaconvert ! 
    capsfilter name=raw_capsfilter caps=video/x-raw(memory:CUDAMemory) ! queue ! 
    nvh264enc name=encoder preset=low-latency-hq gop-size=2560 rc-mode=cbr-ld-hq zerolatency=true ! 
    rtph264pay name=payloader auto-header-extension=true aggregate-mode=zero-latency config-interval=1 ! queue !
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126" ! webrtc.
'''

DEFAULT_SINK_PIPELINE = '''
    webrtcsink name=ws signaller::uri=ws://127.0.0.1:8443 do-retransmission=true do-fec=true congestion-control=disabled
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    video/x-raw,format=I420,framerate=25/1 ! queue ! ws.
'''

DEFAULT_SINK_H265_PIPELINE = '''
    webrtcsink name=ws signaller::uri=ws://127.0.0.1:8443 do-retransmission=true do-fec=true congestion-control=disabled
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! queue ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! videoscale ! videorate !
    video/x-raw,format=I420,framerate=25/1 ! queue ! ws.
'''
