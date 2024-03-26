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
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 !
    x264enc name=encoder tune=zerolatency threads=8 key-int-max=60 aud=true cabac=1 bframes=2 vbv-buf-capacity=120 ! 
    rtph264pay name=payloader auto-header-extension=true aggregate-mode=zero-latency config-interval=1 mtu=1250 !
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126" ! webrtc.
'''
# for internal reolink camera that streams 4k in hevc
DEFAULT_H265_IN_WEBRTCBIN_H264_OUT_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! 
    x264enc name=encoder tune=zerolatency threads=8 key-int-max=60 aud=true cabac=1 bframes=2 vbv-buf-capacity=120 !
    rtph264pay name=payloader auto-header-extension=true aggregate-mode=zero-latency config-interval=1 mtu=1250 ! 
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126, rtcp-fb-goog-remb=(boolean)true, rtcp-fb-transport-cc=(boolean)true" ! webrtc.
'''

DEFAULT_H265_IN_WEBRTCBIN_H264_OUT_CUDA_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! videoscale ! videorate ! cudaupload ! cudaconvert ! 
    capsfilter name=raw_capsfilter caps=video/x-raw(memory:CUDAMemory) ! 
    nvh264enc name=encoder preset=low-latency-hq gop-size=60 rc-mode=cbr-ld-hq aud=true bframes=2 zerolatency=true ! 
    rtph264pay name=payloader auto-header-extension=true config-interval=1 mtu=1350 ! 
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126" ! webrtc.
'''
# NOTE: Chrome does not play HEVC, a black screen
DEFAULT_WEBRTCBIN_H265_OUT_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! 
    x265enc name=encoder key-int-max=2560 speed-preset=superfast tune=zerolatency ! h265parse ! 
    rtph265pay name=payloader auto-header-extension=true config-interval=1 mtu=1350 ! 
    capsfilter name=payloader_capsfilter ! webrtc.
'''

DEFAULT_WEBRTCBIN_H265_OUT_CUDA_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate ! cudaupload ! cudaconvert !
    capsfilter name=raw_capsfilter caps=video/x-raw(memory:CUDAMemory),format=I420 ! 
    nvh265enc name=encoder preset=low-latency-hq gop-size=60 rc-mode=cbr-ld-hq aud=true bframes=2 zerolatency=true ! h265parse ! 
    rtph265pay name=payloader auto-header-extension=true config-interval=1 mtu=1350 ! 
    capsfilter name=payloader_capsfilter ! webrtc.
'''

# works excellent
DEFAULT_WEBRTCBIN_VP8_OUT_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! 
    vp8enc name=encoder deadline=1 keyframe-max-dist=2000 keyframe-mode=disabled threads=8 cpu-used=4 ! 
    rtpvp8pay name=payloader auto-header-extension=true picture-id-mode=15-bit mtu=1350 ! 
    capsfilter name=payloader_capsfilter ! webrtc.
'''

# works good (more or less stable 20 fps on a good machine)
DEFAULT_WEBRTCBIN_VP9_OUT_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! 
    vp9enc name=encoder deadline=1 keyframe-max-dist=2000 keyframe-mode=disabled row-mt=true threads=8 cpu-used=4 ! 
    rtpvp9pay name=payloader auto-header-extension=true picture-id-mode=15-bit mtu=1350 ! 
    capsfilter name=payloader_capsfilter ! webrtc.
'''

# NOTE: does not work so far, breaks down to 0 fps in several seconds
DEFAULT_WEBRTCBIN_AV1_OUT_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! 
    av1enc name=encoder usage-profile=realtime qos=true threads=8 cpu-used=4 ! av1parse !
    rtpav1pay name=payloader auto-header-extensions=true mtu=1350 ! 
    capsfilter name=payloader_capsfilter ! webrtc.
'''

DEFAULT_WEBRTCBIN_SVTAV1_OUT_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! 
    svtav1enc name=encoder ! av1parse !
    rtpav1pay name=payloader mtu=1350 ! 
    capsfilter name=payloader_capsfilter ! webrtc.
'''

DEFAULT_BIN_CUDA_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate ! cudaupload ! cudaconvert ! 
    capsfilter name=raw_capsfilter caps=video/x-raw(memory:CUDAMemory) ! 
    nvh264enc name=encoder preset=low-latency-hq gop-size=60 rc-mode=cbr-ld-hq aud=true bframes=2 zerolatency=true ! 
    rtph264pay name=payloader auto-header-extension=true aggregate-mode=zero-latency config-interval=1 mtu=1250 ! 
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126" ! webrtc.
'''

DEFAULT_SINK_PIPELINE = '''
    webrtcsink name=ws signaller::uri=ws://127.0.0.1:8443 do-retransmission=true do-fec=true congestion-control=disabled
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    video/x-raw,format=I420,framerate=25/1 ! ws.
'''

DEFAULT_SINK_CUDA_PIPELINE = '''
    webrtcsink name=ws signaller::uri=ws://127.0.0.1:8443 do-retransmission=true do-fec=true congestion-control=disabled
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate ! cudaupload ! cudaconvert ! 
    video/x-raw(memory:CUDAMemory),format=I420,framerate=25/1 ! ws.
'''

DEFAULT_SINK_H265_PIPELINE = '''
    webrtcsink name=ws signaller::uri=ws://127.0.0.1:8443 do-retransmission=true do-fec=true congestion-control=disabled
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! videoscale ! videorate !
    video/x-raw,format=I420,framerate=25/1 ! ws.
'''
