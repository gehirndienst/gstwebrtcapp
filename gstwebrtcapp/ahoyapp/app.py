"""
app.py

Description: An application that uses GStreamer's webrtcbin plugin to stream the video source to the webrtc client.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

import asyncio
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Callable, Dict, List

import gi

gi.require_version("Gst", "1.0")
gi.require_version('GstWebRTC', '1.0')
from gi.repository import Gst
from gi.repository import GstWebRTC

from utils.base import GSTWEBRTCAPP_EXCEPTION, LOGGER, wait_for_condition

GST_ENCODERS = ["x264enc", "nvh264enc", "x265enc", "vp8enc", "vp9enc"]

DEFAULT_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate !
    capsfilter name=raw_capsfilter caps=video/x-raw,format=I420 ! queue !
    x264enc name=encoder tune=zerolatency speed-preset=superfast ! 
    rtph264pay name=payloader auto-header-extension=true aggregate-mode=zero-latency config-interval=1 ! queue !
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126, rtcp-fb-goog-remb=(boolean)true, rtcp-fb-transport-cc=(boolean)true" ! webrtc.
'''

DEFAULT_CUDA_PIPELINE = '''
    webrtcbin name=webrtc latency=1 bundle-policy=max-bundle stun-server=stun://stun.l.google.com:19302
    rtspsrc name=source location=rtsp://10.10.3.254:554 latency=10 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! videorate ! cudaupload ! cudaconvert ! 
    capsfilter name=raw_capsfilter caps=video/x-raw(memory:CUDAMemory) ! queue ! 
    nvh264enc name=encoder preset=low-latency-hq ! 
    rtph264pay name=payloader auto-header-extension=true aggregate-mode=zero-latency config-interval=1 ! queue !
    capsfilter name=payloader_capsfilter caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)126" ! webrtc.
'''


@dataclass
class GstWebRTCBinAppConfig:
    """
    Configuration class for GstWebRTCBinApp.

    :param str pipeline_str: GStreamer pipeline string. Default is the default pipeline string above.
    :param str video_url: URL of the video source (RTSP, RTMP, FILE, etc.)
    :param str encoder_gst_name: Name of the video encoder GStreamer element. Default is "x264enc".
    :param Dict[str, int] resolution: Dictionary containing width and height of the video resolution. Default is (1280, 720).
    :param int framerate: Frame rate of the video. Default is 20.
    :param int bitrate: Bitrate of the video in Kbps. Default is 2000.
    :param int fec_percentage: Forward error correction percentage. Default is 20.
    :param int max_timeout: Maximum timeout for operations in seconds. Default is 20.
    :param bool is_debug: Flag indicating whether debugging GStreamer logs are enabled. Default is False.
    """

    pipeline_str: str = DEFAULT_PIPELINE
    video_url: str = "rtsp://10.10.3.254:554"
    encoder_gst_name: str = "x264enc"
    resolution: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    framerate: int = 20
    bitrate: int = 2000
    fec_percentage: int = 20
    max_timeout: int = 20
    is_debug: bool = False


class GstWebRTCBinApp:
    """
    An application that uses GStreamer's webrtcbin plugin to stream the video source to the WebRTC client.

    :param GstWebRTCBinAppConfig config: Configuration object for GstWebRTCBinApp. Default is an instance of GstWebRTCBinAppConfig.
    """

    def __init__(self, config: GstWebRTCBinAppConfig = GstWebRTCBinAppConfig()):
        self.pipeline_str = config.pipeline_str
        self.video_url = config.video_url

        # gst objects
        self.pipeline = None
        self.webrtcbin = None
        self.source = None
        self.raw_caps = None
        self.raw_capsfilter = None
        self.encoder_gst_name = config.encoder_gst_name if config.encoder_gst_name in GST_ENCODERS else "x264enc"
        self.encoder = None
        self.pay_capsfilter = None
        self.transceivers = []
        self.bus = None

        self.is_cuda = self.encoder_gst_name == "nvh264enc"
        if self.is_cuda:
            # FIXME: replace it later with a custom configurator
            pattern = re.compile(r'(!\s*.*?name=encoder.*?!)')
            replacement_line = '! nvh264enc name=encoder preset=low-latency-hq !'
            self.pipeline_str = pattern.sub(replacement_line, self.pipeline_str)

        self.data_channels = {}

        self.resolution = config.resolution
        self.framerate = config.framerate
        self.bitrate = config.bitrate
        self.fec_percentage = config.fec_percentage

        self.max_timeout = config.max_timeout
        self.is_running = False

        Gst.init(None)
        self._init_pipeline()

        if config.is_debug:
            Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
            Gst.debug_set_active(True)

    def _init_pipeline(self):
        LOGGER.info(f"OK: initializing pipeline from a string {self.pipeline_str}...")
        self.pipeline = Gst.parse_launch(self.pipeline_str)

        # webrtcbin
        self.webrtcbin = self.pipeline.get_by_name("webrtc")
        if self.is_webrtc_ready():
            LOGGER.info("OK: webrtcbin is found in the pipeline")
        else:
            raise GSTWEBRTCAPP_EXCEPTION("can't find webrtcbin in the pipeline")

        # elems
        self.source = self.pipeline.get_by_name("source")
        if self.source.get_property("location") is not None:
            # NOTE: only sources with location property are supported (Gst plugins of *src group)
            self.source.set_property("location", self.video_url)
            LOGGER.info(f"OK: video location is set to {self.video_url}")
        self.raw_capsfilter = self.pipeline.get_by_name("raw_capsfilter")
        self.encoder = self.pipeline.get_by_name("encoder")
        self.payloader = self.pipeline.get_by_name("payloader")
        self.pay_capsfilter = self.pipeline.get_by_name("payloader_capsfilter")
        if (
            not self.source
            or not self.raw_capsfilter
            or not self.encoder
            or not self.payloader
            or not self.pay_capsfilter
        ):
            raise GSTWEBRTCAPP_EXCEPTION("can't find needed elements in the pipeline")

        self.get_transceivers()

        self.set_resolution(self.resolution["width"], self.resolution["height"])
        self.set_framerate(self.framerate)
        self.set_bitrate(self.bitrate)
        self.set_fec_percentage(self.fec_percentage)

        LOGGER.info("OK: pipeline is built")

        # switch to playing state
        r = self.pipeline.set_state(Gst.State.PLAYING)
        if r != Gst.StateChangeReturn.SUCCESS:
            raise GSTWEBRTCAPP_EXCEPTION("unable to set the pipeline to the playing state")
        else:
            self.is_running = True
        logging.info("OK: pipeline is PLAYING")

    def get_transceivers(self) -> List[GstWebRTC.WebRTCRTPTransceiver]:
        if len(self.transceivers) > 0:
            return self.transceivers
        else:
            index = 0
            if not self.is_webrtc_ready():
                raise GSTWEBRTCAPP_EXCEPTION("webrtcbin is not ready, can't get transceivers")
            else:
                while True:
                    transceiver = self.webrtcbin.emit('get-transceiver', index)
                    if transceiver:
                        transceiver.set_property("do-nack", True)
                        transceiver.set_property("fec-type", GstWebRTC.WebRTCFECType.ULP_RED)
                        self.transceivers.append(transceiver)
                        index += 1
                    else:
                        break
                if len(self.transceivers) > 0:
                    LOGGER.info(f"OK: got {len(self.transceivers)} transceivers from webrtcbin")
                else:
                    raise GSTWEBRTCAPP_EXCEPTION("can't get any single transceiver from webrtcbin")
            return self.transceivers

    def get_raw_caps(self) -> Gst.Caps:
        raw = 'video/x-raw' if not self.is_cuda else 'video/x-raw(memory:CUDAMemory)'
        s = f"{raw},format=I420,width={self.resolution['width']},height={self.resolution['height']},framerate={self.framerate}/1,"
        return Gst.Caps.from_string(s)

    def is_webrtc_ready(self) -> bool:
        return self.webrtcbin is not None

    def set_framerate(self, framerate: int) -> None:
        self.framerate = framerate
        self.raw_caps = self.get_raw_caps()
        self.raw_capsfilter.set_property("caps", self.raw_caps)
        LOGGER.info(f"ACTION: set framerate to {self.framerate}")

    def set_resolution(self, width: int, height: int) -> None:
        self.resolution = {"width": width, "height": height}
        self.raw_caps = self.get_raw_caps()
        self.raw_capsfilter.set_property("caps", self.raw_caps)
        LOGGER.info(f"ACTION: set resolution to {self.resolution['width']}x{self.resolution['height']}")

    def set_bitrate(self, bitrate_kbps: int) -> None:
        if self.encoder_gst_name.startswith("nv") or self.encoder_gst_name.startswith("x26"):
            self.encoder.set_property("bitrate", bitrate_kbps)
        elif self.encoder_gst_name.startswith("vp"):
            self.encoder.set_property("target-bitrate", bitrate_kbps * 1000)
        else:
            raise GSTWEBRTCAPP_EXCEPTION(f"encoder {self.encoder_gst_name} is not supported")

        self.bitrate = bitrate_kbps
        LOGGER.info(f"ACTION: set bitrate to {bitrate_kbps} kbps")

    def set_fec_percentage(self, percentage: int, index: int = -1) -> None:
        if len(self.transceivers) == 0:
            raise GSTWEBRTCAPP_EXCEPTION("there is no transceivers in the pipeline")
        if index > 0:
            try:
                transceiver = self.transceivers[index]
                transceiver.set_property("fec-percentage", percentage)
            except IndexError:
                raise GSTWEBRTCAPP_EXCEPTION(f"can't find tranceiver with index {index}")
        else:
            for transceiver in self.transceivers:
                transceiver.set_property("fec-percentage", percentage)

        self.fec_percentage = percentage
        LOGGER.info(f"ACTION: set fec percentage to {percentage}")

    def create_data_channel(
        self,
        name: str,
        options: Gst.Structure = None,
        callbacks: Dict[str, Callable[[Dict], Any]] | None = None,
    ) -> None:
        wait_for_condition(lambda: self.is_webrtc_ready(), self.max_timeout)
        if options is None:
            # default dc options which I found good to have
            dc_options = Gst.Structure.new_from_string("application/data-channel")
            dc_options.set_value("ordered", True)
            dc_options.set_value("max-retransmits", 2)
        else:
            dc_options = options

        data_channel = self.webrtcbin.emit('create-data-channel', name, options)
        if not data_channel:
            raise GSTWEBRTCAPP_EXCEPTION(f"Can't create data channel {name}")
        # with false you may override them on your own
        if callbacks is None:
            data_channel.connect('on-open', lambda _: LOGGER.info(f"OK: data channel {name} is opened"))
            data_channel.connect('on-close', lambda _: LOGGER.info(f"OK: data channel {name} is closed"))
            data_channel.connect('on-error', lambda _: LOGGER.info(f"ERROR: data channel {name} met an error"))
            data_channel.connect(
                'on-message-string',
                lambda _, message: LOGGER.info(f"MSG: data channel {name} received message: {message}"),
            )
        else:
            for event in callbacks:
                try:
                    data_channel.connect(event, callbacks[event])
                except Exception as e:
                    raise GSTWEBRTCAPP_EXCEPTION(f"Can't attach callback for event {event} to data channel {name}: {e}")
        self.data_channels[name] = data_channel
        LOGGER.info(f"OK: created data channel {name}")

    def is_data_channel_ready(self, data_channel_name: str) -> bool:
        dc = self.data_channels[data_channel_name]
        return dc and dc.get_property("ready-state") == GstWebRTC.WebRTCDataChannelState.OPEN

    def send_data_channel_message(self, data_channel_name: str, data: Dict[str, Any]) -> None:
        if not self.is_data_channel_ready(data_channel_name):
            LOGGER.debug(f"dropping message, data channel {data_channel_name} is not ready")
            return
        self.data_channels[data_channel_name].emit("send-string", json.dumps(data))

    async def handle_pipeline(self) -> None:
        # run the loop to fetch messages from the bus
        LOGGER.info("OK: PIPELINE HANDLER IS ON -- ready to read pipeline bus messages")
        self.bus = self.pipeline.get_bus()
        try:
            while True:
                message = self.bus.timed_pop_filtered(
                    0.5 * Gst.SECOND,
                    Gst.MessageType.APPLICATION
                    | Gst.MessageType.EOS
                    | Gst.MessageType.ERROR
                    | Gst.MessageType.LATENCY
                    | Gst.MessageType.STATE_CHANGED,
                )
                if message:
                    message_type = message.type
                    if message_type == Gst.MessageType.APPLICATION:
                        if message.get_structure().get_name() == "termination":
                            LOGGER.info("INFO: received termination message, preparing to terminate the pipeline...")
                            break
                    elif message_type == Gst.MessageType.EOS:
                        LOGGER.info("INFO: received EOS message, preparing to terminate the pipeline...")
                        break
                    elif message_type == Gst.MessageType.ERROR:
                        err, _ = message.parse_error()
                        LOGGER.error(f"ERROR: Pipeline error")
                        self.is_running = False
                        raise GSTWEBRTCAPP_EXCEPTION(err.message)
                    elif message_type == Gst.MessageType.LATENCY:
                        try:
                            self.pipeline.recalculate_latency()
                            LOGGER.debug("INFO: latency is recalculated")
                        except Exception as e:
                            raise GSTWEBRTCAPP_EXCEPTION(f"can't recalculate latency, reason: {e}")
                    elif message_type == Gst.MessageType.STATE_CHANGED:
                        if message.src == self.pipeline:
                            old, new, _ = message.parse_state_changed()
                            LOGGER.info(
                                "INFO: Pipeline state changed from "
                                f"{Gst.Element.state_get_name(old)} to "
                                f"{Gst.Element.state_get_name(new)}"
                            )
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            LOGGER.info("ERROR: handle_pipeline, KeyboardInterrupt received, exiting...")

        LOGGER.info("OK: PIPELINE HANDLER IS OFF")
        self.is_running = False
        self.terminate_pipeline()

    def terminate_pipeline(self) -> None:
        LOGGER.info("OK: terminating pipeline...")
        for data_channel_name in self.data_channels.keys():
            self.data_channels[data_channel_name].emit('close')
            LOGGER.info(f"OK: data channel {data_channel_name} is closed")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            LOGGER.info("OK: set pipeline state to NULL")
        if self.webrtcbin:
            self.webrtcbin.set_state(Gst.State.NULL)
            self.webrtcbin = None
            LOGGER.info("OK: set webrtcbin state to NULL")
        self.source = None
        self.encoder = None
        self.raw_caps = None
        self.raw_capsfilter = None
        self.pay_capsfilter = None
        self.transceivers = []
        self.data_channels = {}
        LOGGER.info("OK: pipeline is terminated!")

    def send_termination_message_to_bus(self):
        if self.bus is not None:
            LOGGER.info("OK: sending termination message to the pipeline's bus")
            self.bus.post(Gst.Message.new_application(None, Gst.Structure.new_empty("termination")))
