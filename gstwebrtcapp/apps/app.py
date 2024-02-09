"""
app.py

Description: An abstract class that provides an interface for the applications that use GStreamer's webrtc plugins to stream the video source
and to control the QoE parameters of the video stream.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

from abc import ABCMeta, abstractmethod
import asyncio
from dataclasses import dataclass, field
import enum
import json
import re
from typing import Any, Callable, Dict, List

import gi

gi.require_version("Gst", "1.0")
gi.require_version('GstWebRTC', '1.0')
from gi.repository import Gst
from gi.repository import GstWebRTC

from apps.pipelines import DEFAULT_BIN_PIPELINE
from utils.base import LOGGER, GSTWEBRTCAPP_EXCEPTION, wait_for_condition
from utils.gst import get_gst_encoder_name


@dataclass
class GstWebRTCAppConfig:
    """
    Configuration class for GstWebRTCApp.

    :param str pipeline_str: GStreamer pipeline string. Default is the default pipeline string for webrtcbin.
    :param str video_url: URL of the video source (RTSP, RTMP, FILE, etc.)
    :param str codec: Name of the video codec (encoder). Default is "h264". Possible options are "h264", "h265", "vp8", "vp9", "av1".
    :param int bitrate: Bitrate of the video in Kbps. Default is 2000.
    :param Dict[str, int] resolution: Dictionary containing width and height of the video resolution. Default is {"width": 1280, "height": 720}.
    :param int framerate: Frame rate of the video. Default is 20.
    :param int fec_percentage: Forward error correction percentage. Default is 20.
    :param List[Dict[str, Any]] data_channels_cfgs: List of dictionaries containing data channel configurations.
    :param int max_timeout: Maximum timeout for operations in seconds. Default is 60.
    :param bool is_cuda: Flag indicating whether the pipeline uses CUDA for HA encoding. Currently only H264 is supported. Default is False.
    :param bool is_debug: Flag indicating whether debugging GStreamer logs are enabled. Default is False.
    """

    pipeline_str: str = DEFAULT_BIN_PIPELINE
    video_url: str | None = None
    codec: str = "h264"
    bitrate: int = 2000
    resolution: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    framerate: int = 20
    fec_percentage: int = 20
    data_channels_cfgs: List[Dict[str, Any]] = field(default_factory=lambda: [])
    max_timeout: int = 60
    is_cuda: bool = False
    is_debug: bool = False


class GstWebRTCApp(metaclass=ABCMeta):
    """
    Abstract GstWebRTCApp class that defines set of actions to establish and control the pipeline
    """

    def __init__(self, config: GstWebRTCAppConfig, **kwargs) -> None:
        # NOTE: call super().__init__() in the derived classes AFTER declaring their GST instance variables
        self.pipeline_str = config.pipeline_str
        self.video_url = config.video_url
        self.encoder_gst_name = get_gst_encoder_name(config.codec, config.is_cuda)
        self.is_cuda = config.is_cuda and config.codec.startswith('h26')  # FIXME: so far only h264/h265 are supported
        if self.is_cuda:
            # FIXME: add support for inserting cudaupload and cudaconvert into the pipeline
            pattern = re.compile(r'(!\s*.*?name=encoder.*?!)')
            replacement_line = (
                '! nvh264enc name=encoder preset=low-latency-hq gop-size=2560 rc-mode=cbr-ld-hq zerolatency=true !'
                if config.codec == 'h264'
                else '! nvh265enc name=encoder preset=low-latency-hq gop-size=2560 rc-mode=cbr-ld-hq zerolatency=true !'
            )
            self.pipeline_str = pattern.sub(replacement_line, self.pipeline_str)

        self.bitrate = config.bitrate
        self.resolution = config.resolution
        self.framerate = config.framerate
        self.fec_percentage = config.fec_percentage

        self.data_channels_cfgs = config.data_channels_cfgs
        self.data_channels = {}
        self.max_timeout = config.max_timeout
        self.is_running = False

        Gst.init(None)
        if config.is_debug:
            Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
            Gst.debug_set_active(True)

        self._init_pipeline()

    @abstractmethod
    def _init_pipeline(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _post_init_pipeline(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_bitrate(self, bitrate: int) -> None:
        """
        Set the bitrate of the video stream.

        :param int bitrate: Bitrate of the video in Kbps.
        """
        pass

    @abstractmethod
    def set_resolution(self, width: int, height: int) -> None:
        """
        Set the resolution of the video stream.

        :param int width: Width of the video.
        :param int height: Height of the video.
        """
        pass

    @abstractmethod
    def set_framerate(self, framerate: int) -> None:
        """
        Set the framerate of the video stream.

        :param int framerate: Framerate of the video.
        """
        pass

    @abstractmethod
    def set_fec_percentage(self, percentage: int) -> None:
        """
        Set the FEC percentage of the video stream.

        :param int percentage: FEC percentage of the video.
        """
        pass

    def is_webrtc_ready(self) -> bool:
        return self.webrtcbin is not None

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
                        elif message.get_structure().get_name() == "post-init":
                            LOGGER.info(
                                "INFO: received post-init message, preparing to continue initializing the pipeline"
                            )
                            self._post_init_pipeline()
                    elif message_type == Gst.MessageType.EOS:
                        LOGGER.info("INFO: got EOS message, preparing to terminate the pipeline...")
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

    def send_termination_message_to_bus(self) -> None:
        if self.bus is not None:
            LOGGER.info("OK: sending termination message to the pipeline's bus")
            self.bus.post(Gst.Message.new_application(None, Gst.Structure.new_empty("termination")))
        else:
            LOGGER.error("ERROR: can't send termination message to the pipeline's bus, bus is None")

    def send_post_init_message_to_bus(self) -> None:
        if self.bus is not None:
            LOGGER.info("OK: sending post-init message to the pipeline's bus")
            self.bus.post(Gst.Message.new_application(None, Gst.Structure.new_empty("post-init")))
        else:
            LOGGER.error("ERROR: can't send post-init message to the pipeline's bus, bus is None")
