"""
app.py

Description: An application that controls the GStreamer pipeline with webrtcsink as WebRTC producer.
Requires browser js client as well as websocket signalling server to connect to the pipeline and control it.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

import asyncio
from collections import OrderedDict
import re
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from apps.app import GstWebRTCApp, GstWebRTCAppConfig
from apps.pipelines import DEFAULT_SINK_PIPELINE
from utils.base import GSTWEBRTCAPP_EXCEPTION, LOGGER, wait_for_condition
from utils.gst import DEFAULT_GCC_SETTINGS


class SinkApp(GstWebRTCApp):
    """
    An application that uses GStreamer's WEBRTCSINK plugin to stream the video source to the Google Chrome API Client.
    """

    def __init__(
        self,
        config: GstWebRTCAppConfig = GstWebRTCAppConfig(pipeline_str=DEFAULT_SINK_PIPELINE),
    ) -> None:
        self.pipeline = None
        self.webrtcsink = None
        self.webrtcsink_pipeline = None
        self.webrtcsink_elements = OrderedDict()
        self.webrtcbin = None
        self.source = None
        self.gcc = None
        self.gcc_estimated_bitrates = asyncio.Queue()
        self.encoder = None
        self.encoder_caps = None
        self.encoder_capsfilter = None
        self.transceivers = []
        self.bus = None

        super().__init__(config)

    def _init_pipeline(self) -> None:
        # pipeline
        if self.video_url is not None:
            self.pipeline_str, _ = re.subn(r'(location=)[^ ]*', f'\\1{self.video_url}', self.pipeline_str)
        LOGGER.info(f"OK: initializing pipeline from a string {self.pipeline_str}...")
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        if not self.pipeline:
            raise GSTWEBRTCAPP_EXCEPTION(f"can't create pipeline from {self.pipeline_str}")

        # webrtcsink
        self.webrtcsink = self.pipeline.get_by_name("ws")
        if not self.webrtcsink:
            raise GSTWEBRTCAPP_EXCEPTION(f"Can't get webrtcsink from the pipeline {self.pipeline_str}")
        LOGGER.info("OK: webrtcsink is found in the pipeline")

        # get webrtcsink pipeline and collect its mutable elements
        self.webrtcsink.connect(
            'consumer-pipeline-created',
            self._cb_webrtcsink_pipeline_created,
        )

        # get webrtcbin to create data channels and set up the WebRTC connection
        self.webrtcsink.connect(
            'consumer-added',
            self._cb_webrtcbin_created,
        )

        # get all encoders to tweak their properties later
        self.webrtcsink.connect(
            'encoder-setup',
            self._cb_encoder_setup,
        )

        # assign video caps directly to the encoder selecting therefore the target encoder
        # NOTE: it can only be done for the encoder thanks to video-caps property, others require to tweak their capsfilters
        enc_caps = self.get_caps(is_only_header=True)
        enc_caps.set_value("stream-format", "byte-stream")
        self.webrtcsink.set_property("video-caps", enc_caps)
        LOGGER.info(f"OK: set target video caps to webrtcsink")

        # create gcc estimator
        self.gcc = Gst.ElementFactory.make("rtpgccbwe")
        if not self.gcc:
            raise GSTWEBRTCAPP_EXCEPTION("Can't create rtpgccbwe")
        LOGGER.info("OK: rtpgccbwe is created")

        # switch to playing state
        r = self.pipeline.set_state(Gst.State.PLAYING)
        if r == Gst.StateChangeReturn.FAILURE:
            # NOTE: unlike the webrtcbin, webrtcsink pipeline returns GST_STATE_ASYNC so that we should check it has not failed
            raise GSTWEBRTCAPP_EXCEPTION("unable to set the pipeline to the playing state")
        else:
            self.is_running = True

    def _post_init_pipeline(self) -> None:
        LOGGER.info("OK: start post init pipeline actions after starting the bus...")
        if not self.webrtcsink_elements:
            raise GSTWEBRTCAPP_EXCEPTION("WebRTCSink elements are not collected")

        # wait until the target encoder is found and raise an exception if it is not found
        wait_for_condition(lambda: self.encoder is not None, self.max_timeout)

        # HACK: you can't tweak caps of enc src pad, they are not writable, but you can tweak its following capsfilter
        try:
            elements_key_list = list(self.webrtcsink_elements.keys())
            encoder_index = elements_key_list.index(self.encoder.get_name())
            self.encoder_capsfilter = self.webrtcsink_elements[elements_key_list[encoder_index + 1]]
        except ValueError:
            raise GSTWEBRTCAPP_EXCEPTION("Can't find encoder in the webrtcsink pipeline")

        # set initial values
        self.set_bitrate(self.bitrate)
        self.set_resolution(self.resolution["width"], self.resolution["height"])
        self.set_framerate(self.framerate)
        self.set_fec_percentage(self.fec_percentage)

        # ok!
        LOGGER.info("OK: WebRTCSink is fully ready!")

    def get_caps(self, is_only_header: bool = False) -> Gst.Caps:
        enc_part = ""
        match self.encoder_gst_name:
            case "vp8enc":
                enc_part = "video/x-vp8"
            case "vp9enc":
                enc_part = "video/x-vp9"
            case "x264enc" | "nvh264enc":
                enc_part = "video/x-h264"
            case "x265enc":
                enc_part = "video/x-h265"
            case _:
                raise GSTWEBRTCAPP_EXCEPTION(f"unknown codec {self.encoder_gst_name}")
        if is_only_header:
            return Gst.Caps.from_string(enc_part)
        else:
            return Gst.Caps.from_string(
                f"{enc_part},format=I420,width={self.resolution['width']},height={self.resolution['height']},framerate={self.framerate}/1,"
            )

    def set_bitrate(self, bitrate_kbps: int) -> None:
        if self.encoder_gst_name.startswith("nv") or self.encoder_gst_name.startswith("x26"):
            self.encoder.set_property("bitrate", bitrate_kbps)
        elif self.encoder_gst_name.startswith("vp"):
            self.encoder.set_property("target-bitrate", bitrate_kbps * 1000)
        else:
            raise GSTWEBRTCAPP_EXCEPTION(f"encoder {self.encoder_gst_name} is not supported")

        self.bitrate = bitrate_kbps
        LOGGER.info(f"ACTION: set bitrate to {self.bitrate} kbps")

    def set_resolution(self, width: int, height: int) -> None:
        self.resolution = {"width": width, "height": height}
        self.encoder_caps = self.get_caps()
        self.encoder_capsfilter.set_property("caps", self.encoder_caps)
        LOGGER.info(f"ACTION: set resolution to {self.resolution['width']}x{self.resolution['height']}")

    def set_framerate(self, framerate: int) -> None:
        # FIXME: 25 is hardcoded in a default pipeline. That is reasonable for 99% of streams, make configurable later
        self.framerate = min(25, framerate)
        self.encoder_caps = self.get_caps()
        self.encoder_capsfilter.set_property("caps", self.encoder_caps)
        LOGGER.info(f"ACTION: set framerate to {self.framerate}")

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

    # additional setter to set fully custom encoder caps
    def set_encoder_caps(self, caps_dict: dict) -> None:
        new_caps_str = self.get_sink_video_caps().to_string()
        for key in caps_dict:
            new_caps_str += f",{key}={str(caps_dict[key])}"
        self.encoder_caps = Gst.Caps.from_string(new_caps_str)
        self.encoder_capsfilter.set_property("caps-change-mode", "delayed")
        self.encoder_capsfilter.set_property("caps", self.encoder_caps)
        LOGGER.info(f"ACTION: set new caps for encoder {self.encoder_caps.to_string()}")

    # additional setter to set fully custom transceiver props
    def set_webrtc_transceiver(self, props_dict: dict, index: int = -1) -> None:
        try:
            transceiver = self.transceivers[index]
            for key in props_dict:
                old_prop = transceiver.get_property(key)
                transceiver.set_property(key, props_dict[key])
                LOGGER.info(f"ACTION: changed {key} for {transceiver.get_name()} from {old_prop} to {props_dict[key]}")
        except IndexError:
            raise GSTWEBRTCAPP_EXCEPTION(f"Can't find tranceiver with index {index}")

    ################# NOTIFIERS #####################
    ## gcc
    def on_estimated_bitrate_changed(self, bwe, pspec) -> None:
        if bwe and pspec.name == "estimated-bitrate":
            estimated_bitrate = self.gcc.get_property(pspec.name)
            self.gcc_estimated_bitrates.put_nowait(estimated_bitrate)
        else:
            raise GSTWEBRTCAPP_EXCEPTION("Can't get estimated bitrate by gcc")

    ################# CALLBACKS #####################
    ## get webrtcbin
    def _cb_webrtcbin_created(self, _, __, bin):
        if bin:
            LOGGER.info(f"OK: got webrtcbin, collecting its transceivers...")
            self.webrtcbin = bin
            # NOTE: it is possible to create data channels ONLY here because webrtcbin does not support
            # renegotiation for new data channels. Therefore pass their cfgs as a parameter to the constructor
            # and call here before webrtcbin goes into STABLE state
            for dc_cfg in self.data_channels_cfgs:
                self.create_data_channel(dc_cfg["name"], dc_cfg["options"], dc_cfg["callbacks"])

            # add gcc estimator
            self.webrtcbin.connect("request-aux-sender", self._cb_add_gcc)
            self.webrtcbin.connect('deep-element-added', self._cb_deep_element_added)

            # get all transceivers
            index = 0
            while True:
                transceiver = self.webrtcbin.emit('get-transceiver', index)
                if transceiver:
                    self.transceivers.append(transceiver)
                    index += 1
                else:
                    break

    ## get webrtcsink pipeline an all its elements
    def _cb_webrtcsink_pipeline_created(self, _, __, ppl):
        if ppl:
            LOGGER.info(f"OK: got webrtcsink pipeline, collecting its elements...")
            self.webrtcsink_pipeline = ppl
            self.webrtcsink_pipeline.connect(
                'deep-element-added',
                self._cb_get_all_elements,
            )

    ## get all encoders to tweak their properties later
    def _cb_encoder_setup(self, _, __, ___, enc):
        if enc and self.webrtcsink_elements:
            name = str(enc.get_name())
            self.encoder = enc
            if name.startswith(self.encoder_gst_name):
                LOGGER.info(f"OK: the target encoder is found: {name}")
            else:
                LOGGER.info(f"OK: another than {self.encoder_gst_name} encoder is found: {name}")
        return False

    ## get all elements from the webrtcsink pipeline
    def _cb_get_all_elements(self, _, __, element):
        if element:
            self.webrtcsink_elements[element.get_name()] = element

    def _cb_deep_element_added(self, _, __, ___):
        pass

    ## set gcc algorithm in passive mode and save its estimated bitrate on each notification
    def _cb_add_gcc(self, _, __):
        LOGGER.info("OK: adding gcc estimator...")
        min_bitrate = (
            self.gcc_settings["min-bitrate"]
            if "min-bitrate" in self.gcc_settings
            else DEFAULT_GCC_SETTINGS["min-bitrate"]
        )
        max_bitrate = (
            self.gcc_settings["max-bitrate"]
            if "max-bitrate" in self.gcc_settings
            else DEFAULT_GCC_SETTINGS["max-bitrate"]
        )
        self.gcc.set_property("min-bitrate", min_bitrate)
        self.gcc.set_property("max-bitrate", max_bitrate)
        self.gcc.set_property("estimated-bitrate", self.bitrate * 1000)
        self.gcc.connect("notify::estimated-bitrate", self.on_estimated_bitrate_changed)
        return self.gcc
