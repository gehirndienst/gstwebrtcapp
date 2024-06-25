"""
app.py

Description: An application that uses GStreamer's webrtcbin plugin to stream the video source
to the AhoyMedia webrtc engine maintained by ADDIX GmbH.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

import asyncio
from typing import Any, List
import gi

gi.require_version("Gst", "1.0")
gi.require_version('GstRtp', '1.0')
gi.require_version('GstWebRTC', '1.0')
from gi.repository import Gst
from gi.repository import GstRtp
from gi.repository import GstWebRTC

from gstwebrtcapp.apps.app import GstWebRTCApp, GstWebRTCAppConfig
from gstwebrtcapp.apps.pipelines import DEFAULT_BIN_PIPELINE
from gstwebrtcapp.utils.base import GSTWEBRTCAPP_EXCEPTION, LOGGER
from gstwebrtcapp.utils.gst import DEFAULT_GCC_SETTINGS, DEFAULT_TRANSCIEVER_SETTINGS, get_app_transceiver_properties
from gstwebrtcapp.utils.webrtc import TWCC_URI


class AhoyApp(GstWebRTCApp):
    """
    An application that uses GStreamer's WEBRTCBIN plugin to stream the video source to the AhoyMedia WebRTC client.
    """

    def __init__(
        self,
        config: GstWebRTCAppConfig = GstWebRTCAppConfig(pipeline_str=DEFAULT_BIN_PIPELINE),
    ):
        self.pipeline = None
        self.webrtcbin = None
        self.source = None
        self.raw_caps = None
        self.raw_capsfilter = None
        self.encoder = None
        self.pay_capsfilter = None
        self.transceivers = []
        self.gcc = None
        self.gcc_estimated_bitrates = asyncio.Queue()
        self.bus = None

        super().__init__(config)

    def _init_pipeline(self) -> None:
        LOGGER.info(f"OK: initializing pipeline from a string {self.pipeline_str}...")
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        if not self.pipeline:
            raise GSTWEBRTCAPP_EXCEPTION(f"can't create pipeline from {self.pipeline_str}")

        # webrtcbin
        self.webrtcbin = self.pipeline.get_by_name("webrtc")
        if self.is_webrtc_ready():
            LOGGER.info("OK: webrtcbin is found in the pipeline")
            for dc_cfg in self.data_channels_cfgs:
                self.create_data_channel(dc_cfg["name"], dc_cfg["options"], dc_cfg["callbacks"])
        else:
            raise GSTWEBRTCAPP_EXCEPTION("can't find webrtcbin in the pipeline")

        # collect main named elems
        self.source = self.pipeline.get_by_name("source")
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

        # set video source location
        if self.source.get_property("location") is not None:
            # NOTE: only sources with location property are supported now (Gst plugins of *src group)
            self.source.set_property("location", self.video_url)
            LOGGER.info(f"OK: video location is set to {self.video_url}")

        # set delayed caps for raw_capsfilter to safely change resolution and framerate on the fly
        self.raw_capsfilter.set_property("caps-change-mode", "delayed")

        # set gcc estimator if settings are provided
        if self.gcc_settings is not None:
            self.set_gcc()

        # get webrtcbin transceivers
        self.webrtcbin.connect('on-new-transceiver', self._cb_on_new_transceiver)
        self.get_transceivers()

        # set priority (DSCP marking)
        self.set_priority(self.priority)

        # set resolution, framerate, bitrate and fec percentage
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
        LOGGER.info("OK: pipeline is PLAYING")

    def _post_init_pipeline(self) -> None:
        pass

    def set_gcc(self) -> None:
        # add rtpgccbwe element and enable twcc RTP extension for the payloader
        self.gcc = Gst.ElementFactory.make("rtpgccbwe")
        if not self.gcc:
            raise GSTWEBRTCAPP_EXCEPTION("Can't create rtpgccbwe")
        twcc_ext = GstRtp.RTPHeaderExtension.create_from_uri(TWCC_URI)
        twcc_ext.set_id(1)
        self.payloader.emit("add-extension", twcc_ext)
        self.webrtcbin.connect("request-aux-sender", self._cb_add_gcc)
        self.webrtcbin.connect('deep-element-added', lambda _, __, ___: None)
        LOGGER.info("OK: gcc is set")

    def get_transceivers(self) -> List[GstWebRTC.WebRTCRTPTransceiver]:
        # get transceivers from webrtcbin and set NACK and FEC properties
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
                        self.set_app_transceiver_properties(
                            transceiver=transceiver,
                            props_dict=get_app_transceiver_properties(self.transceiver_settings),
                        )
                        self.transceivers.append(transceiver)
                        index += 1
                    else:
                        break
                if len(self.transceivers) > 0:
                    LOGGER.info(f"OK: got {len(self.transceivers)} transceivers from webrtcbin")
                else:
                    raise GSTWEBRTCAPP_EXCEPTION("can't get any single transceiver from webrtcbin")
            return self.transceivers

    def set_priority(self, priority: int) -> None:
        # set priority to the sender. Corresponds to 8, 0, 36, 38 DSCP values.
        match priority:
            case 1:
                wrtc_priority_type = GstWebRTC.WebRTCPriorityType.VERY_LOW
            case 2:
                wrtc_priority_type = GstWebRTC.WebRTCPriorityType.LOW
            case 3:
                wrtc_priority_type = GstWebRTC.WebRTCPriorityType.MEDIUM
            case 4:
                wrtc_priority_type = GstWebRTC.WebRTCPriorityType.HIGH
            case _:
                wrtc_priority_type = GstWebRTC.WebRTCPriorityType.LOW
        sender = self.transceivers[0].get_property("sender")
        if sender is not None:
            sender.set_priority(wrtc_priority_type)
            LOGGER.info(f"OK: set priority (DSCP marking) to {priority}, min 1, max 4")
        else:
            raise GSTWEBRTCAPP_EXCEPTION("can't set priority, sender is None")

    def get_raw_caps(self) -> Gst.Caps:
        s = f"video/x-raw,format=I420,width={self.resolution['width']},height={self.resolution['height']},framerate={self.framerate}/1,"
        return Gst.Caps.from_string(s)

    def set_bitrate(self, bitrate_kbps: int) -> bool:
        if not self.encoder:
            return False
        if self.encoder_gst_name.startswith("nv") or self.encoder_gst_name.startswith("x26"):
            self.encoder.set_property("bitrate", bitrate_kbps)
        elif self.encoder_gst_name.startswith("vp") or self.encoder_gst_name.startswith("av1"):
            self.encoder.set_property("target-bitrate", bitrate_kbps * 1000)
        else:
            raise GSTWEBRTCAPP_EXCEPTION(f"encoder {self.encoder_gst_name} is not supported")

        self.bitrate = bitrate_kbps
        return True

    def set_resolution(self, width: int, height: int) -> bool:
        if not self.raw_capsfilter:
            return False
        self.resolution = {"width": width, "height": height}
        self.raw_caps = self.get_raw_caps()
        self.raw_capsfilter.set_property("caps", self.raw_caps)
        return True

    def set_framerate(self, framerate: int) -> bool:
        if not self.raw_capsfilter:
            return False
        self.framerate = framerate
        self.raw_caps = self.get_raw_caps()
        self.raw_capsfilter.set_property("caps", self.raw_caps)
        return True

    def set_fec_percentage(self, percentage: int, index: int = -1) -> bool:
        if not self.transceivers:
            LOGGER.error("ERROR: there is no transceivers in the pipeline")
            return False
        if index > 0:
            try:
                transceiver = self.transceivers[index]
                transceiver.set_property("fec-percentage", percentage)
            except IndexError:
                LOGGER.error(f"ERROR: can't find tranceiver with index {index}")
                return False
        else:
            for transceiver in self.transceivers:
                transceiver.set_property("fec-percentage", percentage)

        self.fec_percentage = percentage
        return True

    def set_encoder_param(self, param: str, value: Any) -> None:
        prop = self.encoder.get_property(param)
        if prop is not None:
            r = self.source.set_state(Gst.State.PAUSED)
            if r != Gst.StateChangeReturn.SUCCESS:
                raise GSTWEBRTCAPP_EXCEPTION("set_encoder_param: unable to set the source to the paused state")
            r = self.encoder.set_state(Gst.State.READY)
            if r != Gst.StateChangeReturn.SUCCESS:
                raise GSTWEBRTCAPP_EXCEPTION("set_encoder_param: unable to set the encoder to the ready state")
            self.encoder.set_property(param, value)
            r = self.source.set_state(Gst.State.PLAYING)
            if r != Gst.StateChangeReturn.SUCCESS:
                raise GSTWEBRTCAPP_EXCEPTION("set_encoder_param: unable to set the source to the playing state")
            r = self.encoder.set_state(Gst.State.PLAYING)
            if r != Gst.StateChangeReturn.SUCCESS:
                raise GSTWEBRTCAPP_EXCEPTION("set_encoder_param: unable to set the encoder to the playing state")
            LOGGER.info(f"ACTION: set encoder param {param} from {prop} to {self.encoder.get_property(param)}")
        else:
            raise GSTWEBRTCAPP_EXCEPTION(f"Encoder {self.encoder} doesn't have property {param}")

    def _cb_on_new_transceiver(self, _, transceiver) -> None:
        if transceiver:
            self.set_app_transceiver_properties(
                transceiver=transceiver,
                props_dict=get_app_transceiver_properties(self.transceiver_settings),
            )
            self.transceivers.append(transceiver)
            LOGGER.info(f"OK: got a new transceiver from webrtcbin, total {len(self.transceivers)}")

    def _cb_add_gcc(self, _, __) -> Gst.Element:
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
        self.gcc.connect("notify::estimated-bitrate", self._on_estimated_bitrate_changed)
        return self.gcc

    def _on_estimated_bitrate_changed(self, bwe, pspec) -> None:
        if bwe and pspec.name == "estimated-bitrate":
            estimated_bitrate = self.gcc.get_property(pspec.name)
            self.gcc_estimated_bitrates.put_nowait(estimated_bitrate)
        else:
            raise GSTWEBRTCAPP_EXCEPTION("Can't get estimated bitrate by gcc")
