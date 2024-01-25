"""
connector.py

Description: A connector that connects to the ADDIX AhoyMedia WebRTC client and handles the WebRTC signalling.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
import asyncio
from collections import deque
from datetime import datetime
import json
import requests
import threading

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstWebRTC', '1.0')
gi.require_version('GstSdp', '1.0')
from gi.repository import GstSdp
from gi.repository import GstWebRTC
from gi.repository import Gst

from ahoyapp.app import GstWebRTCBinApp, GstWebRTCBinAppConfig
from control.agent import Agent
from utils.base import LOGGER, wait_for_condition, async_wait_for_condition
from utils.gst import stats_to_dict


class AhoyConnector:
    """
    A connector that connects to the ADDIX AhoyMedia WebRTC client and handles the WebRTC signalling.

    :param str server: AhoyMedia server URL.
    :param str api_key: AhoyMedia API key.
    :param GstWebRTCBinAppConfig pipeline_config: Configuration for the GStreamer WebRTC pipeline.
    :param Agent agent: AI-enhanced agent that optionally controls the quality of the video via GStreamer. Nullable.
    :param str feed_name: Feed name for the connection.
    :param str signalling_channel_name: Name of the signalling channel.
    :param str stats_channel_name: Name of the stats channel.
    :param float stats_update_interval: Interval for requesting/receiving new webrtc stats.
    """

    def __init__(
        self,
        server: str,
        api_key: str,
        pipeline_config: GstWebRTCBinAppConfig = GstWebRTCBinAppConfig(),
        agent: Agent | None = None,
        feed_name: str = "gstreamerwebrtcapp",
        signalling_channel_name: str = "control",
        stats_channel_name: str = "telemetry",
        stats_update_interval: float = 1.0,
    ):
        self.server = server
        self.api_key = api_key
        self.feed_name = feed_name

        # pc_out is used to send SDP offer to Ahoy and receive SDP answer to set the signalling data channel
        self.pc_out = RTCPeerConnection(RTCConfiguration([RTCIceServer("stun:stun.l.google.com:19302")]))
        self.signalling_channel = self.pc_out.createDataChannel(signalling_channel_name, ordered=True)
        self.stats_channel = self.pc_out.createDataChannel(stats_channel_name, ordered=True)

        # internal webrtc
        self._app = None
        self.pipeline_config = pipeline_config
        self.payload_type = None
        self.webrtcbin_sdp = None
        self.webrtcbin_ice_connection_state = GstWebRTC.WebRTCICEConnectionState.NEW
        self.pc_out_ice_connection_state = "new"

        self.agent = agent
        self.agent_thread = None
        self.ahoy_stats = deque(maxlen=10000)
        self.webrtcbin_stats = deque(maxlen=10000)
        self.stats_update_interval = stats_update_interval

        self.is_running = False
        self.is_locked = True

        self.webrtc_coro_control_task = None

    async def connect_coro(self) -> None:
        LOGGER.info(f"OK: connecting to AhoyMedia...")

        # set handlers for pc_out
        self._set_pc_out_handlers()

        # create local offer
        local_offer = await self.pc_out.createOffer()
        await self.pc_out.setLocalDescription(local_offer)

        # send request to ahoy director
        headers = {
            "accept": "application/json",
            "x-api-key": self.api_key,
            "content-type": "application/json",
        }
        data = {
            "sdp": self.pc_out.localDescription.sdp,
            "candidates": "",
            "capabilities": {},
            "name": self.feed_name,
        }
        request = requests.post(
            self.server + requests.utils.quote(self.feed_name),
            headers=headers,
            data=json.dumps(data),
        )
        LOGGER.info(f"INFO: connect, request ... {request}")
        LOGGER.info(f"INFO: connect, request.status_code ... {request.status_code}")

        # retrieve response
        response = request.json()
        LOGGER.info(f"INFO: connect_pc_out, response ... {response}")

        # create sdp with candidates
        sdp_lines = response["sdp"].split("\r\n")[:-1]
        sdp_candidates = [f"a={candidate['candidate']}" for candidate in response["candidates"]]
        remote_sdp = "\r\n".join([*sdp_lines, *sdp_candidates, ""])
        LOGGER.info(f"INFO: connect_pc_out, remote_sdp ... {remote_sdp}")

        # create remote answer
        remote_answer = RTCSessionDescription(type="answer", sdp=remote_sdp)
        await self.pc_out.setRemoteDescription(remote_answer)
        LOGGER.info(f"OK: webrtc signalling is finished, waiting for messages on signalling channel...")

    def _set_pc_out_handlers(self) -> None:
        @self.pc_out.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            LOGGER.info(f"INFO: on_connectionstatechange, pc_out ... {self.pc_out.connectionState}")
            self.pc_out_ice_connection_state = self.pc_out.connectionState

        @self.signalling_channel.on("message")
        async def on_message(msg) -> None:
            msg = json.loads(msg)
            if "streamStartRequest" in msg:
                # streamStartRequest is received when the stream is played on the Ahoy side
                if msg["streamStartRequest"]["feedUuid"] == self.feed_name:
                    LOGGER.info(f"INFO: SIGNALLING CHANNEL received streamStartRequest {msg}")

                    self.is_running = True
                    self.is_locked = False

                    streamStartResponse = {
                        "streamStartResponse": {
                            "success": True,
                            "uuid": msg["streamStartRequest"]["uuid"],
                            "feedUuid": msg["streamStartRequest"]["feedUuid"],
                        }
                    }
                    data = {
                        "message": {
                            "to": f"director.api_{self.api_key}",
                            "payload": streamStartResponse,
                        }
                    }
                    self.signalling_channel.send(json.dumps(data))
            elif "sdpRequest" in msg:
                # sdpRequest is received next after streamStartRequest to set up the WebRTC connection
                if not self.is_locked:
                    self.is_locked = True
                    LOGGER.info(f"INFO: SIGNALLING CHANNEL received sdpRequest {msg}")

                    self._on_received_sdp_request(msg["sdpRequest"]["sdp"])

                    wait_for_condition(lambda: self.webrtcbin_sdp is not None, self._app.max_timeout)
                    LOGGER.info(
                        f"INFO: on_message, succesfully created local answer for webrtcbin on incoming SDP request..."
                    )

                    # force goog-remb to be added to the answer
                    self.webrtcbin_sdp.add_attribute('rtcp-fb', f'{self.payload_type} goog-remb')

                    candidates = msg["sdpRequest"]["candidates"]
                    for candidate in candidates:
                        LOGGER.info(f"INFO: on_message, adding ice candidate ... {candidate['candidate']}")
                        self._app.webrtcbin.emit(
                            'add-ice-candidate', candidate['sdpMLineIndex'], candidate['candidate']
                        )

                    sdpResponse = {
                        "sdpResponse": {
                            "success": True,
                            "uuid": msg["sdpRequest"]["uuid"],
                            "sdp": self.webrtcbin_sdp.as_text(),
                        }
                    }
                    data = {"message": {"to": f"director.api_{self.api_key}", "payload": sdpResponse}}
                    self.signalling_channel.send(json.dumps(data))
            elif "streamStopRequest" in msg:
                # streamStopRequest is received when the stream is stopped on the Ahoy side
                if msg["streamStopRequest"]["feedUuid"] == self.feed_name:
                    LOGGER.info(f"INFO: SIGNALLING CHANNEL received streamStopRequest {msg}")
                    self.terminate_webrtc_coro(is_restart_webrtc_coro=True)
            else:
                LOGGER.info(f"INFO: SIGNALLING CHANNEL received currently unhandled message {msg}")

        @self.stats_channel.on("message")
        async def on_message(msg) -> None:
            # TODO: currently nothing comes from Ahoy to trigger this handler
            LOGGER.info(f"INFO: STATS CHANNEL received stats message {msg}")
            self.ahoy_stats.append((datetime.now().strftime('%H:%M:%S.%f')[:-3], msg))

    def _on_received_sdp_request(self, sdp) -> None:
        LOGGER.info(f"INFO: _on_received_sdp_request callback, processing the incoming SDP request...")
        res, sdpmsg = GstSdp.SDPMessage.new_from_text(sdp)
        if res < 0:
            LOGGER.error(f"ERROR: _on_received_sdp_request callback, failed to parse remote offer SDP")
            self.terminate_webrtc_coro()

        # NOTE: the app (GstPipeline) starts first when the video content is requested. Before that this object is None
        try:
            self._app = GstWebRTCBinApp(self.pipeline_config)
        except Exception as e:
            LOGGER.error(
                "ERROR: _on_received_sdp_request callback, failed to create GstWebRTCBinApp object"
                f", reason: {str(e)}..."
            )
            self.terminate_webrtc_coro()
        wait_for_condition(lambda: self._app.is_webrtc_ready(), self._app.max_timeout)
        self._app.webrtcbin.connect('on-negotiation-needed', lambda _: None)
        self._app.webrtcbin.connect('notify::ice-connection-state', self._on_ice_connection_state_notify)

        # set remote offer and create answer
        remote_offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdpmsg)
        promise = Gst.Promise.new_with_change_func(self._on_offer_set, None, None)
        self._app.webrtcbin.emit('set-remote-description', remote_offer, promise)

    def _on_offer_set(self, promise, _, __) -> None:
        assert promise.wait() == Gst.PromiseResult.REPLIED, "FAIL: offer set promise was not replied"
        promise = Gst.Promise.new_with_change_func(self._on_answer_created, None, None)
        self._app.webrtcbin.emit('create-answer', None, promise)

    def _on_answer_created(self, promise, _, __) -> None:
        assert promise.wait() == Gst.PromiseResult.REPLIED, "FAIL: create answer promise was not replied"
        reply = promise.get_reply()
        answer = reply.get_value('answer')
        answer_sdp = answer.sdp
        for i in range(0, answer_sdp.medias_len()):
            media = answer_sdp.get_media(i)
            for j in range(0, media.attributes_len()):
                attr = media.get_attribute(j)
                if attr.key == 'recvonly':
                    LOGGER.info(f"INFO: _on_answer_created callback, found recvonly attribute")
                    media.remove_attribute(j)
                    attr = GstSdp.SDPAttribute()
                    attr.set('sendrecv', attr.value)
                    media.insert_attribute(j, attr)
                    LOGGER.info(f"INFO: _on_answer_created callback, changed recvonly to sendrecv")
                if attr.key == 'fmtp':
                    self.payload_type = int(attr.value.split(' ')[0])
                    LOGGER.info(f"INFO: _on_answer_created callback, found payload type {self.payload_type}")
        promise = Gst.Promise.new()
        self._app.webrtcbin.emit('set-local-description', answer, promise)
        promise.interrupt()
        self.webrtcbin_sdp = answer.sdp
        LOGGER.info(f"INFO: _on_answer_created callback, successfully created a local answer for webrtcbin")

    def _on_ice_connection_state_notify(self, pspec, _) -> None:
        self.webrtcbin_ice_connection_state = self._app.webrtcbin.get_property('ice-connection-state')
        LOGGER.info(f"INFO: webrtcbin's ICE connecting state has been changed to {self.webrtcbin_ice_connection_state}")
        if self.webrtcbin_ice_connection_state == GstWebRTC.WebRTCICEConnectionState.CONNECTED:
            LOGGER.info("OK: ICE connection is established")

    def _on_get_webrtcbin_stats(self, promise, _, __) -> None:
        assert promise.wait() == Gst.PromiseResult.REPLIED, "FAIL: get webrtcbin stats promise was not replied"
        stats = {}
        stats_struct = promise.get_reply()
        if stats_struct.n_fields() > 0:
            session_struct_n_fields = stats_struct.n_fields()
            for i in range(session_struct_n_fields):
                stat_name = stats_struct.nth_field_name(i)
                stat_value = stats_struct.get_value(stat_name)
                if isinstance(stat_value, Gst.Structure):
                    stats[stat_name] = stats_to_dict(stat_value.to_string())
        else:
            LOGGER.error(f"ERROR: no stats to save...")
        # write only stats that are received after RTP is set up
        if self.agent is not None:
            # if agent is attached, push stats to its controller
            self.agent.controller.push_observation(stats)
        else:
            # if no agent just push stats to the internal queue
            self.webrtcbin_stats.append(stats)

    async def handle_ice_connection(self) -> None:
        LOGGER.info(f"OK: ICE CONNECTION HANDLER IS ON -- ready to check for ICE connection state")
        while (
            self.webrtcbin_ice_connection_state != GstWebRTC.WebRTCICEConnectionState.FAILED
            and self.webrtcbin_ice_connection_state != GstWebRTC.WebRTCICEConnectionState.DISCONNECTED
            and self.webrtcbin_ice_connection_state != GstWebRTC.WebRTCICEConnectionState.CLOSED
            and self.pc_out_ice_connection_state != "failed"
            and self.pc_out_ice_connection_state != "disconnected"
            and self.pc_out_ice_connection_state != "closed"
        ):
            await asyncio.sleep(0.1)
        LOGGER.info(f"OK: ICE CONNECTION HANDLER IS OFF!")
        if self.is_running:
            LOGGER.error(
                f"ERROR: ICE connection has failed! Webrtcbin state is {self.webrtcbin_ice_connection_state}, "
                f"pc_out state is {self.pc_out_ice_connection_state}"
            )
            self.terminate_webrtc_coro()

    async def handle_webrtcbin_stats(self) -> None:
        LOGGER.info(f"OK: WEBRTCBIN STATS HANDLER IS ON -- ready to check for stats")
        while self.is_running:
            await asyncio.sleep(self.stats_update_interval)
            promise = Gst.Promise.new_with_change_func(self._on_get_webrtcbin_stats, None, None)
            self._app.webrtcbin.emit('get-stats', None, promise)
        LOGGER.info(f"OK: GST STATS HANDLER IS OFF!")

    async def webrtc_coro(self) -> None:
        while not (self._app and self._app.is_running):
            await asyncio.sleep(0.1)

        tasks = []
        try:
            LOGGER.info(f"OK: main webrtc coroutine has been started!")
            ######################################## TASKS ########################################
            signalling_task = asyncio.create_task(self.handle_ice_connection())
            pipeline_task = asyncio.create_task(self._app.handle_pipeline())
            webrtcbin_stats_task = asyncio.create_task(self.handle_webrtcbin_stats())
            tasks = [signalling_task, pipeline_task, webrtcbin_stats_task]
            if self.agent is not None:
                # start agent's controller and agent's thread
                controller_task = asyncio.create_task(self.agent.controller.handle_actions(self._app))
                self.agent_thread = threading.Thread(target=self.agent.run, args=(True,), daemon=True)
                self.agent_thread.start()
                tasks.append(controller_task)
            #######################################################################################
            self.webrtc_coro_control_task = asyncio.create_task(asyncio.sleep(float('inf')))
            tasks = [self.webrtc_coro_control_task, *tasks]
            await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError:
            if not self.is_running:
                # this is on streamStopRequest where self.is_running was set to False.
                # In this case webrtc_coro will be recursively rescheduled.
                try:
                    await async_wait_for_condition(lambda: not self._app.is_running, 5)
                except Exception:
                    pass
                for task in tasks:
                    task.cancel()
                if self.agent_thread is not None:
                    self.agent.stop()
                    self.agent_thread.join()
                self._app = None
                LOGGER.info("OK: main webrtc coroutine is stopped on streamStopRequest, pending...")
                return await self.webrtc_coro()
            else:
                # this is on some internal exception where self.is_running is True and terminate_webrtc_coro was trigerred.
                # In this case webrtc_coro will be stopped.
                try:
                    await async_wait_for_condition(lambda: not self._app.is_running, 5)
                except Exception:
                    pass
                for task in tasks:
                    task.cancel()
                if self.agent_thread is not None:
                    self.agent.stop()
                    self.agent_thread.join()
                self._app = None
                LOGGER.error(
                    "ERROR: main webrtc coroutine has been interrupted due to an internal exception, stopping..."
                )
                return
        except Exception as e:
            # this is on some external exception where terminate_webrtc_coro was not trigerred e.g., pipeline error msg.
            # TODO: maybe also try to rerun it in case of pipeline error? since this may be related to the network link failure
            self.terminate_webrtc_coro()
            for task in tasks:
                task.cancel()
            if self.agent_thread is not None:
                self.agent.stop()
                self.agent_thread.join()
            self._app = None
            LOGGER.error(
                "ERROR: main webrtc coroutine has been unexpectedly interrupted due to an exception:"
                f" '{str(e)}', stopping..."
            )
            return

    def terminate_webrtc_coro(self, is_restart_webrtc_coro: bool = False) -> None:
        self.is_running = not is_restart_webrtc_coro
        self.webrtcbin_sdp = None
        if self._app is not None:
            if self._app.is_running:
                self._app.send_termination_message_to_bus()
            else:
                self._app.terminate_pipeline()
        if self.webrtc_coro_control_task is not None:
            self.webrtc_coro_control_task.cancel()

    @property
    def app(self) -> GstWebRTCBinApp | None:
        return self._app
