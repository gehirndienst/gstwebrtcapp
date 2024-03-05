"""
connector.py

Description: A connector that connects SinkApp to the browser JS client. It holds the main coroutine for the SinkApp.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

import asyncio
from collections import deque
import json
import re
import threading
import gi


gi.require_version('Gst', '1.0')
from gi.repository import Gst

from apps.app import GstWebRTCAppConfig
from apps.sinkapp.app import SinkApp
from control.agent import Agent
from message.client import MqttConfig, MqttPair, MqttPublisher, MqttSubscriber
from network.controller import NetworkController
from utils.base import LOGGER, async_wait_for_condition
from utils.gst import stats_to_dict


class SinkConnector:
    def __init__(
        self,
        server: str = "ws://127.0.0.1:8443",
        pipeline_config: GstWebRTCAppConfig = GstWebRTCAppConfig(),
        agent: Agent | None = None,
        mqtt_config: MqttConfig = MqttConfig(),
        network_controller: NetworkController | None = None,
    ):
        self.pipeline_config = pipeline_config
        if 'signaller::uri' in self.pipeline_config.pipeline_str:
            self.pipeline_config.pipeline_str, _ = re.subn(
                r'(signaller::uri=)[^ ]*', f'\\1{server}', self.pipeline_config.pipeline_str
            )
        else:
            self.pipeline_config.pipeline_str, _ = re.subn(
                r'(webrtcsink[^\n]*)', fr'\1 signaller::uri={server}', self.pipeline_config.pipeline_str
            )

        self.agent = agent
        self.agent_thread = None
        self.mqtt_config = mqtt_config
        if self.agent is not None:
            self.agent.mqtt_config = mqtt_config
        self.mqtts = MqttPair(
            publisher=MqttPublisher(self.mqtt_config),
            subscriber=MqttSubscriber(self.mqtt_config),
        )
        self.mqtts_threads = None
        self.network_controller = network_controller

        self._app = None
        self.webrtcbin_stats = deque(maxlen=10000)

        self.is_running = False

    async def webrtc_coro(self) -> None:
        try:
            self._app = SinkApp(self.pipeline_config)
            if self._app is None:
                raise Exception("SinkApp object is None!")
            self.is_running = True
        except Exception as e:
            LOGGER.error(f"ERROR: Failed to create SinkApp object, reason:\n {str(e)}")
            self.terminate_webrtc_coro()
            return

        tasks = []
        try:
            LOGGER.info(f"OK: main webrtc coroutine has been started!")
            self.mqtts_threads = [
                threading.Thread(target=self.mqtts.publisher.run, daemon=True).start(),
                threading.Thread(target=self.mqtts.subscriber.run, daemon=True).start(),
            ]
            self.mqtts.subscriber.subscribe([self.mqtt_config.topics.actions])
            ######################################## TASKS ########################################
            pipeline_task = asyncio.create_task(self._app.handle_pipeline())
            post_init_pipeline_task = asyncio.create_task(self.handle_post_init_pipeline())
            webrtcsink_stats_task = asyncio.create_task(self.handle_webrtcsink_stats())
            actions_task = asyncio.create_task(self.handle_actions())
            be_task = asyncio.create_task(self.handle_bandwidth_estimations())
            tasks = [pipeline_task, post_init_pipeline_task, webrtcsink_stats_task, actions_task, be_task]
            if self.agent is not None:
                # start agents thread
                self.agent_thread = threading.Thread(target=self.agent.run, args=(True,), daemon=True)
                self.agent_thread.start()
            if self.network_controller is not None:
                # start network controller's task
                network_controller_task = asyncio.create_task(self.network_controller.update_network_rule())
                tasks.append(network_controller_task)
            #######################################################################################
            await asyncio.gather(*tasks)

        except Exception as e:
            self.terminate_webrtc_coro()
            for task in tasks:
                task.cancel()
            if self.agent_thread is not None:
                self.agent.stop()
                self.agent_thread.join()
            self.mqtts.publisher.stop()
            self.mqtts.subscriber.stop()
            for t in self.mqtts_threads:
                if t:
                    t.join()
            self._app = None
            LOGGER.error(
                "ERROR: main webrtc coroutine has been unexpectedly interrupted due to an exception:"
                f" '{str(e)}', stopping..."
            )
            return

    async def handle_post_init_pipeline(self) -> None:
        await async_wait_for_condition(lambda: self._app.bus is not None, timeout_sec=5.0)
        await async_wait_for_condition(lambda: self._app.webrtcsink_elements, timeout_sec=self._app.max_timeout)
        self._app.send_post_init_message_to_bus()

    def terminate_webrtc_coro(self) -> None:
        self.is_running = False
        if self._app is not None:
            if self._app.is_running:
                self._app.send_termination_message_to_bus()
            else:
                self._app.terminate_pipeline()

    async def handle_webrtcsink_stats(self) -> None:
        LOGGER.info(f"OK: WEBRTCSINK STATS HANDLER IS ON -- ready to check for stats")
        while self.is_running:
            await asyncio.sleep(0.1)
            stats = {}
            stats_struct = self.app.webrtcsink.get_property("stats")
            if stats_struct.n_fields() > 0:
                session_name = stats_struct.nth_field_name(0)
                if session_name is not None:
                    session_struct = stats_struct.get_value(session_name)
                    session_struct_n_fields = session_struct.n_fields()
                    for i in range(session_struct_n_fields):
                        stat_name = session_struct.nth_field_name(i)
                        stat_value = session_struct.get_value(stat_name)
                        if isinstance(stat_value, Gst.Structure):
                            stats[stat_name] = stats_to_dict(stat_value.to_string())
            # push the stats to the agent's controller queue or save it to an own one
            if stats:
                self.mqtts.publisher.publish(self.mqtt_config.topics.stats, json.dumps(stats))

        LOGGER.info(f"OK: WEBRTCSINK STATS HANDLER IS OFF!")

    async def handle_actions(self) -> None:
        LOGGER.info(f"OK: ACTIONS HANDLER IS ON -- ready to pick and apply actions")
        while self.is_running:
            action_msg = await self.mqtts.subscriber.message_queues[self.mqtt_config.topics.actions].get()
            msg = json.loads(action_msg.msg)
            if self._app is not None and len(msg) > 0:
                for action in msg:
                    if msg.get(action) is None:
                        LOGGER.error(f"ERROR: Action {action} has no value!")
                        continue
                    else:
                        match action:
                            case "bitrate":
                                # add 10% policy: if bitrate difference is less than 10% then don't change it
                                if abs(self._app.bitrate - msg[action]) / self._app.bitrate > 0.1:
                                    self._app.set_bitrate(msg[action])
                            case "resolution":
                                self._app.set_resolution(msg[action])
                            case "framerate":
                                self._app.set_framerate(msg[action])
                            case _:
                                LOGGER.error(f"ERROR: Unknown action in the message: {msg}")

    async def handle_bandwidth_estimations(self) -> None:
        LOGGER.info(f"OK: BANDWIDTH ESTIMATIONS HANDLER IS ON -- ready to publish bandwidth estimations")
        while self.is_running:
            gcc_bw = await self._app.gcc_estimated_bitrates.get()
            self.mqtts.publisher.publish(self.mqtt_config.topics.gcc, str(gcc_bw))
        LOGGER.info(f"OK: BANDWIDTH ESTIMATIONS HANDLER IS OFF!")

    @property
    def app(self) -> SinkApp | None:
        return self._app
