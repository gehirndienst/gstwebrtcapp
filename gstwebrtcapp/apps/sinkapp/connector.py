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
from typing import List
import gi


gi.require_version('Gst', '1.0')
from gi.repository import Gst

from apps.app import GstWebRTCAppConfig
from apps.sinkapp.app import SinkApp
from control.agent import Agent, AgentType
from control.safety.switcher import SwitchingPair
from media.preset import get_video_preset
from message.client import MqttConfig, MqttPair, MqttPublisher, MqttSubscriber
from network.controller import NetworkController
from utils.base import LOGGER, async_wait_for_condition
from utils.gst import stats_to_dict


class SinkConnector:
    def __init__(
        self,
        server: str = "ws://127.0.0.1:8443",
        pipeline_config: GstWebRTCAppConfig = GstWebRTCAppConfig(),
        agents: List[Agent] | None = None,
        feed_name: str = "gst-stream",
        mqtt_config: MqttConfig = MqttConfig(),
        network_controller: NetworkController | None = None,
        switching_pair: SwitchingPair | None = None,
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

        self.agents = {}
        self.agent_threads = {}
        self.switching_pair = switching_pair
        self.is_switching = False
        self._prepare_agents(agents)

        self.mqtt_config = mqtt_config
        self.mqtts = MqttPair(
            publisher=MqttPublisher(self.mqtt_config),
            subscriber=MqttSubscriber(self.mqtt_config),
        )
        self.mqtts_threads = []
        self.feed_name = feed_name
        self.network_controller = network_controller

        self._app = None
        self.is_running = False

    async def webrtc_coro(self) -> None:
        try:
            self._app = SinkApp(self.pipeline_config)
            if self._app is None:
                raise Exception("SinkApp object is None!")
            self._app.webrtcsink.set_property("meta", Gst.Structure.new_from_string(f"meta,name={self.feed_name}"))
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
            if self.agents:
                # start agent threads
                for agent in self.agents.values():
                    agent_thread = threading.Thread(target=agent.run, args=(True,), daemon=True)
                    agent_thread.start()
                    self.agent_threads[agent.id] = agent_thread
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
            if self.agent_threads:
                self._terminate_agents()
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
                                # add 5% policy: if bitrate difference is less than 5% then don't change it
                                if abs(self._app.bitrate - msg[action]) / self._app.bitrate > 0.05:
                                    self._app.set_bitrate(msg[action])
                            case "resolution":
                                self._app.set_resolution(msg[action])
                            case "framerate":
                                self._app.set_framerate(msg[action])
                            case "preset":
                                self._app.set_preset(get_video_preset(msg[action]))
                            case "switch":
                                self._switch_agents(msg[action])
                            case _:
                                LOGGER.error(f"ERROR: Unknown action in the message: {msg}")

    async def handle_bandwidth_estimations(self) -> None:
        LOGGER.info(f"OK: BANDWIDTH ESTIMATIONS HANDLER IS ON -- ready to publish bandwidth estimations")
        while self.is_running:
            gcc_bw = await self._app.gcc_estimated_bitrates.get()
            self.mqtts.publisher.publish(self.mqtt_config.topics.gcc, str(gcc_bw))
        LOGGER.info(f"OK: BANDWIDTH ESTIMATIONS HANDLER IS OFF!")

    def _prepare_agents(self, agents: List[Agent] | None) -> None:
        if agents is not None:
            self.agents = {agent.id: agent for agent in agents}

            if self.switching_pair is None:
                return
            if (
                AgentType.SAFETY_DETECTOR in [agent.type for agent in self.agents.values()]
                and self.switching_pair is None
            ):
                LOGGER.error("ERROR: SafetyDetectorAgent has no corresponding switching pair. Switching is off")
                return
            if not all(
                agent_id in self.agents.keys()
                for agent_id in [self.switching_pair.safe_id, self.switching_pair.unsafe_id]
            ):
                LOGGER.error("ERROR: Swithing pair contains invalid agent ids. Switching is off")
                return

            self.is_switching = True
            LOGGER.info("INFO: SafetyAgentDetector is on, switching is enabled!")

    def _switch_agents(self, algo: int) -> None:
        if self.is_switching:
            is_first_switch = False  # if true then do not restart the agents as they are already running
            if not self.switching_pair.is_warmups_resetted:
                # reset warmups
                self.agents[self.switching_pair.safe_id].warmup = 0.0
                self.agents[self.switching_pair.unsafe_id].warmup = 0.0
                self.switching_pair.is_warmups_resetted = True
                is_first_switch = True

            if algo == 0:
                # enable gcc, disable drl
                if not is_first_switch:
                    self.agent_threads[self.switching_pair.safe_id] = threading.Thread(
                        target=self.agents[self.switching_pair.safe_id].run, args=(True,), daemon=True
                    )
                    self.agent_threads[self.switching_pair.safe_id].start()
                else:
                    self.agents[self.switching_pair.safe_id].enable_actions()
                self.agents[self.switching_pair.unsafe_id].stop()
                self.agent_threads[self.switching_pair.unsafe_id].join()
            else:
                # enable drl, disable gcc
                if not is_first_switch:
                    self.agent_threads[self.switching_pair.unsafe_id] = threading.Thread(
                        target=self.agents[self.switching_pair.unsafe_id].run, args=(True,), daemon=True
                    )
                    self.agent_threads[self.switching_pair.unsafe_id].start()
                self.agents[self.switching_pair.safe_id].stop()
                self.agent_threads[self.switching_pair.safe_id].join()

    def _terminate_agents(self) -> None:
        if self.agent_threads:
            for agent in self.agents.values():
                agent.stop()
            for agent_thread in self.agent_threads.values():
                if agent_thread:
                    agent_thread.join()

    @property
    def app(self) -> SinkApp | None:
        return self._app
