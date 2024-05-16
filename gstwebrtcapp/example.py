'''
Test main functionalities by replacing the default one with one of the desired coroutine presented above in the main() endpoint.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>
'''

import asyncio
import threading

from apps.app import GstWebRTCAppConfig
from apps.ahoyapp.connector import AhoyConnector
from apps.pipelines import DEFAULT_BIN_PIPELINE, DEFAULT_BIN_CUDA_PIPELINE, DEFAULT_SINK_PIPELINE
from apps.sinkapp.connector import SinkConnector
from control.cc.gcc_agent import GccAgent
from control.drl.agent import DrlAgent
from control.drl.config import DrlConfig
from control.drl.mdp import ViewerSeqMDP
from control.recorder.agent import CsvViewerRecorderAgent
from control.safety.agent import SafetyDetectorAgent
from control.safety.switcher import SwitcherConfig, SwitchingPair
from message.broker import MosquittoBroker
from message.client import MqttConfig
from network.controller import NetworkController
from utils.base import LOGGER

try:
    import uvloop
except ImportError:
    uvloop = None

AHOY_DIRECTOR_URL = "..."
API_KEY = "..."
VIDEO_SOURCE = "rtsp://..."

MQTT_CFG = MqttConfig(
    id="local_instance",
    broker_port=1884,
)

APP_CFG = GstWebRTCAppConfig(
    pipeline_str=DEFAULT_BIN_PIPELINE,
    video_url=VIDEO_SOURCE,
    codec="h264",
    bitrate=2000,
    resolution={"width": 1280, "height": 720},
    gcc_settings={"min-bitrate": 400000, "max-bitrate": 10000000},
    is_cuda=False,
)


async def test_csv_recorder():
    # run it to test csv viewer recorder agent
    try:
        broker = MosquittoBroker(port=1884)
        broker_thread = threading.Thread(target=broker.run, daemon=True)
        broker_thread.start()

        stats_update_interval = 1.0

        agent = CsvViewerRecorderAgent(
            mqtt_config=MQTT_CFG,
            stats_update_interval=stats_update_interval,
            warmup=20.0,
            log_path="./logs",
            verbose=2,
        )

        conn = AhoyConnector(
            pipeline_config=APP_CFG,
            agents=[agent],
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="recorder_test",
            mqtt_config=MQTT_CFG,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

        broker.stop()
        broker_thread.join()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_drl():
    # run it to test drl agent train and eval
    try:
        broker = MosquittoBroker(port=1884)
        broker_thread = threading.Thread(target=broker.run, daemon=True)
        broker_thread.start()

        episodes = 10
        episode_length = 50
        stats_update_interval = 3.0  # sec

        # choose either train or eval configuration
        train_drl_cfg = DrlConfig(
            mode="train",
            model_name="sac",
            episodes=episodes,
            episode_length=episode_length,
            state_update_interval=stats_update_interval,
            state_max_inactivity_time=60.0,
            hyperparams_cfg={
                "policy": "MultiInputPolicy",
                "batch_size": 128,
                "ent_coef": "auto",
                "policy_kwargs": {"log_std_init": -1, "activation_fn": "relu", "net_arch": [256, 256]},
            },
            callbacks=['print_step'],
            save_model_path="./models",
            save_log_path="./logs",
            device="cpu",
            verbose=2,
        )

        eval_drl_cfg = DrlConfig(
            mode="eval",
            model_file="./models/fantastic_sb3_drl_model.zip",
            model_name="sac",
            episodes=episodes,
            episode_length=episode_length,
            state_update_interval=stats_update_interval,
            state_max_inactivity_time=60.0,
            deterministic=True,
        )

        mdp = ViewerSeqMDP(
            reward_function_name="qoe_ahoy_seq",
            episode_length=episode_length,
            num_observations_for_state=5,
            constants={
                "MAX_BITRATE_STREAM_MBPS": 10,
                "MAX_BANDWIDTH_MBPS": APP_CFG.gcc_settings["max-bitrate"] / 1000000,
                "MIN_BANDWIDTH_MBPS": APP_CFG.gcc_settings["min-bitrate"] / 1000000,
            },
        )

        drl_agent = DrlAgent(
            drl_config=train_drl_cfg,  # train by default
            mdp=mdp,
            mqtt_config=MQTT_CFG,
            warmup=20.0,
        )

        logger_agent = CsvViewerRecorderAgent(
            mqtt_config=MQTT_CFG,
            stats_update_interval=stats_update_interval,
            warmup=20.0,
            log_path="./logs",
            verbose=2,
        )

        conn = AhoyConnector(
            pipeline_config=APP_CFG,
            agents=[drl_agent, logger_agent],
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="drl_test",
            mqtt_config=MQTT_CFG,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

        broker.stop()
        broker_thread.join()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_network_controller():
    try:
        broker = MosquittoBroker(port=1884)
        broker_thread = threading.Thread(target=broker.run, daemon=True)
        broker_thread.start()

        network_controller = NetworkController(
            gt_bandwidth=10.0,
            interval=(20.0, 120.0),  # change to e.g., 3.0 to have a fixed interval
            interface="eth0",
            additional_rule_str="--delay 50ms",
            is_stop_after_no_rule=True,
            log_path="./logs",  # enables csv logging to the pointed folder, set to None to disable
            warmup=20.0,
        )

        network_controller.generate_rules(
            count=1000,
            weights=[0.7, 0.2, 0.1],
        )
        # # OR generate rules from traces
        # network_controller.generate_rules_from_traces(trace_folder="/home/traces")

        conn = AhoyConnector(
            pipeline_config=APP_CFG,
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="test_network_controller",
            mqtt_config=MQTT_CFG,
            network_controller=network_controller,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

        broker.stop()
        broker_thread.join()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_sink_app():
    # run to test sink app. NOTE: you need to have a running signalling server and JS client to test this.
    # Check: https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc?ref_type=heads#usage
    try:
        broker = MosquittoBroker(port=1884)
        broker_thread = threading.Thread(target=broker.run, daemon=True)
        broker_thread.start()

        app_cfg_sink = GstWebRTCAppConfig(
            pipeline_str=DEFAULT_SINK_PIPELINE,  # here is the difference
            video_url=VIDEO_SOURCE,
            codec="h264",
            bitrate=2000,
            resolution={"width": 1280, "height": 720},
            gcc_settings={"min-bitrate": 400000, "max-bitrate": 10000000},
            is_cuda=False,
        )

        conn = SinkConnector(
            pipeline_config=app_cfg_sink,
            mqtt_config=MQTT_CFG,
            feed_name="sink_test",
        )

        await conn.webrtc_coro()

        broker.stop()
        broker_thread.join()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_agent_switching():
    # test agent switcher (DRL -> GCC and vice versa) and safety detector
    try:
        broker = MosquittoBroker(port=1884)
        broker_thread = threading.Thread(target=broker.run, daemon=True)
        broker_thread.start()

        episodes = 1
        episode_length = 5000000
        stats_update_interval = 3.0  # sec

        eval_drl_cfg = DrlConfig(
            mode="eval",
            model_file="./models/fantastic_sb3_drl_model.zip",
            model_name="sac",
            episodes=episodes,
            episode_length=episode_length,
            is_reset_episodes=True,
            state_update_interval=stats_update_interval,
            state_max_inactivity_time=60.0,
            deterministic=True,
        )

        mdp = ViewerSeqMDP(
            reward_function_name="qoe_ahoy_seq_sensible",
            episode_length=episode_length,
            num_observations_for_state=5,
            constants={
                "MAX_BITRATE_STREAM_MBPS": 10,
                "MAX_BANDWIDTH_MBPS": APP_CFG.gcc_settings["max-bitrate"] / 1000000,
                "MIN_BANDWIDTH_MBPS": APP_CFG.gcc_settings["min-bitrate"] / 1000000,
            },
        )

        drl_agent = DrlAgent(
            drl_config=eval_drl_cfg,
            mdp=mdp,
            mqtt_config=MQTT_CFG,
            warmup=20.0,
        )

        sd_agent = SafetyDetectorAgent(
            mqtt_config=MQTT_CFG,
            switcher_configs={
                "fractionRtt": SwitcherConfig(
                    recover_iterations=8,
                    switch_forgives=3,
                )
            },
            switch_update_interval=stats_update_interval,
            warmup=20.0,
        )

        gcc_agent = GccAgent(
            mqtt_config=MQTT_CFG,
            action_period=stats_update_interval,
            is_enable_actions_on_start=False,
            warmup=20.0,
        )
        switching_pair = SwitchingPair(gcc_agent.id, drl_agent.id)

        conn = SinkConnector(
            pipeline_config=APP_CFG,
            agents=[drl_agent, sd_agent, gcc_agent],
            feed_name="test",
            mqtt_config=MQTT_CFG,
            switching_pair=switching_pair,
        )

        await conn.webrtc_coro()

        broker.stop()
        broker_thread.join()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def default():
    try:
        broker = MosquittoBroker(port=1884)
        broker_thread = threading.Thread(target=broker.run, daemon=True)
        broker_thread.start()

        is_cuda = False  # change to true if you have an NVIDIA GPU with HA support
        pipeline = DEFAULT_BIN_PIPELINE if not is_cuda else DEFAULT_BIN_CUDA_PIPELINE

        app_cfg = GstWebRTCAppConfig(
            pipeline_str=pipeline,
            video_url=APP_CFG.video_url,
            codec=APP_CFG.codec,
            bitrate=APP_CFG.bitrate,
            resolution=APP_CFG.resolution,
            gcc_settings=APP_CFG.gcc_settings,
            is_cuda=is_cuda,
        )

        conn = AhoyConnector(
            pipeline_config=app_cfg,
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="enjoy_the_stream",
            mqtt_config=MQTT_CFG,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

        broker.stop()
        broker_thread.join()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


if __name__ == "__main__":
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(default())
