'''
Test main functionalities by replacing the default one with one of the desired coroutine presented above in the main() endpoint.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>
'''

import asyncio

from apps.app import GstWebRTCAppConfig
from apps.ahoyapp.connector import AhoyConnector
from apps.pipelines import DEFAULT_BIN_CUDA_PIPELINE, DEFAULT_SINK_PIPELINE
from apps.sinkapp.connector import SinkConnector
from control.controller import Controller
from control.drl.agent import DrlAgent
from control.drl.config import DrlConfig
from control.drl.mdp import ViewerMDP
from control.recorder.agent import CsvViewerRecorderAgent
from network.controller import NetworkController
from utils.base import LOGGER

try:
    import uvloop
except ImportError:
    uvloop = None

AHOY_DIRECTOR_URL = "..."
API_KEY = "..."
VIDEO_SOURCE = "rtsp://..."


async def manipulate_video_coro(conn: AhoyConnector):
    while conn.app is None:
        await asyncio.sleep(0.5)
    await asyncio.sleep(20)
    LOGGER.info("----------Run test video manipulating coro...")
    LOGGER.info("----------Setting new bitrate...")
    conn.app.set_bitrate(1000)
    await asyncio.sleep(20)
    LOGGER.info("----------Setting new resolution...")
    conn.app.set_resolution(192, 144)
    await asyncio.sleep(20)
    LOGGER.info("----------Setting new framerate...")
    conn.app.set_framerate(15)
    await asyncio.sleep(20)
    LOGGER.info("----------Setting new fec...")
    conn.app.set_fec_percentage(50)
    await asyncio.sleep(20)
    LOGGER.info("----------Finished test video manipulating coro...")


async def test_manipulate_video():
    # run it to test video manipulation.
    try:
        cfg = GstWebRTCAppConfig(video_url=VIDEO_SOURCE)

        conn = AhoyConnector(
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            pipeline_config=cfg,
        )

        await conn.connect_coro()

        # note that a new coroutine is created and awaited together with webrtc_coro to manipulate the video
        conn_task = asyncio.create_task(conn.webrtc_coro())
        pipeline_task = asyncio.create_task(manipulate_video_coro(conn))
        await asyncio.gather(conn_task, pipeline_task)

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_nvenc():
    # run it to test nvenc hardware acceleration.
    try:
        cfg = GstWebRTCAppConfig(
            video_url="VIDEO_SOURCE",
            pipeline_str=DEFAULT_BIN_CUDA_PIPELINE,
            codec="h264",
            is_cuda=True,
        )

        conn = AhoyConnector(
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            pipeline_config=cfg,
        )

        await conn.connect_coro()

        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_csv_recorder():
    # run it to test csv viewer recorder agent
    try:
        cfg = GstWebRTCAppConfig(video_url=VIDEO_SOURCE)

        stats_update_interval = 1.0

        agent = CsvViewerRecorderAgent(
            controller=Controller(),
            stats_update_interval=stats_update_interval,
            warmup=10.0,
            log_path="./logs",
            verbose=2,
        )

        conn = AhoyConnector(
            pipeline_config=cfg,
            agent=agent,
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="recorder_test",
            stats_update_interval=stats_update_interval,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_drl():
    # run it to test drl agent
    try:
        episodes = 10
        episode_length = 50
        stats_update_interval = 3.0

        app_cfg = GstWebRTCAppConfig(video_url=VIDEO_SOURCE)

        agent = DrlAgent(
            config=DrlConfig(
                mode="train",
                model_name="sac",
                episodes=episodes,
                episode_length=episode_length,
                state_update_interval=stats_update_interval,
                hyperparams_cfg={
                    "policy": "MultiInputPolicy",
                    "batch_size": 128,
                    "ent_coef": "auto",
                    "policy_kwargs": {"log_std_init": -1, "activation_fn": "relu", "net_arch": [256, 256]},
                },
                callbacks=['print_step'],
                save_model_path="./models",
                save_log_path="./logs",
                verbose=2,
            ),
            controller=Controller(),
            mdp=ViewerMDP(
                reward_function_name="qoe_ahoy",
                episode_length=episode_length,
                constants={"MAX_BITRATE_STREAM_MBPS": 6},  # Ahoy fixes the max bitrate to 6 Mbps in SDP
            ),
        )

        conn = AhoyConnector(
            pipeline_config=app_cfg,
            agent=agent,
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="drl_test",
            stats_update_interval=stats_update_interval,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_drl_eval():
    try:
        episodes = 5
        episode_length = 512
        stats_update_interval = 3.0

        app_cfg = GstWebRTCAppConfig(video_url=VIDEO_SOURCE)

        drl_cfg = DrlConfig(
            mode="eval",
            model_file="model.zip",
            model_name="sac",
            episodes=episodes,
            episode_length=episode_length,
            state_update_interval=stats_update_interval,
            deterministic=True,
        )

        agent = DrlAgent(
            config=drl_cfg,
            controller=Controller(),
            mdp=ViewerMDP(
                reward_function_name="qoe_ahoy",
                episode_length=episode_length,
                constants={"MAX_BITRATE_STREAM_MBPS": 6},
            ),
        )

        conn = AhoyConnector(
            pipeline_config=app_cfg,
            agent=agent,
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="drl_test_eval",
            stats_update_interval=stats_update_interval,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_network_controller():
    try:
        cfg = GstWebRTCAppConfig(video_url=VIDEO_SOURCE)

        network_controller = NetworkController(gt_bandwidth=10.0, interval=10.0)
        network_controller.generate_rules(100, [0.7, 0.2, 0.1])

        conn = AhoyConnector(
            pipeline_config=cfg,
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            feed_name="test_network_controller",
            stats_update_interval=1.0,
            network_controller=network_controller,
        )

        await conn.connect_coro()
        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def test_sink_app():
    # run to test sink app. NOTE: you need to have a running signalling server and JS client to test this.
    # Check: https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/tree/main/net/webrtc?ref_type=heads#usage
    try:
        cfg = GstWebRTCAppConfig(
            pipeline_str=DEFAULT_SINK_PIPELINE,
            bitrate=6000,
            video_url=VIDEO_SOURCE,
        )

        conn = SinkConnector(
            pipeline_config=cfg,
        )

        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def default():
    try:
        cfg = GstWebRTCAppConfig(video_url=VIDEO_SOURCE)

        conn = AhoyConnector(
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            pipeline_config=cfg,
        )

        await conn.connect_coro()

        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


if __name__ == "__main__":
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(default())
