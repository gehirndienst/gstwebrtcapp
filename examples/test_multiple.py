import asyncio
import copy
from typing import Dict

from gstwebrtcapp.apps.app import GstWebRTCAppConfig
from gstwebrtcapp.apps.pipelines import DEFAULT_SINK_PIPELINE
from gstwebrtcapp.apps.sinkapp.connector import SinkConnector
from gstwebrtcapp.control.cc.gcc_agent import GccAgent
from gstwebrtcapp.control.recorder.agent import RecorderAgent
from gstwebrtcapp.message.client import (
    MqttConfig,
    MqttExternalEstimationTopics,
    MqttGstWebrtcAppTopics,
)
from gstwebrtcapp.run.controller import FeedController
from gstwebrtcapp.run.wrappers import executor_wrapper

try:
    import uvloop  # type: ignore
except ImportError:
    uvloop = None

RTSP_URL = None  # TODO: set RTSP URL here
WARMUP = 10.0


def make_feed_mqtt_config(
    feed_name: str,
    mqtt_prefix: str = "",
    allocator_actions_topic: str | None = None,
    controller_topic: str | None = None,
    broker_host: str = "0.0.0.0",
    broker_port: int = 1883,
    keepalive: int = 20,
    username: str | None = None,
    password: str | None = None,
    is_tls: bool = False,
    external_topics: MqttExternalEstimationTopics | None = None,
) -> MqttConfig:
    prefix = f"{mqtt_prefix}/{feed_name}" if mqtt_prefix else feed_name
    return MqttConfig(
        id=feed_name,
        broker_host=broker_host,
        broker_port=broker_port,
        keepalive=keepalive,
        username=username,
        password=password,
        is_tls=is_tls,
        topics=MqttGstWebrtcAppTopics(
            gcc=f"{prefix}/gcc",
            stats=f"{prefix}/stats",
            state=f"{prefix}/state",
            actions=allocator_actions_topic or f"{prefix}/actions",
            controller=controller_topic or "",
        ),
        external_topics=external_topics,
    )


def make_inactive_mqtt_config(
    id: str,
    broker_host: str = "0.0.0.0",
    broker_port: int = 1883,
    keepalive: int = 20,
    username: str | None = None,
    password: str | None = None,
    is_tls: bool = False,
    controller_topic: str | None = None,
) -> MqttConfig:
    return MqttConfig(
        id=id,
        broker_host=broker_host,
        broker_port=broker_port,
        keepalive=keepalive,
        username=username,
        password=password,
        is_tls=is_tls,
        topics=MqttGstWebrtcAppTopics(
            gcc="",
            stats="",
            state="",
            actions="",
            controller=controller_topic or "",
        ),
        external_topics=None,
    )


def make_default_gcc_agent(
    mqtt_config: MqttConfig,
    action_period: float = 3.0,
) -> GccAgent:
    return GccAgent(
        mqtt_config=mqtt_config,
        action_period=action_period,
        warmup=WARMUP,
    )


def make_default_recorder_agent(
    mqtt_config: MqttConfig,
    id: str = "recorder",
    stats_update_interval: float = 1.0,
    max_inactivity_time: float = 20.0,
    log_path: str = "./logs",
    verbose: bool = False,
) -> RecorderAgent:
    return RecorderAgent(
        mqtt_config=mqtt_config,
        id=id,
        stats_update_interval=stats_update_interval,
        max_inactivity_time=max_inactivity_time,
        log_path=log_path,
        warmup=WARMUP,
        verbose=verbose,
    )


def make_default_app_cfg() -> GstWebRTCAppConfig:
    # app cfg with all available fields set to (almost) default values
    return GstWebRTCAppConfig(
        pipeline_str=DEFAULT_SINK_PIPELINE,
        video_url=RTSP_URL,  # TODO: here is where you need to supply the RTSP URL of your stream
        codec="h264",
        bitrate=2000,
        resolution={"width": 1280, "height": 720},
        framerate=20,
        fec_percentage=0,
        gcc_settings={"min-bitrate": 400000, "max-bitrate": 10000000},
        data_channels_cfgs=[],
        priority=4,
        max_timeout=30,
        is_debug=False,
    )


async def make_connector_coro(
    feed: str,
    app_config: GstWebRTCAppConfig | None = None,
    control_agent_type: str = "gcc",
    control_agent_action_period: float = 1.0,  # sec
    mqtt_prefix: str = "",
    allocator_actions_topic: str | None = None,
    controller_topic: str | None = None,
    is_add_recorder: bool = False,
) -> None:
    mqtt_cfg = make_feed_mqtt_config(
        feed_name=feed,
        mqtt_prefix=mqtt_prefix,
        allocator_actions_topic=allocator_actions_topic,
        controller_topic=controller_topic,
    )

    app_cfg = app_config or make_default_app_cfg()

    agents = []
    if control_agent_type == "gcc":
        control_agent = make_default_gcc_agent(
            mqtt_config=copy.deepcopy(mqtt_cfg), action_period=control_agent_action_period
        )
        agents.append(control_agent)
    else:
        control_agent = None

    if is_add_recorder:
        recorder_agent = make_default_recorder_agent(
            mqtt_config=copy.deepcopy(mqtt_cfg),
            id=f"recorder_{feed}",
        )
        agents.append(recorder_agent)

    connector_mqtt_cfg = copy.deepcopy(mqtt_cfg)
    connector_mqtt_cfg.id = f"connector_{feed}"
    connector_mqtt_cfg.topics.actions = f"{feed}/actions"

    connector = SinkConnector(
        pipeline_config=app_cfg,
        agents=agents,
        feed_name=feed,
        mqtt_config=connector_mqtt_cfg,
    )

    await connector.webrtc_coro()


async def make_allocation_coro(controller: FeedController) -> None:
    await controller.allocation_coro()


async def make_controller_coro(controller: FeedController) -> None:
    await controller.controller_coro()


async def main(
    feeds: Dict[str, GstWebRTCAppConfig] | None = None,
    control_agent_type: str = "gcc",
    control_agent_action_period: float = 3.0,
    mqtt_prefix: str = "",
    controller_topic: str | None = None,
    aggregation_topic: str | None = None,
    is_add_recorder: bool = False,
) -> None:
    tasks = []

    controller_topic = f"{mqtt_prefix}/{controller_topic}" if mqtt_prefix else controller_topic
    aggregation_topic = f"{mqtt_prefix}/{aggregation_topic}" if mqtt_prefix else aggregation_topic

    feeds = feeds or {"video1": make_default_app_cfg(), "video2": make_default_app_cfg()}
    for feed_name, feed_cfg in feeds.items():
        # make connector coros
        # NOTE: all essential coros must be wrapped in wrappers to handle cancelling and exceptions
        tasks.append(
            asyncio.create_task(
                executor_wrapper(
                    make_connector_coro,
                    feed_name,
                    feed_cfg,
                    control_agent_type,
                    control_agent_action_period,
                    mqtt_prefix,
                    aggregation_topic,
                    controller_topic,
                    is_add_recorder,
                    is_raise_exception=False,
                )
            )
        )

    # make feed controller
    feed_topic_prefix = f"{mqtt_prefix}/" if mqtt_prefix else ""
    controller = FeedController(
        mqtt_config=make_inactive_mqtt_config("controller", controller_topic=controller_topic),
        feed_topics={feed: f"{feed_topic_prefix}{feed}/actions" for feed in feeds},
        controller_topic=controller_topic,
        aggregation_topic=aggregation_topic,
        allocation_weights={},
        action_limits={"bitrate": (400, 10000)},  # kbps
        max_inactivity_time=20.0,
        warmup=WARMUP,
    )
    # make feed controller coro
    tasks.append(
        asyncio.create_task(
            executor_wrapper(
                make_controller_coro,
                controller,
                is_raise_exception=False,
            )
        )
    )

    if aggregation_topic:
        # make allocation coro
        tasks.append(
            asyncio.create_task(
                executor_wrapper(
                    make_allocation_coro,
                    controller,
                    is_raise_exception=False,
                )
            )
        )

    # go
    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    if RTSP_URL is None:
        raise ValueError("RTSP_URL is not set!")

    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # NOTE: define video feeds here, by default they use the same RTSP URL aka duplicate
    feeds = {
        "video1": make_default_app_cfg(),
        "video2": make_default_app_cfg(),
    }

    asyncio.run(
        main(
            feeds=feeds,
            control_agent_type="gcc",
            control_agent_action_period=1.0,
            mqtt_prefix="",  # or smth like "data/gstreamer"
            controller_topic="internal/controller",
            aggregation_topic="internal/aggregation",
            is_add_recorder=False,  # enable to generate csv with webrtc stats for each feed
        )
    )
