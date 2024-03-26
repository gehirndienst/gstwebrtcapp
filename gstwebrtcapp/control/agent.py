"""
agent.py

Description:
    Base Agent class for the gstreamerwebrtcapp to control the pipeline.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

from abc import ABCMeta
from enum import Enum
import threading

from message.client import MqttConfig, MqttPair, MqttPublisher, MqttSubscriber


class AgentType(Enum):
    ABSTRACT = "ABSTRACT"
    CC = "CC"
    DRL = "DRL"
    DRL_OFFLINE = "DRL_OFFLINE"
    RECORDER = "RECORDER"


class Agent(metaclass=ABCMeta):
    def __init__(
        self,
        mqtt_config: MqttConfig,
    ) -> None:
        self.mqtt_config = mqtt_config
        self.mqtts = MqttPair(
            publisher=MqttPublisher(self.mqtt_config),
            subscriber=MqttSubscriber(self.mqtt_config),
        )
        self.mqtts_threads = []
        self.type = AgentType.ABSTRACT

    def run(self, *args, **kwargs) -> None:
        self.mqtts_threads = [
            threading.Thread(target=self.mqtts.publisher.run, daemon=True).start(),
            threading.Thread(target=self.mqtts.subscriber.run, daemon=True).start(),
        ]
        self.mqtts.subscriber.subscribe([self.mqtt_config.topics.gcc])
        self.mqtts.subscriber.subscribe([self.mqtt_config.topics.stats])

    def stop(self) -> None:
        self.mqtts.publisher.stop()
        self.mqtts.subscriber.stop()
        if self.mqtts_threads:
            for t in self.mqtts_threads:
                if t:
                    t.join()
