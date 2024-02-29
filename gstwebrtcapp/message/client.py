from abc import ABCMeta
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import secrets
import time
import paho.mqtt.client as mqtt
from typing import Dict, List

from utils.base import LOGGER, wait_for_condition


@dataclass
class MqttGstWebrtcAppTopics:
    gcc: str = "gstwebrtcapp/gcc"
    stats: str = "gstwebrtcapp/stats"
    actions: str = "gstwebrtcapp/actions"


@dataclass
class MqttConfig:
    id: str = ""
    broker_host: str = "0.0.0.0"
    broker_port: int = 1883
    keepalive: int = 20
    username: str | None = None
    password: str | None = None
    is_tls: bool = False
    topics: MqttGstWebrtcAppTopics = field(default_factory=lambda: MqttGstWebrtcAppTopics)


@dataclass
class MqttMessage:
    timestamp: str
    id: str
    msg: str
    topic: str


class MqttClient(metaclass=ABCMeta):
    def __init__(
        self,
        config: MqttConfig = MqttConfig(""),
    ) -> None:
        self.id = config.id if config.id else secrets.token_hex(4)
        self.broker_host = config.broker_host
        self.broker_port = config.broker_port
        self.keepalive = config.keepalive
        self.username = config.username
        self.password = config.password
        self.is_tls = config.is_tls
        self.topics = config.topics

        self.message_queue = None
        self.client = None

        self.is_running = False

    def run(self) -> None:
        # this method should be called in a separate thread/process
        self._spawn()
        self.client.loop_start()
        try:
            wait_for_condition(lambda: self.client.is_connected(), 10)
            LOGGER.info(f"OK: MQTT client {self.id} has been started")
        except TimeoutError:
            LOGGER.error(f"ERROR: MQTT client {self.id} has not been started")
            return
        self.is_running = True
        while self.is_running:
            time.sleep(0.1)

    def stop(self) -> None:
        self.is_running = False
        self.client.loop_stop()
        if self.client.is_connected():
            _ = self.client.disconnect()
        LOGGER.info(f"INFO: MQTT client {self.id} has been stopped")

    def _spawn(self) -> None:
        self.client = mqtt.Client(self.id)
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        if self.is_tls:
            self.client.tls_set()
        _ = self.client.connect(self.broker_host, self.broker_port, self.keepalive)


class MqttPublisher(MqttClient):
    def __init__(
        self,
        config: MqttConfig = MqttConfig(""),
    ) -> None:
        super().__init__(config)

    def publish(self, topic: str, msg: str) -> None:
        wait_for_condition(lambda: self.is_running, 10)
        self.client.publish(
            topic,
            json.dumps(
                {
                    'timestamp': datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")[:-3],
                    'id': self.id,
                    'msg': msg,
                }
            ),
        )
        LOGGER.debug(f"INFO: MQTT publisher {self.id} has published message: {msg} to {topic}")


class MqttSubscriber(MqttClient):
    def __init__(
        self,
        config: MqttConfig = MqttConfig(""),
    ) -> None:
        super().__init__(config)
        self.message_queue = asyncio.Queue()

    def on_message(self, _, __, msg) -> None:
        payload = json.loads(msg.payload.decode('utf8'))
        mqtt_message = MqttMessage(
            timestamp=payload['timestamp'],
            id=payload['id'],
            msg=payload['msg'],
            topic=msg.topic,
        )
        self.message_queue.put_nowait(mqtt_message)
        LOGGER.debug(f"Received message: {payload}")

    def subscribe(self, topics: List[str]) -> None:
        wait_for_condition(lambda: self.is_running, 10)
        for topic in topics:
            self.client.subscribe(topic, qos=1)
            self.client.on_message = self.on_message
            LOGGER.info(f"OK: MQTT subscriber {self.id} has successfully subscribed to {topic}")

    def get_message(self) -> MqttMessage | None:
        if self.is_running and not self.message_queue.empty():
            return self.message_queue.get_nowait()
        return None

    def clean_message_queue(self) -> None:
        while not self.message_queue.empty():
            _ = self.message_queue.get_nowait()


@dataclass
class MqttPair:
    publisher: MqttPublisher
    subscriber: MqttSubscriber
