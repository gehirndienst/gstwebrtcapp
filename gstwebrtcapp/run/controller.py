import asyncio
import json
import threading
import time
from typing import Any, Dict

from message.client import MqttConfig, MqttPair, MqttPublisher, MqttSubscriber
from utils.base import LOGGER, wait_for_condition


class FeedController:
    def __init__(
        self,
        mqtt_config: MqttConfig,
        feed_topics: Dict[str, str],
        controller_topic: str | None = None,
        aggregation_topic: str | None = None,
        allocation_weights: Dict[str, float] = {},
        max_inactivity_time: float = 10.0,
        warmup: float = 0.0,
    ) -> None:
        self.mqtt_config = mqtt_config
        self.mqtts = MqttPair(
            publisher=MqttPublisher(self.mqtt_config),
            subscriber=MqttSubscriber(self.mqtt_config),
        )
        self.mqtts_threads = [
            threading.Thread(target=self.mqtts.publisher.run, daemon=True).start(),
            threading.Thread(target=self.mqtts.subscriber.run, daemon=True).start(),
        ]

        self.feed_topics = feed_topics
        self.inactive_feed_topics = {}
        self.controller_topic = controller_topic
        self.aggregation_topic = aggregation_topic or self.mqtt_config.topics.actions
        self.mqtts.subscriber.subscribe([self.controller_topic, self.aggregation_topic])

        self.allocation_weights = {}
        self._init_allocation_weights(allocation_weights)

        self.max_inactivity_time = max_inactivity_time
        self.warmup = warmup

        self.is_running = False

    async def controller_coro(self) -> None:
        if self.controller_topic is None:
            LOGGER.error("ERROR: FeedController: controller topic is not set")
            return
        self.is_running = True
        LOGGER.info(f"INFO: FeedController's controller coroutine is starting...")

        while self.is_running:
            try:
                mqtt_msg = await self.mqtts.subscriber.await_message(self.controller_topic)
                feed_action_dict = json.loads(mqtt_msg.msg)
                if not (
                    isinstance(feed_action_dict, dict) and all(isinstance(v, dict) for v in feed_action_dict.values())
                ):
                    LOGGER.error(
                        f"ERROR: FeedController: invalid action format: {type(feed_action_dict)} but should be a dict of a dict"
                    )
                    self.is_running = False
                elif not all(
                    [k in self.feed_topics or k in self.inactive_feed_topics for k in feed_action_dict.keys()]
                ):
                    LOGGER.error(f"ERROR: FeedController: unknown feed name in the actions: {feed_action_dict.keys()}")
                    self.is_running = False
                else:
                    for feed_name, action_dict in feed_action_dict.items():
                        for action_name, action_value in action_dict.items():
                            match action_name:
                                case "bitrate" | "resolution" | "framerate" | "preset" | "switch":
                                    # send to the feed's connector
                                    topic = self.feed_topics.get(feed_name, None) or self.inactive_feed_topics.get(
                                        feed_name, None
                                    )
                                    self.mqtts.publisher.publish(topic, json.dumps({action_name: action_value}))
                                case "weight":
                                    # FIXME: maybe allow only all weights to be updated at once
                                    self.update_allocation_weights({feed_name: action_value})
                                    LOGGER.info(
                                        f"ACTION: feed {feed_name} has been updated with a new weight: {action_value}"
                                    )
                                case "manual":
                                    # pass "manual": True/False
                                    if action_value:
                                        self.remove_feed(feed_name)
                                        LOGGER.info(f"ACTION: feed {feed_name} has been turned into manual mode")
                                    else:
                                        self.add_feed(feed_name)
                                        LOGGER.info(f"ACTION: feed {feed_name} has been turned into AI-allocation mode")

                                case _:
                                    LOGGER.error(
                                        f"ERROR: FeedController : Unknown action in the message: {action_name}"
                                    )
            except Exception as e:
                raise Exception(f"ERROR: FeedController has thrown an exception: reason {e}")

    async def allocation_coro(self) -> None:
        await asyncio.sleep(self.warmup)
        if self.aggregation_topic is None:
            LOGGER.error("ERROR: FeedController: aggregation topic is not set")
            return
        self.mqtts.subscriber.clean_message_queue(self.aggregation_topic)
        self.is_running = True
        LOGGER.info(f"INFO: FeedController's allocation coroutine is starting...")

        while self.is_running:
            try:
                aggregated_actions = await self._aggregation_coro()
                if aggregated_actions is not None:
                    self._allocate_actions(aggregated_actions)
                else:
                    self.is_running = False
            except Exception as e:
                raise Exception(f"ERROR: FeedController has thrown an exception: reason {e}")

    async def _aggregation_coro(self) -> Dict[str, Any] | None:
        # it should aggregate N actions where N is the number of feeds
        if self.mqtts.subscriber.message_queues.get(self.aggregation_topic, None) is None:
            # if first message has not appeared, wait for it for 5 seconds
            wait_for_condition(
                lambda: self.mqtts.subscriber.message_queues.get(self.aggregation_topic, None) is not None, 5
            )
        aggregated_msgs = {}
        while len(aggregated_msgs) < len(self.feed_topics):
            mqtt_msg = await self.mqtts.subscriber.await_message(self.aggregation_topic, self.max_inactivity_time)
            feed_name = next((k for k in self.feed_topics if mqtt_msg.id.startswith(k)), None)
            if feed_name is None:
                if next((k for k in self.inactive_feed_topics if mqtt_msg.id.startswith(k)), None) is None:
                    LOGGER.error(f"ERROR: FeedController: unknown feed name in the MQTT message: {mqtt_msg.id}")
                    return None
            else:
                aggregated_msgs[feed_name] = json.loads(mqtt_msg.msg)

        # check if all feeds' agents have sent their actions
        if not all([feed_name in aggregated_msgs for feed_name in self.feed_topics]):
            LOGGER.error("ERROR: FeedController: not all feeds have sent their actions")
            return None
        return aggregated_msgs

    def _allocate_actions(self, actions: Dict[str, Any] | Dict[str, Dict[str, Any]]) -> None:
        # reallocate multiple actions to feeds according to the allocation weights
        # sum all actions for each feed
        summed_actions = {
            action_key: sum([actions[feed_name][action_key] for feed_name in actions])
            for action_key in actions[list(actions.keys())[0]]
        }
        allocated_actions = {
            feed_name: {
                action_key: summed_actions[action_key] * self.allocation_weights[feed_name]
                for action_key in summed_actions
            }
            for feed_name in self.feed_topics
        }
        # publish allocated actions to the feeds' connectors listening for them
        for feed_name, feed_topic in self.feed_topics.items():
            self.mqtts.publisher.publish(feed_topic, json.dumps(allocated_actions[feed_name]))

    def add_feed(
        self,
        name: str,
        action_topic: str | None = None,
        new_weights: Dict[str, float] | None = None,
    ) -> None:
        if name not in self.feed_topics:
            if action_topic is not None:
                self.feed_topics[name] = action_topic
                self.mqtts.subscriber.subscribe([action_topic])
            else:
                if name in self.inactive_feed_topics:
                    self.feed_topics[name] = self.inactive_feed_topics.pop(name)
                    self.mqtts.subscriber.subscribe([self.feed_topics[name]])
                else:
                    raise Exception(
                        f"ERROR: FeedController: no action topic is provided but feed {name} is not inactive"
                    )
            if new_weights is not None:
                self.update_allocation_weights(new_weights)
            else:
                # if no corrected weights are provided, then we need to reinit the weights
                self._init_allocation_weights()

    def remove_feed(
        self,
        name: str,
        new_weights: Dict[str, float] | None = None,
    ) -> None:
        if name in self.feed_topics:
            self.mqtts.subscriber.unsubscribe([self.feed_topics[name]])
            self.inactive_feed_topics[name] = self.feed_topics[name]
            _ = self.feed_topics.pop(name, None)
            if new_weights is not None:
                self.update_allocation_weights(new_weights)
            else:
                self._init_allocation_weights()

    def _init_allocation_weights(self, weights: Dict[str, float] | None = None) -> None:
        if weights:
            self.allocation_weights = weights
        else:
            self.allocation_weights = {feed_name: 1.0 / len(self.feed_topics) for feed_name in self.feed_topics}

    def update_allocation_weights(self, weights: Dict[str, float]) -> None:
        if sum(weights.values()) > 1.0:
            LOGGER.error(f"ERROR: FeedController: sum of weights is greater than 1")
            return
        elif sum(weights.values()) != 1.0 and len(self.feed_topics) == len(weights):
            LOGGER.error(f"ERROR: FeedController: sum of weights for all feeds is not equal to 1")
            return
        for feed_name, weight in weights.items():
            if feed_name in self.allocation_weights:
                self.allocation_weights[feed_name] = weight
            else:
                LOGGER.error(f"ERROR: FeedController: unknown feed name {feed_name} in the weights update")
        # recalculate other weights to keep the sum equal to 1
        if sum(self.allocation_weights.values()) != 1.0 and len(self.feed_topics) > len(weights):
            remaining_weight = (1.0 - sum(weights.values())) / len(self.feed_topics) - len(weights)
            for feed_name in self.feed_topics:
                if feed_name not in weights:
                    self.allocation_weights[feed_name] = remaining_weight
