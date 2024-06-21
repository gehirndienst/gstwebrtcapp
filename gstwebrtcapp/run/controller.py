import asyncio
import copy
import enum
import json
import threading
from typing import Any, Dict, Tuple

from gstwebrtcapp.message.client import MqttConfig, MqttPair, MqttPublisher, MqttSubscriber
from gstwebrtcapp.utils.base import LOGGER, wait_for_condition


class FeedState(enum.Enum):
    CONTROLLED = "controlled"
    MANUAL = "manual"
    OFF = "off"


class FeedController:
    def __init__(
        self,
        mqtt_config: MqttConfig,
        feed_topics: Dict[str, str],
        controller_topic: str | None = None,
        aggregation_topic: str | None = None,
        allocation_weights: Dict[str, float] = {},
        action_limits: Dict[str, Tuple[float, float]] = {},
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

        self.feeds = {feed_name: FeedState.CONTROLLED for feed_name in feed_topics}
        self.feed_topics = feed_topics
        self.controlled_feed_topics = copy.deepcopy(self.feed_topics)
        self.controller_topic = controller_topic or self.mqtt_config.topics.controller
        self.aggregation_topic = aggregation_topic or self.mqtt_config.topics.actions
        self.mqtts.subscriber.subscribe([self.controller_topic, self.aggregation_topic])

        self.allocation_weights = {}
        self._init_allocation_weights(allocation_weights)
        self.action_limits = action_limits
        self.max_inactivity_time = max_inactivity_time
        self.warmup = warmup

        self.is_running = False

    async def controller_coro(self) -> None:
        if not self.controller_topic:
            LOGGER.error("ERROR: FeedController: controller topic is not set, STOPPING...")
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
                elif not all([k in self.feeds or k == "all" for k in feed_action_dict.keys()]):
                    LOGGER.error(f"ERROR: FeedController: unknown feed name in the actions: {feed_action_dict.keys()}")
                    self.is_running = False
                else:
                    for feed_name, action_dict in feed_action_dict.items():
                        # NOTE: fix "all" feed name for actions for all feeds
                        if feed_name == "all":
                            for action_name, action_value in action_dict.items():
                                match action_name:
                                    case "weights":
                                        # pass it in a form of {"all": {"weights": {feed_name: weight, ...}}}
                                        self.update_allocation_weights(action_value)
                                        LOGGER.info(f"ACTION: updated weights for all feeds: {self.allocation_weights}")
                                    case _:
                                        LOGGER.error(
                                            f"ERROR: FeedController : Unknown general action with the name: {action_name}"
                                        )
                            continue
                        # INDIVIDUAL ACTIONS
                        for action_name, action_value in action_dict.items():
                            match action_name:
                                case "bitrate" | "resolution" | "framerate" | "preset" | "switch":
                                    # pass key: val: Any and catch it in the connector
                                    self.mqtts.publisher.publish(
                                        self.feed_topics[feed_name],
                                        json.dumps({action_name: action_value}),
                                    )
                                case "manual":
                                    # pass "manual": True/False
                                    if action_value:
                                        self.remove_feed(feed_name, FeedState.MANUAL)
                                        LOGGER.info(
                                            f"ACTION: feed {feed_name} has been turned into manual state, controlled feeds are: {self.controlled_feed_topics}"
                                        )
                                    else:
                                        self.add_feed(feed_name)
                                        LOGGER.info(
                                            f"ACTION: feed {feed_name} has been turned into controlled state, controlled feeds are: {self.controlled_feed_topics}"
                                        )
                                case "off":
                                    # pass "off": True/False
                                    if action_value:
                                        # notify the connector to stop the feed, else it is the other way around
                                        self.mqtts.publisher.publish(
                                            self.feed_topics[feed_name], json.dumps({action_name: action_value})
                                        )
                                    self.remove_feed(feed_name, FeedState.OFF)
                                    LOGGER.info(
                                        f"ACTION: feed {feed_name} has been turned off, controlled feeds are: {self.controlled_feed_topics}"
                                    )
                                case "on":
                                    # pass "on": action_topic: str | None
                                    self.add_feed(feed_name, action_value)
                                    LOGGER.info(
                                        f"ACTION: feed {feed_name} has been turned on, controlled feeds are: {self.controlled_feed_topics}"
                                    )
                                case _:
                                    LOGGER.error(f"ERROR: FeedController : Unknown action with the name: {action_name}")
            except Exception as e:
                raise Exception(f"ERROR: FeedController has thrown an exception: reason {e}")

    async def allocation_coro(self) -> None:
        await asyncio.sleep(self.warmup)
        if not self.aggregation_topic:
            LOGGER.error("ERROR: FeedController: aggregation topic is not set, STOPPING...")
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
        while len(aggregated_msgs) < len(self.controlled_feed_topics):
            mqtt_msg = await self.mqtts.subscriber.await_message(self.aggregation_topic)
            feed_name = next((k for k in self.controlled_feed_topics if mqtt_msg.id.startswith(k)), None)
            if feed_name is None:
                feed_name = next((k for k in self.feed_topics if mqtt_msg.id.startswith(k)), None)
                if feed_name is None:
                    # if not found -- unknown feed, if found, just non-controlled, ignore
                    LOGGER.error(f"ERROR: FeedController: unknown feed name in the MQTT message: {mqtt_msg.id}")
                    return None
            else:
                aggregated_msgs[feed_name] = json.loads(mqtt_msg.msg)

        if not aggregated_msgs:
            # could be only if there are no controlled feeds
            return None
        # check if all feeds' agents have sent their actions
        if not all([feed_name in aggregated_msgs for feed_name in self.controlled_feed_topics]):
            LOGGER.error("ERROR: FeedController: not all controlled feeds have sent their actions")
            return None
        return aggregated_msgs

    def _allocate_actions(self, actions: Dict[str, Any] | Dict[str, Dict[str, Any]]) -> None:
        # sum all actions for each feed that are controlled (a mismatch could be due to a concurrency)
        actions = {feed_name: actions[feed_name] for feed_name in actions if feed_name in self.controlled_feed_topics}
        summed_actions = {
            action_key: sum([actions[feed_name][action_key] for feed_name in actions])
            for action_key in actions[list(actions.keys())[0]]
        }

        # initial allocation based on weights
        allocated_actions = {
            feed_name: {
                action_key: summed_actions[action_key] * self.allocation_weights[feed_name]
                for action_key in summed_actions
            }
            for feed_name in self.controlled_feed_topics
        }

        # adjust allocations to respect min and max limits if provided
        for action_key in summed_actions:
            if action_key in self.action_limits:
                min_limit, max_limit = self.action_limits[action_key]
                while True:
                    adjustment_value = 0.0
                    max_feeds = []
                    min_feeds = []
                    for feed_name in self.controlled_feed_topics:
                        allocated = allocated_actions[feed_name][action_key]
                        if allocated > max_limit:
                            adjustment_value += allocated - max_limit
                            allocated_actions[feed_name][action_key] = max_limit
                            max_feeds.append(feed_name)
                        elif allocated < min_limit:
                            adjustment_value -= min_limit - allocated
                            allocated_actions[feed_name][action_key] = min_limit
                            min_feeds.append(feed_name)

                    if adjustment_value != 0.0:
                        no_more_feeds = min_feeds if adjustment_value < 0.0 else max_feeds
                        remaining_feeds = [feed for feed in self.controlled_feed_topics if feed not in no_more_feeds]
                        if remaining_feeds:
                            remaining_weight_denom = sum([self.allocation_weights[feed] for feed in remaining_feeds])
                            for feed_name in remaining_feeds:
                                weight = self.allocation_weights[feed_name] / remaining_weight_denom
                                allocated_actions[feed_name][action_key] += adjustment_value * weight
                        else:
                            # should not be reachable as the AI agent should also follow the same limits
                            break
                    else:
                        break

        # publish allocated actions to the feeds' connectors listening for them
        for feed_name, feed_topic in self.controlled_feed_topics.items():
            self.mqtts.publisher.publish(feed_topic, json.dumps(allocated_actions[feed_name]))

    def add_feed(
        self,
        name: str,
        action_topic: str | None = None,
        new_weights: Dict[str, float] | None = None,
    ) -> None:
        if name not in self.feeds:
            if action_topic is not None:
                self.feeds[name] = FeedState.CONTROLLED
                self.feed_topics[name] = action_topic
                self.controlled_feed_topics[name] = action_topic
                self.mqtts.subscriber.subscribe([action_topic])
            else:
                raise Exception(f"ERROR: FeedController: no action topic is provided for a new feed {name}")
        else:
            if self.feeds[name] != FeedState.CONTROLLED:
                self.feeds[name] = FeedState.CONTROLLED
                self.controlled_feed_topics[name] = self.feed_topics[name]
                self.mqtts.subscriber.subscribe([self.feed_topics[name]])
        if new_weights is not None:
            self.update_allocation_weights(new_weights)
        else:
            # if no corrected weights are provided, then we need to reinit the weights
            self._init_allocation_weights()

    def remove_feed(
        self,
        name: str,
        state: FeedState,
        is_forever: bool = False,
        new_weights: Dict[str, float] | None = None,
    ) -> None:
        if name in self.feeds:
            self.mqtts.subscriber.unsubscribe([self.feed_topics[name]])
            self.feeds[name] = state
            _ = self.controlled_feed_topics.pop(name, None)
            if is_forever:
                _ = self.feed_topics.pop(name, None)
                _ = self.feeds.pop(name, None)
            if new_weights is not None:
                self.update_allocation_weights(new_weights)
            else:
                self._init_allocation_weights()
        else:
            raise Exception(f"ERROR: FeedController: unknown feed {name} to remove")

    def _init_allocation_weights(self, weights: Dict[str, float] | None = None) -> None:
        if weights:
            self.allocation_weights = weights
        else:
            self.allocation_weights = {
                feed_name: 1.0 / len(self.controlled_feed_topics) for feed_name in self.controlled_feed_topics
            }

    def update_allocation_weights(self, weights: Dict[str, float]) -> None:
        if sum(weights.values()) > 1.0:
            LOGGER.error(f"ERROR: FeedController: sum of given weights is greater than 1, cannot update the weights")
            return
        for feed_name, weight in weights.items():
            if feed_name in self.allocation_weights:
                self.allocation_weights[feed_name] = weight
            else:
                LOGGER.error(f"ERROR: FeedController: unknown feed name {feed_name} in the weights update")
        # recalculate other weights to keep the sum equal to 1
        if sum(self.allocation_weights.values()) > 1.0 and len(self.controlled_feed_topics) > len(weights):
            remaining_weight = (1.0 - sum(weights.values())) / (len(self.controlled_feed_topics) - len(weights))
            for feed_name in self.controlled_feed_topics:
                if feed_name not in weights:
                    self.allocation_weights[feed_name] = remaining_weight
        elif sum(self.allocation_weights.values()) < 1.0:
            # acceptable but should be warned
            LOGGER.warning(
                f"WARNING: FeedController: sum of weights for all feeds is less than 1: {sum(self.allocation_weights)}"
            )
