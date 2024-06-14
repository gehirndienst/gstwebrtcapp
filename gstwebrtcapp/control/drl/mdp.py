from abc import ABCMeta, abstractmethod
import collections
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, OrderedDict, Tuple

from control.drl.reward import RewardFunctionFactory
from media.preset import VideoPresets
from utils.base import LOGGER, scale, unscale, get_list_average, slice_list_in_intervals
from utils.gst import GstWebRTCStatsType, find_stat, get_stat_diff, get_stat_diff_concat
from utils.webrtc import clock_units_to_seconds, ntp_short_format_to_seconds


class MDP(metaclass=ABCMeta):
    '''
    MDP is an abstract class for Markov Decision Process. It defines the interface for the environment.
    It also provides methods to translate the environment state and action to the controller's state and action
    that could be directly applied to the GStreamer pipeline and vice versa.
    '''

    MAX_BITRATE_STREAM_MBPS = 10.0  # so far for 1 stream only, later we can have multiple streams
    MIN_BITRATE_STREAM_MBPS = 0.4  # so far for 1 stream only, later we can have multiple streams
    MAX_BANDWIDTH_MBPS = 20.0
    MIN_BANDWIDTH_MBPS = 0.4
    MAX_DELAY_SEC = 1  # assume we target the sub-second latency

    CONSTANTS = {
        "MAX_BITRATE_STREAM_MBPS": MAX_BITRATE_STREAM_MBPS,
        "MIN_BITRATE_STREAM_MBPS": MIN_BITRATE_STREAM_MBPS,
        "MAX_BANDWIDTH_MBPS": MAX_BANDWIDTH_MBPS,
        "MIN_BANDWIDTH_MBPS": MIN_BANDWIDTH_MBPS,
        "MAX_DELAY_SEC": MAX_DELAY_SEC,
    }

    def __init__(
        self,
        reward_function_name: str,
        episode_length: int,
        num_observations_for_state: int = 1,
        is_deliver_all_observations: bool = False,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.reward_function = RewardFunctionFactory().create_reward_function(reward_function_name)
        self.reward_params = {}
        self.episode_length = episode_length
        self.num_observations_for_state = num_observations_for_state
        self.is_deliver_all_observations = is_deliver_all_observations
        self.state_history_size = state_history_size
        if constants is not None:
            for key in constants:
                self.CONSTANTS[key] = constants[key]

        self.mqtts = None
        self.states_made = 0
        self.is_scaled = False
        self.last_stats = None
        self.last_states = collections.deque(maxlen=self.state_history_size + 1)
        self.last_actions = collections.deque(maxlen=self.state_history_size + 1)
        self.max_rb_packetslost = 0
        self.first_ssrc = None
        self.obs_filter = None

    @abstractmethod
    def reset(self):
        # memento pattern
        self.reward_params = {}
        self.first_ssrc = None
        self.states_made = 0
        self.last_stats = None
        self.last_states = collections.deque(maxlen=self.state_history_size + 1)
        self.last_actions = collections.deque(maxlen=self.state_history_size + 1)

    @abstractmethod
    def create_observation_space(self) -> spaces.Dict:
        pass

    @abstractmethod
    def create_action_space(self) -> spaces.Space:
        pass

    @abstractmethod
    def make_default_state(self) -> OrderedDict[str, Any]:
        pass

    @abstractmethod
    def make_state(self, stats: Dict[str, Any], action: Any) -> OrderedDict[str, Any]:
        self.states_made += 1
        self.last_actions.append(action)
        pass

    @abstractmethod
    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        pass

    @abstractmethod
    def convert_to_unscaled_action(self, action: Any) -> Any:
        pass

    @abstractmethod
    def pack_action_for_controller(self, action: Any) -> Dict[str, Any]:
        pass

    def update_reward_params(self, *args, **kwargs) -> None:
        self.reward_params = {
            "constants": self.CONSTANTS,
            "last_actions": self.last_actions,
        }

    def calculate_reward(self) -> Tuple[float, Dict[str, Any | float] | None]:
        return self.reward_function.calculate_reward(self.last_states, self.reward_params)

    def get_default_reward_parts_dict(self) -> Dict[str, Any | float] | None:
        return dict(zip(self.reward_function.reward_parts, [0.0] * len(self.reward_function.reward_parts)))

    def check_observation(self, obs: Dict[str, Any]) -> bool:
        # rtp inbound stream is the most important stat
        rtp_inbounds = find_stat(obs, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        if not rtp_inbounds:
            return False

        # filter over other stats
        if self.obs_filter is not None:
            for stat in self.obs_filter:
                if stat != GstWebRTCStatsType.RTP_INBOUND_STREAM:
                    if not find_stat(obs, stat):
                        return False

        # check that pl not smaller than last max seen packet lost
        for rtp_inbound in rtp_inbounds:
            # haven't seen any packet lost yet
            if self.first_ssrc is None:
                return True
            else:
                # check viewer ssrc
                if rtp_inbound["ssrc"] == self.first_ssrc:
                    rb_packetslost = rtp_inbound["rb-packetslost"]
                    # assumed that packet lost increases more or less in the same manner
                    if rb_packetslost < self.max_rb_packetslost:
                        if self.states_made <= 10:
                            # could be some initial distortions
                            return True
                        else:
                            return False
                    else:
                        self.max_rb_packetslost = rb_packetslost
                        return True
                else:
                    return False

    def is_truncated(self, step) -> bool:
        return step >= self.episode_length


class ViewerMDP(MDP):
    '''
    This MDP takes VIEWER (aka BROWSER) stats delivered by GStreamer.
    '''

    def __init__(
        self,
        reward_function_name: str,
        episode_length: int,
        num_observations_for_state: int = 1,
        is_deliver_all_observations: bool = False,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            reward_function_name,
            episode_length,
            num_observations_for_state,
            is_deliver_all_observations,
            state_history_size,
            constants,
        )

        # obs are scaled to [0, 1], actions are scaled to [-1, 1]
        self.is_scaled = True

        # filter obs only with needed stats in
        self.obs_filter = [
            GstWebRTCStatsType.RTP_OUTBOUND_STREAM,
            GstWebRTCStatsType.RTP_INBOUND_STREAM,
            GstWebRTCStatsType.ICE_CANDIDATE_PAIR,
        ]

        self.reset()

    def reset(self):
        super().reset()
        self.rtts = []

    def create_observation_space(self) -> spaces.Dict:
        # normalized to [0, 1]
        return spaces.Dict(
            {
                "bandwidth": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionLossRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionNackRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionPliRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionQueueingRtt": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "fractionRtt": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "interarrivalRttJitter": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "lossRate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rttMean": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rttStd": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rxGoodput": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "txGoodput": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )

    def create_action_space(self) -> spaces.Space:
        # basic AS uses only bitrate, normalized to [-1, 1]
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def make_default_state(self) -> OrderedDict[str, Any]:
        return collections.OrderedDict(
            {
                "bandwidth": 0.0,
                "fractionLossRate": 0.0,
                "fractionNackRate": 0.0,
                "fractionPliRate": 0.0,
                "fractionQueueingRtt": 0.0,
                "fractionRtt": 0.0,
                "interarrivalRttJitter": 0.0,
                "lossRate": 0.0,
                "rttMean": 0.0,
                "rttStd": 0.0,
                "rxGoodput": 0.0,
                "txGoodput": 0.0,
            }
        )

    def make_state(self, stats: Dict[str, Any], action: Any) -> OrderedDict[str, Any]:
        super().make_state(stats, action)

        # get gcc bandwidth
        bws = []
        while not self.mqtts.subscriber.message_queues[self.mqtts.subscriber.topics.gcc].empty():
            msg = self.mqtts.subscriber.get_message(self.mqtts.subscriber.topics.gcc)
            bws.append(float(msg.msg))
        bws = [b / 1e6 for b in bws]
        if len(bws) == 0:
            if not self.last_states:
                bandwidth = 0.0
            else:
                bandwidth = self.last_states[-1]["bandwidth"] if 1.0 not in self.last_states[-1]["bandwidth"] else 1.0
        elif len(bws) == 1:
            bandwidth = scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"])
        else:
            bandwidth = scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"])
        # get dicts with needed stats
        rtp_outbound = find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        ice_candidate_pair = find_stat(stats, GstWebRTCStatsType.ICE_CANDIDATE_PAIR)
        if not rtp_outbound or not rtp_inbound or not ice_candidate_pair:
            return self.make_default_state()

        # get previous state for calculating fractional values
        last_rtp_outbound = (
            find_stat(self.last_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        last_rtp_inbound = (
            find_stat(self.last_stats, GstWebRTCStatsType.RTP_INBOUND_STREAM) if self.last_stats is not None else None
        )
        self.last_stats = stats

        if self.first_ssrc is None:
            # FIXME: make first ever ssrc to be the privileged one and take stats only from it
            self.first_ssrc = rtp_inbound[0]["ssrc"]

        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if rtp_inbound_ssrc["ssrc"] == self.first_ssrc:
                last_rtp_inbound_ssrc = last_rtp_inbound[i] if last_rtp_inbound is not None else None
                last_rtp_outbound_ssrc = last_rtp_outbound[0] if last_rtp_outbound is not None else None

                # get needed stats
                packets_sent_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                packets_recv_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-received")
                ts_diff_sec = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "timestamp") / 1000

                # loss rates
                # 1. fraction loss rate
                rb_packetslost_diff = get_stat_diff(rtp_inbound[i], last_rtp_inbound_ssrc, "rb-packetslost")
                fraction_loss_rate = (
                    rb_packetslost_diff / (packets_sent_diff + rb_packetslost_diff)
                    if packets_sent_diff + rb_packetslost_diff > 0
                    else 0
                )
                fraction_loss_rate = max(0, min(1, fraction_loss_rate))
                # 7. global loss rate
                loss_rate = (
                    rtp_inbound[i]["rb-packetslost"]
                    / (rtp_outbound[0]["packets-sent"] + rtp_inbound[i]["rb-packetslost"])
                    if rtp_outbound[0]["packets-sent"] + rtp_inbound[i]["rb-packetslost"] > 0
                    else 0.0
                )

                # 2. fraction nack rate
                recv_nack_count_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "nack-count")
                fraction_nack_rate = recv_nack_count_diff / packets_recv_diff if packets_recv_diff > 0 else 0.0

                # 3. fraction pli rate
                recv_pli_count_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "pli-count")
                fraction_pli_rate = recv_pli_count_diff / packets_recv_diff if packets_recv_diff > 0 else 0.0

                # rtts: RTT comes in NTP short format
                rtt = ntp_short_format_to_seconds(rtp_inbound[i]["rb-round-trip"]) / self.CONSTANTS["MAX_DELAY_SEC"]
                self.rtts.append(rtt)

                # 4. fraction queueing rtt
                fraction_queueing_rtt = rtt - min(self.rtts) if len(self.rtts) > 0 else 0.0
                # 9. mean rtt
                rtt_mean = np.mean(self.rtts) if len(self.rtts) > 0 else 0.0
                # 10. std rtt
                rtt_std = np.std(self.rtts) if len(self.rtts) > 0 else 0.0

                # 6. jitter: comes in clock units
                interarrival_jitter = (
                    clock_units_to_seconds(rtp_inbound[i]["rb-jitter"], rtp_outbound[0]["clock-rate"])
                    / self.CONSTANTS["MAX_DELAY_SEC"]
                )

                # 11. rx rate
                try:
                    bitrate_recv = ice_candidate_pair[0]["bitrate-recv"]
                    rx_rate = scale(
                        bitrate_recv / 1000000,
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    )
                except KeyError:
                    rx_bytes_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-received")
                    rx_mbits_diff = rx_bytes_diff * 8 / 1000000
                    rx_rate = rx_mbits_diff / ts_diff_sec if ts_diff_sec > 0 else 0.0
                    rx_rate = scale(
                        rx_rate, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"]
                    )

                # 12. tx rate
                try:
                    bitrate_sent = ice_candidate_pair[0]["bitrate-sent"]
                    tx_rate = scale(
                        bitrate_sent / 1000000,
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    )
                except KeyError:
                    tx_bytes_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-sent")
                    tx_mbits_diff = tx_bytes_diff * 8 / 1000000
                    tx_rate = tx_mbits_diff / ts_diff_sec if ts_diff_sec > 0 else 0.0
                    tx_rate = scale(
                        tx_rate, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"]
                    )

                # form the final state
                state = collections.OrderedDict(
                    {
                        "bandwidth": bandwidth,
                        "fractionLossRate": fraction_loss_rate,
                        "fractionNackRate": fraction_nack_rate,
                        "fractionPliRate": fraction_pli_rate,
                        "fractionQueueingRtt": fraction_queueing_rtt,
                        "fractionRtt": rtt,
                        "interarrivalRttJitter": interarrival_jitter,
                        "lossRate": loss_rate,
                        "rttMean": rtt_mean,
                        "rttStd": rtt_std,
                        "rxGoodput": rx_rate,
                        "txGoodput": tx_rate,
                    }
                )

                self.last_states.append(state)
                self.update_reward_params()
                return state

        LOGGER.warning("WARNING: Drl Agent: ViewerMDP: make_state: no ssrc stats found")
        return self.make_default_state()

    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        return (
            collections.OrderedDict(
                {
                    "bandwidth": unscale(
                        state["bandwidth"],
                        self.CONSTANTS["MIN_BANDWIDTH_MBPS"],
                        self.CONSTANTS["MAX_BANDWIDTH_MBPS"],
                    ),
                    "fractionLossRate": state["fractionLossRate"],
                    "fractionNackRate": state["fractionNackRate"],
                    "fractionPliRate": state["fractionPliRate"],
                    "fractionQueueingRtt": state["fractionQueueingRtt"] * self.MAX_DELAY_SEC,
                    "fractionRtt": state["fractionRtt"] * self.MAX_DELAY_SEC,
                    "interarrivalRttJitter": state["interarrivalRttJitter"] * self.MAX_DELAY_SEC,
                    "lossRate": state["lossRate"],
                    "rttMean": state["rttMean"] * self.MAX_DELAY_SEC,
                    "rttStd": state["rttStd"] * self.MAX_DELAY_SEC,
                    "rxGoodput": unscale(
                        state["rxGoodput"],
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    ),
                    "txGoodput": unscale(
                        state["txGoodput"],
                        self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                        self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                    ),
                }
            )
            if self.is_scaled
            else state
        )

    def convert_to_unscaled_action(self, action: np.ndarray | float | int) -> np.ndarray | float:
        return self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"] + (
            (action + 1) * (self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"] - self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"]) / 2
        )

    def pack_action_for_controller(self, action: Any) -> Dict[str, Any]:
        # here we have only bitrate decisions that come as a 1-size np array in mbps (check create_action_space)
        return {"bitrate": self.convert_to_unscaled_action(action)[0] * 1000}


class ViewerSeqMDP(MDP):
    '''
    This MDP takes VIEWER (aka BROWSER) sequential stats (stacked observations) delivered by GStreamer.
    '''

    def __init__(
        self,
        reward_function_name: str = "qoe_ahoy_seq",
        episode_length: int = 256,
        num_observations_for_state: int = 5,
        is_deliver_all_observations: bool = True,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            reward_function_name,
            episode_length,
            num_observations_for_state,
            is_deliver_all_observations,
            state_history_size,
            constants,
        )

        # obs are scaled to [0, 1], actions are scaled to [-1, 1]
        self.is_scaled = True

        # filter obs only with needed stats in
        self.obs_filter = [
            GstWebRTCStatsType.RTP_OUTBOUND_STREAM,
            GstWebRTCStatsType.RTP_INBOUND_STREAM,
            GstWebRTCStatsType.ICE_CANDIDATE_PAIR,
        ]

        self.reset()

    def reset(self):
        super().reset()
        self.rtts = []

    def create_observation_space(self) -> spaces.Dict:
        # normalized to [0, 1]
        shape = (self.num_observations_for_state,)
        return spaces.Dict(
            {
                "bandwidth": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
                "fractionLossRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionNackRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionPliRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionQueueingRtt": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionRtt": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "interarrivalRttJitter": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "lossRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "rttMean": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rttStd": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rxGoodput": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "txGoodput": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
            }
        )

    def create_action_space(self) -> spaces.Space:
        # basic AS uses only bitrate, normalized to [-1, 1]
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def make_default_state(self) -> OrderedDict[str, Any]:
        def_val = 0.0 if self.num_observations_for_state == 1 else [0.0] * self.num_observations_for_state
        return collections.OrderedDict(
            {
                "bandwidth": [0.0, 0.0],
                "fractionLossRate": def_val,
                "fractionNackRate": def_val,
                "fractionPliRate": def_val,
                "fractionQueueingRtt": def_val,
                "fractionRtt": def_val,
                "interarrivalRttJitter": def_val,
                "lossRate": def_val,
                "rttMean": 0.0,
                "rttStd": 0.0,
                "rxGoodput": def_val,
                "txGoodput": def_val,
            }
        )

    def make_state(self, stats: Dict[str, Any], action: Any) -> OrderedDict[str, Any]:
        super().make_state(stats, action)

        # get gcc bandiwdth
        bws = []
        while not self.mqtts.subscriber.message_queues[self.mqtts.subscriber.topics.gcc].empty():
            msg = self.mqtts.subscriber.get_message(self.mqtts.subscriber.topics.gcc)
            bws.append(float(msg.msg))
        bws = [b / 1e6 for b in bws]
        if len(bws) == 0:
            if not self.last_states:
                bandwidth = [0.0, 0.0]
            else:
                bandwidth = (
                    self.last_states[-1]["bandwidth"] if 1.0 not in self.last_states[-1]["bandwidth"] else [1.0, 1.0]
                )
        elif len(bws) == 1:
            bandwidth = [
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
            ]
        else:
            bandwidth = [
                scale(bws[0], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
                scale(bws[-1], self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"]),
            ]

        # get dicts with needed stats
        rtp_outbound = find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        ice_candidate_pair = find_stat(stats, GstWebRTCStatsType.ICE_CANDIDATE_PAIR)
        if not rtp_outbound or not rtp_inbound or not ice_candidate_pair:
            return self.make_default_state()

        # get previous state for calculating fractional values
        last_rtp_outbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        last_rtp_inbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM) if self.last_stats is not None else None
        )
        self.last_stats = stats

        if self.first_ssrc is None:
            # FIXME: make first ever ssrc to be the privileged one and take stats only from it
            self.first_ssrc = rtp_inbound[0]["ssrc"][0]

        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if rtp_inbound_ssrc["ssrc"][0] == self.first_ssrc:
                last_rtp_inbound_ssrc = last_rtp_inbound[i] if last_rtp_inbound is not None else None
                last_rtp_outbound_ssrc = last_rtp_outbound[0] if last_rtp_outbound is not None else None

                # get needed stats
                packets_sent_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                packets_recv_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-received")
                ts_diff_sec = [
                    ts / 1000 for ts in get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "timestamp")
                ]

                # loss rates
                # 1. fraction loss rate
                rb_packetslost_diff = get_stat_diff_concat(rtp_inbound[i], last_rtp_inbound_ssrc, "rb-packetslost")
                fraction_loss_rates_raw = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rb_packetslost_diff, packets_sent_diff)
                ]
                fraction_loss_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(fraction_loss_rates_raw, self.num_observations_for_state)
                ]

                # 7. global loss rate
                loss_rates_raw = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rtp_inbound[i]["rb-packetslost"], rtp_outbound[0]["packets-sent"])
                ]

                loss_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(loss_rates_raw, self.num_observations_for_state)
                ]

                # 2. fraction nack rate
                recv_nack_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "nack-count")
                fraction_nack_rates_raw = [
                    nack / recv if recv > 0 else 0.0 for nack, recv in zip(recv_nack_count_diff, packets_recv_diff)
                ]
                fraction_nack_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(fraction_nack_rates_raw, self.num_observations_for_state)
                ]

                # 3. fraction pli rate
                recv_pli_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "pli-count")
                fraction_pli_rates_raw = [
                    pli / recv if recv > 0 else 0.0 for pli, recv in zip(recv_pli_count_diff, packets_recv_diff)
                ]
                fraction_pli_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(fraction_pli_rates_raw, self.num_observations_for_state)
                ]

                # rtts: RTT comes in NTP short format
                rtts_raw = [
                    scale(ntp_short_format_to_seconds(rtt), 0, self.CONSTANTS["MAX_DELAY_SEC"])
                    for rtt in rtp_inbound[i]["rb-round-trip"]
                ]
                self.rtts.extend(rtts_raw)
                rtts = [
                    get_list_average(mi) for mi in slice_list_in_intervals(rtts_raw, self.num_observations_for_state)
                ]

                # 4. fraction queueing rtt
                fraction_queueing_rtts = [rtt - min(self.rtts) if len(self.rtts) > 0 else 0.0 for rtt in rtts]
                # 9. mean rtt
                rtt_mean = np.mean(self.rtts) if len(self.rtts) > 0 else 0.0
                # 10. std rtt
                rtt_std = np.std(self.rtts) if len(self.rtts) > 0 else 0.0

                # 6. jitter: comes in clock units
                interarrival_jitters_raw = [
                    scale(
                        clock_units_to_seconds(j, rtp_outbound[0]["clock-rate"][0]), 0, self.CONSTANTS["MAX_DELAY_SEC"]
                    )
                    for j in rtp_inbound[i]["rb-jitter"]
                ]
                interarrival_jitters = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(interarrival_jitters_raw, self.num_observations_for_state)
                ]

                # 11. rx rate
                try:
                    bitrate_recvs = ice_candidate_pair[0]["bitrate-recv"]
                    rx_rates_raw = [
                        scale(
                            b / 1000000,
                            self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                            self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                        )
                        for b in bitrate_recvs
                    ]
                    rx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(rx_rates_raw, self.num_observations_for_state)
                    ]
                except KeyError:
                    rx_bytes_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-received")
                    rx_mbits_diff = [r * 8 / 1000000 for r in rx_bytes_diff]
                    rx_rates = [r / ts if ts > 0 else 0.0 for r, ts in zip(rx_mbits_diff, ts_diff_sec)]
                    rx_rates_raw = [
                        scale(r, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for r in rx_rates
                    ]
                    rx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(rx_rates_raw, self.num_observations_for_state)
                    ]

                # 12. tx rate
                try:
                    bitrate_sents = ice_candidate_pair[0]["bitrate-sent"]
                    tx_rates_raw = [
                        scale(
                            b / 1000000,
                            self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                            self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                        )
                        for b in bitrate_sents
                    ]
                    tx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(tx_rates_raw, self.num_observations_for_state)
                    ]
                except KeyError:
                    tx_bytes_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-sent")
                    tx_mbits_diff = [t * 8 / 1000000 for t in tx_bytes_diff]
                    tx_rates = [t / ts if ts > 0 else 0.0 for t, ts in zip(tx_mbits_diff, ts_diff_sec)]
                    tx_rates_raw = [
                        scale(t, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for t in tx_rates
                    ]
                    tx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(tx_rates_raw, self.num_observations_for_state)
                    ]

                # form the final state
                state = collections.OrderedDict(
                    {
                        "bandwidth": bandwidth,
                        "fractionLossRate": fraction_loss_rates,
                        "fractionNackRate": fraction_nack_rates,
                        "fractionPliRate": fraction_pli_rates,
                        "fractionQueueingRtt": fraction_queueing_rtts,
                        "fractionRtt": rtts,
                        "interarrivalRttJitter": interarrival_jitters,
                        "lossRate": loss_rates,
                        "rttMean": rtt_mean,
                        "rttStd": rtt_std,
                        "rxGoodput": rx_rates,
                        "txGoodput": tx_rates,
                    }
                )

                self.last_states.append(state)
                self.update_reward_params()
                return state

        LOGGER.warning("WARNING: Drl Agent: ViewerMDP: make_state: no ssrc stats found")
        return self.make_default_state()

    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        return (
            collections.OrderedDict(
                {
                    "bandwidth": [
                        unscale(s, self.CONSTANTS["MIN_BANDWIDTH_MBPS"], self.CONSTANTS["MAX_BANDWIDTH_MBPS"])
                        for s in state["bandwidth"]
                    ],
                    "fractionLossRate": state["fractionLossRate"],
                    "fractionNackRate": state["fractionNackRate"],
                    "fractionPliRate": state["fractionPliRate"],
                    "fractionQueueingRtt": [fqr * self.MAX_DELAY_SEC for fqr in state["fractionQueueingRtt"]],
                    "fractionRtt": [fr * self.MAX_DELAY_SEC for fr in state["fractionRtt"]],
                    "interarrivalRttJitter": [irj * self.MAX_DELAY_SEC for irj in state["interarrivalRttJitter"]],
                    "lossRate": state["lossRate"],
                    "rttMean": state["rttMean"] * self.MAX_DELAY_SEC,
                    "rttStd": state["rttStd"] * self.MAX_DELAY_SEC,
                    "rxGoodput": [
                        unscale(r, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for r in state["rxGoodput"]
                    ],
                    "txGoodput": [
                        unscale(t, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for t in state["txGoodput"]
                    ],
                }
            )
            if self.is_scaled
            else state
        )

    def convert_to_unscaled_action(self, action: np.ndarray | float | int) -> np.ndarray | float:
        return self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"] + (
            (action + 1) * (self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"] - self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"]) / 2
        )

    def pack_action_for_controller(self, action: Any) -> Dict[str, Any]:
        # here we have only bitrate decisions that come as a 1-size np array in mbps (check create_action_space)
        return {"bitrate": self.convert_to_unscaled_action(action)[0] * 1000}


class ViewerSeqNoBaselineMDP(ViewerSeqMDP):
    '''
    This MDP takes VIEWER (aka BROWSER) sequential stats (stacked observations) w/o GCC baseline delivered by GStreamer.
    '''

    def __init__(
        self,
        reward_function_name: str = "qoe_ahoy_seq",
        episode_length: int = 256,
        num_observations_for_state: int = 5,
        is_deliver_all_observations: bool = True,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            reward_function_name,
            episode_length,
            num_observations_for_state,
            is_deliver_all_observations,
            state_history_size,
            constants,
        )

    def create_observation_space(self) -> spaces.Dict:
        # normalized to [0, 1]
        shape = (self.num_observations_for_state,)
        return spaces.Dict(
            {
                "fractionLossRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionNackRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionPliRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionQueueingRtt": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "fractionRtt": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "interarrivalRttJitter": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "lossRate": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "rttMean": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rttStd": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "rxGoodput": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
                "txGoodput": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
            }
        )

    def make_default_state(self) -> OrderedDict[str, Any]:
        def_val = 0.0 if self.num_observations_for_state == 1 else [0.0] * self.num_observations_for_state
        return collections.OrderedDict(
            {
                "fractionLossRate": def_val,
                "fractionNackRate": def_val,
                "fractionPliRate": def_val,
                "fractionQueueingRtt": def_val,
                "fractionRtt": def_val,
                "interarrivalRttJitter": def_val,
                "lossRate": def_val,
                "rttMean": 0.0,
                "rttStd": 0.0,
                "rxGoodput": def_val,
                "txGoodput": def_val,
            }
        )

    def make_state(self, stats: Dict[str, Any], action: Any) -> OrderedDict[str, Any]:
        super().make_state(stats, action)

        # get dicts with needed stats
        rtp_outbound = find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        ice_candidate_pair = find_stat(stats, GstWebRTCStatsType.ICE_CANDIDATE_PAIR)
        if not rtp_outbound or not rtp_inbound or not ice_candidate_pair:
            return self.make_default_state()

        # get previous state for calculating fractional values
        last_rtp_outbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        last_rtp_inbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM) if self.last_stats is not None else None
        )
        self.last_stats = stats

        if self.first_ssrc is None:
            # FIXME: make first ever ssrc to be the privileged one and take stats only from it
            self.first_ssrc = rtp_inbound[0]["ssrc"][0]

        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if rtp_inbound_ssrc["ssrc"][0] == self.first_ssrc:
                last_rtp_inbound_ssrc = last_rtp_inbound[i] if last_rtp_inbound is not None else None
                last_rtp_outbound_ssrc = last_rtp_outbound[0] if last_rtp_outbound is not None else None

                # get needed stats
                packets_sent_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                packets_recv_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-received")
                ts_diff_sec = [
                    ts / 1000 for ts in get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "timestamp")
                ]

                # loss rates
                # 1. fraction loss rate
                rb_packetslost_diff = get_stat_diff_concat(rtp_inbound[i], last_rtp_inbound_ssrc, "rb-packetslost")
                fraction_loss_rates_raw = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rb_packetslost_diff, packets_sent_diff)
                ]
                fraction_loss_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(fraction_loss_rates_raw, self.num_observations_for_state)
                ]

                # 7. global loss rate
                loss_rates_raw = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rtp_inbound[i]["rb-packetslost"], rtp_outbound[0]["packets-sent"])
                ]

                loss_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(loss_rates_raw, self.num_observations_for_state)
                ]

                # 2. fraction nack rate
                recv_nack_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "nack-count")
                fraction_nack_rates_raw = [
                    nack / recv if recv > 0 else 0.0 for nack, recv in zip(recv_nack_count_diff, packets_recv_diff)
                ]
                fraction_nack_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(fraction_nack_rates_raw, self.num_observations_for_state)
                ]

                # 3. fraction pli rate
                recv_pli_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "pli-count")
                fraction_pli_rates_raw = [
                    pli / recv if recv > 0 else 0.0 for pli, recv in zip(recv_pli_count_diff, packets_recv_diff)
                ]
                fraction_pli_rates = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(fraction_pli_rates_raw, self.num_observations_for_state)
                ]

                # rtts: RTT comes in NTP short format
                rtts_raw = [
                    scale(ntp_short_format_to_seconds(rtt), 0, self.CONSTANTS["MAX_DELAY_SEC"])
                    for rtt in rtp_inbound[i]["rb-round-trip"]
                ]
                self.rtts.extend(rtts_raw)
                rtts = [
                    get_list_average(mi) for mi in slice_list_in_intervals(rtts_raw, self.num_observations_for_state)
                ]

                # 4. fraction queueing rtt
                fraction_queueing_rtts = [rtt - min(self.rtts) if len(self.rtts) > 0 else 0.0 for rtt in rtts]
                # 9. mean rtt
                rtt_mean = np.mean(self.rtts) if len(self.rtts) > 0 else 0.0
                # 10. std rtt
                rtt_std = np.std(self.rtts) if len(self.rtts) > 0 else 0.0

                # 6. jitter: comes in clock units
                interarrival_jitters_raw = [
                    scale(
                        clock_units_to_seconds(j, rtp_outbound[0]["clock-rate"][0]), 0, self.CONSTANTS["MAX_DELAY_SEC"]
                    )
                    for j in rtp_inbound[i]["rb-jitter"]
                ]
                interarrival_jitters = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(interarrival_jitters_raw, self.num_observations_for_state)
                ]

                # 11. rx rate
                try:
                    bitrate_recvs = ice_candidate_pair[0]["bitrate-recv"]
                    rx_rates_raw = [
                        scale(
                            b / 1000000,
                            self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                            self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                        )
                        for b in bitrate_recvs
                    ]
                    rx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(rx_rates_raw, self.num_observations_for_state)
                    ]
                except KeyError:
                    rx_bytes_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-received")
                    rx_mbits_diff = [r * 8 / 1000000 for r in rx_bytes_diff]
                    rx_rates = [r / ts if ts > 0 else 0.0 for r, ts in zip(rx_mbits_diff, ts_diff_sec)]
                    rx_rates_raw = [
                        scale(r, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for r in rx_rates
                    ]
                    rx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(rx_rates_raw, self.num_observations_for_state)
                    ]

                # 12. tx rate
                try:
                    bitrate_sents = ice_candidate_pair[0]["bitrate-sent"]
                    tx_rates_raw = [
                        scale(
                            b / 1000000,
                            self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"],
                            self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"],
                        )
                        for b in bitrate_sents
                    ]
                    tx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(tx_rates_raw, self.num_observations_for_state)
                    ]
                except KeyError:
                    tx_bytes_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-sent")
                    tx_mbits_diff = [t * 8 / 1000000 for t in tx_bytes_diff]
                    tx_rates = [t / ts if ts > 0 else 0.0 for t, ts in zip(tx_mbits_diff, ts_diff_sec)]
                    tx_rates_raw = [
                        scale(t, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for t in tx_rates
                    ]
                    tx_rates = [
                        get_list_average(mi, is_skip_zeroes=True)
                        for mi in slice_list_in_intervals(tx_rates_raw, self.num_observations_for_state)
                    ]

                # form the final state
                state = collections.OrderedDict(
                    {
                        "fractionLossRate": fraction_loss_rates,
                        "fractionNackRate": fraction_nack_rates,
                        "fractionPliRate": fraction_pli_rates,
                        "fractionQueueingRtt": fraction_queueing_rtts,
                        "fractionRtt": rtts,
                        "interarrivalRttJitter": interarrival_jitters,
                        "lossRate": loss_rates,
                        "rttMean": rtt_mean,
                        "rttStd": rtt_std,
                        "rxGoodput": rx_rates,
                        "txGoodput": tx_rates,
                    }
                )

                self.last_states.append(state)
                self.update_reward_params()
                return state

        LOGGER.warning("WARNING: Drl Agent: ViewerMDP: make_state: no ssrc stats found")
        return self.make_default_state()

    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        return (
            collections.OrderedDict(
                {
                    "fractionLossRate": state["fractionLossRate"],
                    "fractionNackRate": state["fractionNackRate"],
                    "fractionPliRate": state["fractionPliRate"],
                    "fractionQueueingRtt": [fqr * self.MAX_DELAY_SEC for fqr in state["fractionQueueingRtt"]],
                    "fractionRtt": [fr * self.MAX_DELAY_SEC for fr in state["fractionRtt"]],
                    "interarrivalRttJitter": [irj * self.MAX_DELAY_SEC for irj in state["interarrivalRttJitter"]],
                    "lossRate": state["lossRate"],
                    "rttMean": state["rttMean"] * self.MAX_DELAY_SEC,
                    "rttStd": state["rttStd"] * self.MAX_DELAY_SEC,
                    "rxGoodput": [
                        unscale(r, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for r in state["rxGoodput"]
                    ],
                    "txGoodput": [
                        unscale(t, self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"])
                        for t in state["txGoodput"]
                    ],
                }
            )
            if self.is_scaled
            else state
        )


class ViewerSeqDiscreteMDP(ViewerSeqMDP):
    '''
    This MDP takes VIEWER (aka BROWSER) sequential stats (stacked observations) and outputs precooked video presets from a discrete action space.
    '''

    def __init__(
        self,
        reward_function_name: str = "qoe_ahoy_seq",
        episode_length: int = 256,
        num_observations_for_state: int = 5,
        is_deliver_all_observations: bool = True,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            reward_function_name,
            episode_length,
            num_observations_for_state,
            is_deliver_all_observations,
            state_history_size,
            constants,
        )

    def create_action_space(self) -> spaces.Space:
        # discrete AS uses precooked video presets
        return spaces.Discrete(len(VideoPresets))

    def convert_to_unscaled_action(self, action: np.int64) -> int:
        # from int64 to int
        return int(action)

    def pack_action_for_controller(self, action: np.int64) -> Dict[str, int]:
        return {"preset": self.convert_to_unscaled_action(action)}


class ViewerSeqOfflineMDP(MDP):
    '''
    This MDP takes VIEWER (aka BROWSER) sequential stats processed for offline DRL delivered by GStreamer.
    '''

    def __init__(
        self,
        reward_function_name: str,
        episode_length: int,
        num_observations_for_state: int = 5,
        is_deliver_all_observations: bool = True,
        state_history_size: int = 10,
        constants: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            reward_function_name,
            episode_length,
            num_observations_for_state,
            is_deliver_all_observations,
            state_history_size,
            constants,
        )

        # scale with d3rlpy scaler
        self.is_scaled = False

        # default
        self.obs_filter = [
            GstWebRTCStatsType.RTP_OUTBOUND_STREAM,
            GstWebRTCStatsType.RTP_INBOUND_STREAM,
            GstWebRTCStatsType.ICE_CANDIDATE_PAIR,
        ]

        self.min_delay = 0.0

        self.reset()

    def reset(self):
        super().reset()
        self.delays = []

    def create_observation_space(self) -> spaces.Dict:
        shape = (self.num_observations_for_state,)
        return spaces.Dict(
            {
                "00_RECV_RATE": spaces.Box(
                    low=0,
                    high=self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"] * 1e6,
                    shape=shape,
                    dtype=np.float32,
                ),  # in bps
                "03_QUEUING_DELAY": spaces.Box(
                    low=0, high=self.CONSTANTS["MAX_DELAY_SEC"] * 1000, shape=shape, dtype=np.float32
                ),  # in ms
                "04_DELAY": spaces.Box(
                    low=-200, high=self.CONSTANTS["MAX_DELAY_SEC"] * 1000, shape=shape, dtype=np.float32
                ),  # -200 is substracted in the training data
                "05_MIN_SEEN_DELAY": spaces.Box(
                    low=0, high=self.CONSTANTS["MAX_DELAY_SEC"] * 1000, shape=shape, dtype=np.float32
                ),
                "07_DELAY_MIN_DIFF": spaces.Box(
                    low=0, high=self.CONSTANTS["MAX_DELAY_SEC"] * 1000, shape=shape, dtype=np.float32
                ),
                "09_PKT_JITTER": spaces.Box(
                    low=0, high=self.CONSTANTS["MAX_DELAY_SEC"] * 1000, shape=shape, dtype=np.float32
                ),
                "10_PKT_LOSS_RATIO": spaces.Box(low=0, high=1, shape=shape, dtype=np.float32),
            }
        )

    def create_action_space(self) -> spaces.Space:
        # basic AS uses only bitrate
        return spaces.Box(
            low=self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"] * 1e6,
            high=self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"] * 1e6,
            shape=(1,),
            dtype=np.float32,
        )

    def make_default_state(self) -> OrderedDict[str, Any]:
        def_val = 0.0 if self.num_observations_for_state == 1 else [0.0] * self.num_observations_for_state
        return collections.OrderedDict(
            {
                "00_RECV_RATE": def_val,
                "03_QUEUING_DELAY": def_val,
                "04_DELAY": def_val,
                "05_MIN_SEEN_DELAY": def_val,
                "07_DELAY_MIN_DIFF": def_val,
                "09_PKT_JITTER": def_val,
                "10_PKT_LOSS_RATIO": def_val,
            }
        )

    def make_state(self, stats: Dict[str, Any], action: Any) -> OrderedDict[str, Any]:
        super().make_state(stats, action)

        # get dicts with needed stats
        rtp_outbound = find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        ice_candidate_pair = find_stat(stats, GstWebRTCStatsType.ICE_CANDIDATE_PAIR)
        if not rtp_outbound or not rtp_inbound or not ice_candidate_pair:
            return self.make_default_state()

        # get previous state for calculating fractional values
        last_rtp_outbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        last_rtp_inbound = (
            find_stat(stats, GstWebRTCStatsType.RTP_INBOUND_STREAM) if self.last_stats is not None else None
        )
        self.last_stats = stats

        if self.first_ssrc is None:
            # FIXME: make first ever ssrc to be the privileged one and take stats only from it
            self.first_ssrc = rtp_inbound[0]["ssrc"][0]

        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if rtp_inbound_ssrc["ssrc"][0] == self.first_ssrc:
                last_rtp_inbound_ssrc = last_rtp_inbound[i] if last_rtp_inbound is not None else None
                last_rtp_outbound_ssrc = last_rtp_outbound[0] if last_rtp_outbound is not None else None

                # get needed stats
                packets_sent_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                ts_diff_sec = [
                    ts / 1000 for ts in get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "timestamp")
                ]

                # 00_RECV_RATE
                rx_bytes_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "bytes-received")
                rx_bits_diff = [r * 8 for r in rx_bytes_diff]
                rx_rates = [r / ts if ts > 0 else 0.0 for r, ts in zip(rx_bits_diff, ts_diff_sec)]
                # NOTE: it is important to reverse all the lists to get the correct order of observations
                rx_rates.reverse()
                rx_rates_final = [
                    get_list_average(mi, is_skip_zeroes=True)
                    for mi in slice_list_in_intervals(rx_rates, self.num_observations_for_state, 'sliding')
                ]

                # 04_DELAY
                delays_raw_ms = [
                    ntp_short_format_to_seconds(rtt) * 1000 / 2  # ms
                    for rtt in rtp_inbound[i]["rb-round-trip"]  # roughly assuming d = rtt / 2
                ]
                delays_raw_ms.reverse()
                self.delays.extend(delays_raw_ms)
                delays_shifted = [d - 200 for d in delays_raw_ms]
                delays_final = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(delays_shifted, self.num_observations_for_state, 'sliding')
                ]

                # 05_MIN_SEEN_DELAY
                min_seen_delays = []
                for d in delays_raw_ms:
                    if self.min_delay == 0.0:
                        self.min_delay = d
                    if d < self.min_delay and d > 0:
                        self.min_delay = d
                    min_seen_delays.append(self.min_delay)

                min_seen_delays_final = [
                    min(mi)
                    for mi in slice_list_in_intervals(min_seen_delays, self.num_observations_for_state, 'sliding')
                ]

                # 03_QUEUING_DELAY
                av_delays_raw = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(delays_raw_ms, self.num_observations_for_state, 'sliding')
                ]
                queuing_delays_final = [
                    av_delay - min_seen_delay for av_delay, min_seen_delay in zip(av_delays_raw, min_seen_delays_final)
                ]

                # 07_DELAY_MIN_DIFF
                delay_min_diffs = [
                    av_delay - min(delays_raw)
                    for av_delay, delays_raw in zip(
                        av_delays_raw,
                        slice_list_in_intervals(delays_raw_ms, self.num_observations_for_state, 'sliding'),
                    )
                ]

                # 09_PKT_JITTER
                jitters_raw = [
                    clock_units_to_seconds(j, rtp_outbound[0]["clock-rate"][0]) * 1000  # ms
                    for j in rtp_inbound[i]["rb-jitter"]
                ]
                jitters_raw.reverse()
                jitters_final = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(jitters_raw, self.num_observations_for_state, 'sliding')
                ]

                # 10_PKT_LOSS_RATIO
                packets_sent_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                rb_packetslost_diff = get_stat_diff_concat(rtp_inbound[i], last_rtp_inbound_ssrc, "rb-packetslost")
                loss_rates = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rb_packetslost_diff, packets_sent_diff)
                ]
                loss_rates.reverse()
                loss_rates_final = [
                    get_list_average(mi)
                    for mi in slice_list_in_intervals(loss_rates, self.num_observations_for_state, 'sliding')
                ]

                # form the final state
                state = collections.OrderedDict(
                    {
                        "00_RECV_RATE": rx_rates_final,
                        "03_QUEUING_DELAY": queuing_delays_final,
                        "04_DELAY": delays_final,
                        "05_MIN_SEEN_DELAY": min_seen_delays_final,
                        "07_DELAY_MIN_DIFF": delay_min_diffs,
                        "09_PKT_JITTER": jitters_final,
                        "10_PKT_LOSS_RATIO": loss_rates_final,
                    }
                )

                self.last_states.append(state)
                self.update_reward_params()

                return state

        LOGGER.warning("WARNING: Drl Agent.make_state: no ssrc stats found")
        return self.make_default_state()

    def convert_to_unscaled_state(self, state: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
        return state

    def convert_to_unscaled_action(self, action: np.ndarray | float | int) -> np.ndarray | float:
        # action comes usually as np array in bps
        if isinstance(action, np.ndarray):
            a = action[0] / 1e6  # mbps
            a_final = min(self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"], max(self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], a))
            return a_final * 1e3  # kbps
        else:
            a = action / 1e6
            a_final = min(self.CONSTANTS["MAX_BITRATE_STREAM_MBPS"], max(self.CONSTANTS["MIN_BITRATE_STREAM_MBPS"], a))
            return a_final * 1e3

    def pack_action_for_controller(self, action: Any) -> Dict[str, Any]:
        return {"bitrate": self.convert_to_unscaled_action(action)}

    def update_reward_params(self) -> None:
        super().update_reward_params()
        self.reward_params["max_delay"] = max(self.delays) if self.delays else 0.0
