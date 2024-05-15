import json
import time
from typing import Any, Dict

from control.agent import Agent, AgentType
from control.safety.switcher import Switcher, SwitcherConfig
from message.client import MqttConfig
from utils.base import (
    LOGGER,
    merge_observations,
    sleep_until_condition_with_intervals,
    get_list_average,
    slice_list_in_intervals,
)
from utils.gst import GstWebRTCStatsType, find_stat, get_stat_diff_concat, is_same_rtcp
from utils.webrtc import clock_units_to_seconds, ntp_short_format_to_seconds


class SafetyDetectorAgent(Agent):
    STATS_KEYS = ["fractionLossRate", "fractionNackRate", "fractionPliRate", "fractionRtt", "fractionJitter"]

    def __init__(
        self,
        mqtt_config: MqttConfig,
        switcher_configs: Dict[str, SwitcherConfig],
        id: str = "safety_detector",
        switch_update_interval: float = 1.0,
        max_inactivity_time: float = 60.0,
        warmup: float = 10.0,
    ) -> None:
        super().__init__(mqtt_config, id, warmup)
        self.type = AgentType.SAFETY_DETECTOR

        self.stats = []
        self.last_gst_stats = None
        self.first_ssrc = None

        if not switcher_configs:
            raise ValueError("SafetyDetectorAgent: No switcher configs provided")
        self.switchers = {key: Switcher(config) for key, config in switcher_configs.items() if key in self.STATS_KEYS}
        if not self.switchers:
            LOGGER.warning("WARNING: SafetyDetectorAgent: No switchers were created. False keys in a config dict?")

        self.switch_update_interval = switch_update_interval
        self.max_inactivity_time = max_inactivity_time

        self.algo = 1  # default "unsafe"

    def run(self, _) -> None:
        super().run()
        time.sleep(self.warmup)
        self.mqtts.subscriber.clean_message_queue(self.mqtts.subscriber.topics.stats)
        self.mqtts.subscriber.subscribe([self.mqtts.subscriber.topics.state])
        self.is_running = True
        LOGGER.info(f"INFO: SafetyDetectorAgent is starting...")

        while self.is_running:
            final_stats = self._cook_stats()
            if final_stats is not None:
                self._decide_on_switch(final_stats)

    def _cook_stats(self) -> Dict[str, float] | None:
        is_stopped = sleep_until_condition_with_intervals(10, self.switch_update_interval, lambda: not self.is_running)
        if is_stopped:
            return None

        time_inactivity_starts = time.time()
        final_stats_dict = {}
        if self.algo == 0:
            # if safe is running, collect the raw gst stats by ourselves
            gst_stats = []
            while not self.mqtts.subscriber.message_queues[self.mqtts.subscriber.topics.stats].empty():
                gst_stats_mqtt = self.mqtts.subscriber.get_message(self.mqtts.subscriber.topics.stats)
                if gst_stats_mqtt is None:
                    if time.time() - time_inactivity_starts > self.max_inactivity_time:
                        LOGGER.warning(
                            "WARNING: SafetyDetectorAgent: No stats were pulled from the observation queue after"
                            f" {self.max_inactivity_time} sec"
                        )
                        self.is_running = False
                        return None
                else:
                    gst_stats.append(json.loads(gst_stats_mqtt.msg))
            if gst_stats:
                merged_stats = merge_observations(gst_stats)
                final_stats_dict = self._select_raw_stats(merged_stats)
        else:
            # if unsafe is running, just collect the unsafe agent's state from the environment
            state_mqtt = None
            while state_mqtt is None:
                state_mqtt = self.mqtts.subscriber.get_message(self.mqtts.subscriber.topics.state)
                if time.time() - time_inactivity_starts > self.max_inactivity_time:
                    LOGGER.warning(
                        "WARNING: SafetyDetectorAgent: No state was received from the unsafe agent after"
                        f" {self.max_inactivity_time} sec"
                    )
                    self.is_running = False
                    return None
            state = json.loads(state_mqtt.msg)
            for key in self.switchers.keys():
                if key in state:
                    if isinstance(state[key], list):
                        final_stats_dict[key] = get_list_average(state[key])
                    else:
                        final_stats_dict[key] = float(state[key])
                else:
                    LOGGER.warning(f"WARNING: SafetyDetectorAgent: Key {key} not found in the state")

        return final_stats_dict or None

    def _select_raw_stats(self, gst_stats: Dict[str, Any]) -> Dict[str, float] | None:
        rtp_outbound = find_stat(gst_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(gst_stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        if not rtp_outbound or not rtp_inbound:
            LOGGER.info("WARNING: SafetyDetectorAgent: no stats were found...")
            return None

        # last stats
        last_rtp_outbound = (
            find_stat(self.last_gst_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
            if self.last_gst_stats is not None
            else None
        )
        last_rtp_inbound = (
            find_stat(self.last_gst_stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
            if self.last_gst_stats is not None
            else None
        )
        if last_rtp_outbound is None or last_rtp_inbound is None:
            self.last_gst_stats = gst_stats
            return None

        if self.first_ssrc is None:
            self.first_ssrc = rtp_inbound[0]["ssrc"][0]

        # len(rtp_inbound) = number of viewers. Iterate by their ssrc
        # outbound stats are the same for all viewers
        final_stats = {}
        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if rtp_inbound_ssrc["ssrc"][0] == self.first_ssrc:
                if 0 <= i < len(last_rtp_inbound) and is_same_rtcp(rtp_inbound_ssrc, last_rtp_inbound[i]):
                    continue
                last_rtp_inbound_ssrc = last_rtp_inbound[i] if last_rtp_inbound is not None else None
                last_rtp_outbound_ssrc = last_rtp_outbound[0] if last_rtp_outbound is not None else None

                # get needed stats
                packets_sent_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-sent")
                packets_recv_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "packets-received")

                # fraction loss rate
                rb_packetslost_diff = get_stat_diff_concat(rtp_inbound[i], last_rtp_inbound_ssrc, "rb-packetslost")
                fraction_loss_rates_raw = [
                    lost / (sent + lost) if sent + lost > 0 else 0.0
                    for lost, sent in zip(rb_packetslost_diff, packets_sent_diff)
                ]
                fraction_loss_rates = [
                    get_list_average(mi) for mi in slice_list_in_intervals(fraction_loss_rates_raw, 5)
                ]
                fraction_loss_rate = get_list_average(fraction_loss_rates)

                # nack rate
                recv_nack_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "nack-count")
                fraction_nack_rates_raw = [
                    nack / recv if recv > 0 else 0.0 for nack, recv in zip(recv_nack_count_diff, packets_recv_diff)
                ]
                fraction_nack_rates = [
                    get_list_average(mi) for mi in slice_list_in_intervals(fraction_nack_rates_raw, 5)
                ]
                fraction_nack_rate = get_list_average(fraction_nack_rates)

                # pli rate
                recv_pli_count_diff = get_stat_diff_concat(rtp_outbound[0], last_rtp_outbound_ssrc, "pli-count")
                fraction_pli_rates_raw = [
                    pli / recv if recv > 0 else 0.0 for pli, recv in zip(recv_pli_count_diff, packets_recv_diff)
                ]
                fraction_pli_rates = [get_list_average(mi) for mi in slice_list_in_intervals(fraction_pli_rates_raw, 5)]
                fraction_pli_rate = get_list_average(fraction_pli_rates)

                # rtt
                rtts_raw = [ntp_short_format_to_seconds(rtt) for rtt in rtp_inbound[i]["rb-round-trip"]]
                rtts = [get_list_average(mi) for mi in slice_list_in_intervals(rtts_raw, 5)]
                rtt = get_list_average(rtts)

                # jitter
                jitters_raw = [
                    clock_units_to_seconds(j, rtp_outbound[0]["clock-rate"][0]) for j in rtp_inbound[i]["rb-jitter"]
                ]
                jitters = [get_list_average(mi) for mi in slice_list_in_intervals(jitters_raw, 5)]
                jitter = get_list_average(jitters)

                # form the final state with the keys defined for switchers
                final_stats = {
                    "fractionLossRate": fraction_loss_rate,
                    "fractionNackRate": fraction_nack_rate,
                    "fractionPliRate": fraction_pli_rate,
                    "fractionRtt": rtt,
                    "fractionJitter": jitter,
                }
                final_stats = {key: final_stats[key] for key in self.switchers.keys()}

        self.last_gst_stats = gst_stats
        return final_stats

    def _decide_on_switch(self, new_stats: Dict[str, float]) -> None:
        switch_algos = []
        for key, switcher in self.switchers.items():
            switch_algos.append(switcher.switch(new_stats[key]))
        if not all(algo == self.algo for algo in switch_algos):
            self.algo = 1 if self.algo == 0 else 0
            self.mqtts.publisher.publish(self.mqtts.publisher.topics.actions, json.dumps({"switch": self.algo}))
            self.reset()
            LOGGER.info(f"INFO: AgentSwitcher: Switching agent to {'safe' if self.algo == 0 else 'unsafe'}")

    def reset(self) -> None:
        self.last_gst_stats = None

    def stop(self) -> None:
        super().stop()
        LOGGER.info(f"INFO: SafetyDetectorAgent is stopping...")
