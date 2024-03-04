import csv
from datetime import datetime
import json
import os
import time
from typing import Any, Dict, List

from control.agent import Agent, AgentType
from message.client import MqttConfig
from utils.base import LOGGER
from utils.gst import GstWebRTCStatsType, find_stat, get_stat_diff, is_same_rtcp
from utils.webrtc import clock_units_to_seconds, ntp_short_format_to_seconds


class CsvViewerRecorderAgent(Agent):
    def __init__(
        self,
        mqtt_config: MqttConfig,
        stats_update_interval: float = 1.0,
        warmup: float = 3.0,
        log_path: str = "./logs",
        max_inactivity_time: float = 5.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(mqtt_config)
        self.stats_update_interval = stats_update_interval
        self.warmup = warmup
        self.log_path = log_path
        self.max_inactivity_time = max_inactivity_time
        self.verbose = min(verbose, 2)
        self.type = AgentType.RECORDER

        # cooked stats
        self.stats = []
        # raw gst stats
        self.last_stats = None

        self.csv_handler = None
        self.csv_writer = None

        self.is_running = False

    def run(self, _) -> None:
        super().run()
        time.sleep(self.warmup)
        # clean the queue from the messages obtained before warmup
        self.mqtts.subscriber.clean_message_queue(self.mqtts.subscriber.topics.stats)
        self.is_running = True
        LOGGER.info(f"INFO: Csv Viewer Recorder agent warmup {self.warmup} sec is finished, starting...")

        while self.is_running:
            gst_stats_collected = self._fetch_stats()
            if gst_stats_collected is not None:
                for gst_stats in gst_stats_collected:
                    is_stats = self._select_stats(gst_stats)
                    if is_stats and self.verbose > 0:
                        if self.verbose == 1:
                            LOGGER.info(f"INFO: Browser Recorder agent stats:\n {self.stats[-1]}")
                        elif self.verbose == 2:
                            self._save_stats_to_csv()

    def _fetch_stats(self) -> List[Dict[str, Any]] | None:
        time.sleep(self.stats_update_interval)
        time_inactivity_starts = time.time()
        stats = []
        while not self.mqtts.subscriber.message_queues[self.mqtts.subscriber.topics.stats].empty():
            gst_stats = self.mqtts.subscriber.get_message(self.mqtts.subscriber.topics.stats)
            if gst_stats is None:
                if time.time() - time_inactivity_starts > self.max_inactivity_time:
                    LOGGER.warning(
                        "WARNING: No stats were pulled from the observation queue after"
                        f" {self.max_inactivity_time} sec"
                    )
                    return None
            else:
                stats.append(json.loads(gst_stats.msg))
        return stats

    def _select_stats(self, gst_stats: Dict[str, Any]) -> bool:
        rtp_outbound = find_stat(gst_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_inbound = find_stat(gst_stats, GstWebRTCStatsType.RTP_INBOUND_STREAM)
        ice_candidate_pair = find_stat(gst_stats, GstWebRTCStatsType.ICE_CANDIDATE_PAIR)
        if not rtp_outbound or not rtp_inbound or not ice_candidate_pair:
            LOGGER.info("WARNING: Csv Viewer Recorder agent: no stats were found...")
            return False

        # last stats
        last_rtp_outbound = (
            find_stat(self.last_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        last_rtp_inbound = (
            find_stat(self.last_stats, GstWebRTCStatsType.RTP_INBOUND_STREAM) if self.last_stats is not None else None
        )
        if last_rtp_outbound is None or last_rtp_inbound is None:
            self.last_stats = gst_stats
            return False

        n_stats = len(self.stats)

        # len(rtp_inbound) = number of viewers. Iterate by their ssrc
        # outbound stats are the same for all viewers
        for i, rtp_inbound_ssrc in enumerate(rtp_inbound):
            if 0 <= i < len(last_rtp_inbound) and is_same_rtcp(rtp_inbound_ssrc, last_rtp_inbound[i]):
                continue
            # ssrc
            ssrc = rtp_inbound_ssrc["ssrc"]

            # loss rate
            loss_rate = (
                float(rtp_inbound_ssrc["rb-packetslost"]) / rtp_outbound[0]["packets-sent"]
                if rtp_outbound[0]["packets-sent"] > 0
                else 0.0
            )

            ts_diff_sec = get_stat_diff(rtp_outbound[0], last_rtp_outbound[0], "timestamp") / 1000

            # fraction tx rate in Mbits
            try:
                bitrate_sent = ice_candidate_pair[0]["bitrate-sent"]
                tx_rate = bitrate_sent / 1000000
            except KeyError:
                tx_bytes_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound[0], "bytes-sent")
                tx_mbits_diff = tx_bytes_diff * 8 / 1000000
                tx_rate = tx_mbits_diff / ts_diff_sec if ts_diff_sec > 0 else 0.0

            # fraction rx rate in Mbits
            try:
                bitrate_recv = ice_candidate_pair[0]["bitrate-recv"]
                rx_rate = bitrate_recv / 1000000
            except KeyError:
                rx_bytes_diff = get_stat_diff(rtp_outbound[0], last_rtp_outbound[0], "bytes-received")
                rx_mbits_diff = rx_bytes_diff * 8 / 1000000
                rx_rate = rx_mbits_diff / ts_diff_sec if ts_diff_sec > 0 else 0.0

            # rtts / jitter
            rtt_ms = ntp_short_format_to_seconds(rtp_inbound_ssrc["rb-round-trip"]) * 1000
            last_rtt_ms = (
                ntp_short_format_to_seconds(last_rtp_inbound[i]["rb-round-trip"]) * 1000
                if 0 <= i < len(last_rtp_inbound)
                else 0.0
            )
            gradient_rtt_ms = rtt_ms - last_rtt_ms
            jitter_ms = clock_units_to_seconds(rtp_inbound_ssrc["rb-jitter"], rtp_outbound[0]["clock-rate"]) * 1000

            # opened to extensions
            final_stats = {
                "timestamp": datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f")[:-3],
                "ssrc": ssrc,
                "fraction_packets_lost": rtp_inbound_ssrc["rb-fractionlost"],
                "packets_lost": rtp_inbound_ssrc["rb-packetslost"],
                "loss_rate_%": loss_rate,
                "ext_highest_seq": rtp_inbound_ssrc["rb-exthighestseq"],
                "rtt_ms": rtt_ms,
                "gradient_rtt_ms": gradient_rtt_ms,
                "jitter_ms": jitter_ms,
                "nack_count": rtp_outbound[0]["recv-nack-count"],
                "pli_count": rtp_outbound[0]["recv-pli-count"],
                "rx_packets": rtp_outbound[0]["packets-received"],
                "rx_mbytes": rtp_outbound[0]["bytes-received"] / 1000000,
                "tx_rate_mbits": tx_rate,
                "rx_rate_mbits": rx_rate,
            }
            self.stats.append(final_stats)

        self.last_stats = gst_stats
        return len(self.stats) > n_stats

    def _save_stats_to_csv(self) -> None:
        if self.csv_handler is None:
            datetime_now = datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")[:-3]
            os.makedirs(self.log_path, exist_ok=True)
            filename = os.path.join(self.log_path, f"webrtc_viewer_{datetime_now}.csv")
            header = self.stats[-1].keys()
            self.csv_handler = open(filename, mode="a", newline="\n")
            self.csv_writer = csv.DictWriter(self.csv_handler, fieldnames=header)
            if os.stat(filename).st_size == 0:
                self.csv_writer.writeheader()
            self.csv_handler.flush()
        else:
            for stat in self.stats:
                self.csv_writer.writerow(stat)
            self.stats = []

    def stop(self) -> None:
        super().stop()
        LOGGER.info("INFO: stopping Csv Viewer Recorder agent...")
        self.is_running = False
        if self.csv_handler is not None:
            self.csv_handler.close()
            self.csv_handler = None
            self.csv_writer = None
