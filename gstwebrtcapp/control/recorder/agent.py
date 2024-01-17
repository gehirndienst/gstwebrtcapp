from collections import deque
import csv
import datetime
import os
import time
from typing import Any, Dict

from control.agent import Agent, AgentType
from control.controller import Controller
from utils.base import LOGGER
from utils.gst import GstWebRTCStatsType, find_stat, get_stat_diff
from utils.webrtc import clock_units_to_seconds, ntp_short_format_to_seconds

class BrowserRecorderAgent(Agent):
    def __init__(
        self,
        controller: Controller,
        stats_update_interval: float = 1.0,
        warmup: float = 3.0,
        log_path: str = "./logs",
        verbose: int = 0,
    ) -> None:
        super().__init__(controller)
        self.stats_update_interval = stats_update_interval
        self.warmup = warmup
        self.log_path = log_path
        self.verbose = min(verbose, 2)
        self.type = AgentType.RECORDER
        
        self.stats = deque(maxlen=10000)
        self.last_stats = None
        
        self.csv_handler = None
        self.csv_writer = None
        
        self.is_running = False

    def run(self, _) -> None:
        time.sleep(self.warmup)
        self.is_running = True
        LOGGER.info(f"INFO: Browser Recorder agent warmup {self.warmup} sec is finished, starting...")
        while self.is_running:
            gst_stats = self._fetch_stats()
            if gst_stats is not None:
                is_stats = self._select_stats(gst_stats)
                self.controller.clean_observation_queue()
                if is_stats and self.verbose > 0:
                    if self.verbose == 1:
                        LOGGER.info(f"INFO: Browser Recorder agent stats:\n {self.stats[-1]}")
                    elif self.verbose == 2:
                        self._save_stats_to_csv()
        
    def _fetch_stats(self) -> Dict[str, Any] | None:
        time.sleep(self.stats_update_interval)
        ticks = 0
        gst_stats = self.controller.get_observation()
        while gst_stats is None:
            time.sleep(0.1)
            ticks += 1
            if ticks > 10:
                LOGGER.info("WARNING: No stats were pulled from the observation queue after 1 second timeout...")
                return None
            else:
                gst_stats = self.controller.get_observation()
        return gst_stats
    
    def _select_stats(self, gst_stats: Dict[str, Any]) -> bool:
        rtp_outbound = find_stat(gst_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM)
        rtp_remote_inbound = find_stat(gst_stats, GstWebRTCStatsType.RTP_REMOTE_INBOUND_STREAM)
        if rtp_outbound is None or rtp_remote_inbound is None:
            LOGGER.info("WARNING: No RTP outbound or remote inbound stats were found in the GStreamer stats...")
            return False
        
        # fraction rx rate in Mbits
        last_rtp_outbound = (
            find_stat(self.last_stats, GstWebRTCStatsType.RTP_OUTBOUND_STREAM) if self.last_stats is not None else None
        )
        ts_diff_sec = get_stat_diff(rtp_outbound, last_rtp_outbound, "timestamp") / 1000
        rx_bytes_diff = get_stat_diff(rtp_outbound, last_rtp_outbound, "bytes-received")
        rx_mbits_diff = rx_bytes_diff * 8 / 1000000
        rx_rate = rx_mbits_diff / ts_diff_sec  if ts_diff_sec > 0 else 0.0
        
        # opened to extensions
        final_stats = {
            "timestamp": time.strftime("%Y-%m-%d-%H:%M:%S:%f")[:-3],
            "rr_fraction_loss": rtp_remote_inbound["rb-fractionlost"],
            "rr_packets_lost": rtp_remote_inbound["rb-packetslost"],
            "rr_ext_highest_seq": rtp_remote_inbound["rb-exthighestseq"],
            "rr_jitter_ms": clock_units_to_seconds(rtp_remote_inbound["rb-jitter"], rtp_outbound["clock-rate"]) * 1000,
            "rr_rtt_ms": ntp_short_format_to_seconds(rtp_remote_inbound["rb-round-trip"]) * 1000,
            "nack_count": rtp_outbound["recv-nack-count"],
            "pli_count": rtp_outbound["recv-pli-count"],
            "rx_packets": rtp_outbound["packets-received"],
            "rx_mbytes": rtp_outbound["bytes-received"] / 1000000,
            "tx_rate_mbits": rtp_outbound["bitrate"] / 1000000,
            "rx_rate_mbits": rx_rate,
        }
        
        self.stats.append(final_stats)
        self.last_stats = gst_stats
        return True
    
    def _save_stats_to_csv(self) -> None:
        if self.csv_handler is None:
            datetime_now = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            os.makedirs(self.log_path, exist_ok=True)
            filename = os.path.join(self.log_path, f"webrtc_browser_{datetime_now}.csv")
            header = self.stats[-1].keys()
            self.csv_handler = open(filename, mode="a", newline="\n")
            self.csv_writer = csv.DictWriter(self.csv_handler, fieldnames=header)
            if os.stat(filename).st_size == 0:
                self.csv_writer.writeheader()
            self.csv_handler.flush()
        else:
            self.csv_writer.writerow(self.stats[-1])
        

    def stop(self) -> None:
        LOGGER.info("INFO: stopping Browser Recorder agent...")
        self.is_running = False
        if self.csv_handler is not None:
            self.csv_handler.close()
            self.csv_handler = None
            self.csv_writer = None