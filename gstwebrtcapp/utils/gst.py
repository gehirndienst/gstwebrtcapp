from enum import Enum
import re
from typing import Any, Dict, List


# encoder
CODECS = ["h264", "h265", "vp8", "vp9", "av1"]
GST_ENCODERS = ["x264enc", "nvh264enc", "x265enc", "vp8enc", "vp9enc", "av1enc"]


def get_gst_encoder_name(codec: str, is_cuda: bool = False) -> str:
    # TODO: add cuda support for other codecs later
    if codec == "h264":
        return "x264enc" if not is_cuda else "nvh264enc"
    elif codec == "h265":
        return "x265enc"
    elif codec == "vp8":
        return "vp8enc"
    elif codec == "vp9":
        return "vp9enc"
    elif codec == "av1":
        return "av1enc" if not is_cuda else "nvav1enc"
    else:
        raise ValueError(f"Invalid codec: {codec}")


# gcc
DEFAULT_GCC_SETTINGS = {
    "min-bitrate": 400000,  # 0.4 mbps
    "max-bitrate": 20000000,  # 20 mbps
}


# stats
class GstWebRTCStatsType(Enum):
    CODEC = "codec"
    ICE_CANDIDATE_LOCAL = "ice-candidate-local"
    ICE_CANDIDATE_REMOTE = "ice-candidate-remote"
    ICE_CANDIDATE_PAIR = "ice-candidate-pair"
    TRANSPORT = "transport"
    RTP_REMOTE_INBOUND_STREAM = "rtp-remote-inbound-stream"
    RTP_REMOTE_OUTBOUND_STREAM = "rtp-remote-outbound-stream"
    RTP_INBOUND_STREAM = "rtp-inbound-stream"
    RTP_OUTBOUND_STREAM = "rtp-outbound-stream"


def stats_to_dict(input_stats_string: str) -> Dict[str, Any]:
    return _cast_stat_dict(_parse_stat_string(input_stats_string))


def _parse_stat_string(input_string: str) -> Dict[str, Any]:
    input_string = input_string.strip(';').strip().replace(">", '').replace(";", '').replace('"', '').replace("\\", '')
    pairs = re.split(r',\s+', input_string)
    result_dict = {}
    for pair in pairs:
        pair_parts = pair.split('=', 1)
        if len(pair_parts) == 2:
            key, value = pair_parts
            if value.startswith("(structure)"):
                value = _parse_stat_string(value[len("(structure)") :])
                result_dict = result_dict | value
            else:
                result_dict[key] = value
    return result_dict


def _cast_stat_dict(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    cast_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, str) and value.startswith("(") and ")" in value:
            data_type, raw_value = re.match(r'\(([^)]*)\)(.*)', value).groups()
            try:
                if data_type == 'string':
                    cast_dict[key] = raw_value
                elif data_type == 'double':
                    cast_dict[key] = float(raw_value)
                elif (
                    data_type == 'int'
                    or data_type == 'uint'
                    or data_type == 'gint'
                    or data_type == 'int64'
                    or data_type == 'uint64'
                    or data_type == 'guint64'
                    or data_type == 'long'
                ):
                    cast_dict[key] = int(raw_value)
                elif data_type == 'boolean':
                    cast_dict[key] = raw_value.lower() in ['true', '1']
                elif data_type == 'GstWebRTCStatsType':
                    cast_dict[key] = str(raw_value)
            except ValueError:
                print(f"Failed to cast {key}={value} to {data_type}")
                cast_dict[key] = None
    return cast_dict


def find_stat(stats: Dict[str, Any], stat: GstWebRTCStatsType) -> List[Dict[str, Any]]:
    res = []
    for key in stats:
        if key.startswith(stat.value):
            res.append(stats[key])
    return res


def get_stat_diff(stats: Dict[str, Any], last_stats: Dict[str, Any] | None, stat: str) -> float | int:
    return stats[stat] - last_stats[stat] if last_stats is not None else stats[stat]


def get_stat_diff_concat(
    stats: Dict[str, List[Any]], last_stats: Dict[str, List[Any]] | None, stat: str
) -> List[float | int]:
    last_value = last_stats[stat][-1] if last_stats is not None else 0.0
    res = []
    for i in range(len(stats[stat])):
        res.append(stats[stat][i] - stats[stat][i - 1] if i > 0 else stats[stat][i] - last_value)
    return res


def is_same_rtcp(rtp_inbound: Dict[str, Any], last_rtp_inbound: Dict[str, Any] | None) -> bool:
    if last_rtp_inbound is None:
        return False
    else:
        # check if rtt and jitter are absolutely the same
        return (
            rtp_inbound["rb-round-trip"] == last_rtp_inbound["rb-round-trip"]
            and rtp_inbound["rb-jitter"] == last_rtp_inbound["rb-jitter"]
        )
