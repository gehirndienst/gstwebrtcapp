import asyncio
import logging
import re
import time
from typing import Any, Callable, Dict

# logger
LOGGER = logging
LOGGER.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# exceptions
class GSTWEBRTCAPP_EXCEPTION(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)

    def __str__(self):
        return f"{self.args[0]}"


class APPBIN_EXCEPTION(GSTWEBRTCAPP_EXCEPTION):
    pass


class CONNECTOR_EXCEPTION(GSTWEBRTCAPP_EXCEPTION):
    pass


class CANCEL_CORO_EXCEPTION(GSTWEBRTCAPP_EXCEPTION):
    pass


# utils
def wait_for_condition(condition_func: Callable[[], bool], timeout_sec: int) -> bool:
    start_time = time.time()
    while not condition_func():
        if time.time() - start_time >= float(timeout_sec):
            raise APPBIN_EXCEPTION(f"Timeout {timeout_sec} sec is reached for condition {condition_func.__name__}")
    return True


async def async_wait_for_condition(condition_func: Callable[[], bool], timeout_sec: int) -> bool:
    start_time = time.time()
    while not condition_func():
        if time.time() - start_time >= float(timeout_sec):
            raise APPBIN_EXCEPTION(f"Timeout {timeout_sec} sec is reached for condition {condition_func.__name__}")
        await asyncio.sleep(0.5)
    return True


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
