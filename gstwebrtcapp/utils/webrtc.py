def clock_units_to_seconds(clock_units: int, clock_rate: int = 90000) -> float:
    return float(clock_units) / clock_rate


def ntp_short_format_to_seconds(ntp_short_format: int) -> float:
    seconds = (ntp_short_format >> 16) + ((ntp_short_format & 0xFFFF) / 2**16)
    return seconds
