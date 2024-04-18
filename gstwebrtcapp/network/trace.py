from dataclasses import dataclass
from typing import List


@dataclass
class NetworkTrace:
    size: int
    av_value: float
    ooc_rate: float
    values: List[float]
