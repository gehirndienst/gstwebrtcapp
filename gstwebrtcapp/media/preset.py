from dataclasses import dataclass
import enum


# TODO: add encoder parameters if needed
@dataclass
class VideoPreset:
    name: str
    width: int
    height: int
    framerate: int
    bitrate: int  # kbps


class VideoPresets(enum.Enum):
    # 270p
    LOW = VideoPreset("low", 640, 360, 15, 400)
    # 360p
    MEDIUM = VideoPreset("medium", 640, 360, 20, 1000)
    # high
    HIGH = VideoPreset("high", 1280, 720, 20, 1500)
    # hd
    HD = VideoPreset("hd", 1280, 720, 20, 2500)
    # full hd
    FHD = VideoPreset("fhd", 1920, 1080, 20, 4000)
    # 4k
    UHD = VideoPreset("uhd", 1920, 1080, 20, 8000)


def get_video_preset(get_by: str | int) -> VideoPreset:
    if isinstance(get_by, int):
        return list(VideoPresets)[get_by].value
    elif isinstance(get_by, str):
        return VideoPresets[get_by.upper()].value
    else:
        raise ValueError("Invalid type for get_preset parameter. Use either int (preset index) or str (preset name).")
