import asyncio

from ahoyapp.app import GstWebRTCBinAppConfig
from ahoyapp.connector import AhoyConnector
from utils.utils import LOGGER

try:
    import uvloop
except ImportError:
    uvloop = None

AHOY_DIRECTOR_URL = "..."
API_KEY = "..."
VIDEO_SOURCE = "rtsp://..."


async def manipulate_video_coro(conn: AhoyConnector):
    while conn.app is None:
        await asyncio.sleep(0.5)
    await asyncio.sleep(20)
    LOGGER.info("----------Run test video manipulating coro...")
    LOGGER.info("----------Setting new bitrate...")
    conn.app.set_bitrate(1000)
    await asyncio.sleep(20)
    LOGGER.info("----------Setting new resolution...")
    conn.app.set_resolution(192, 144)
    await asyncio.sleep(20)
    LOGGER.info("----------Setting new framerate...")
    conn.app.set_framerate(15)
    await asyncio.sleep(20)
    LOGGER.info("----------Setting new fec...")
    conn.app.set_fec_percentage(50)
    await asyncio.sleep(20)
    LOGGER.info("----------Finished test video manipulating coro...")


async def test_manipulate_video():
    # run it to test video manipulation.
    try:
        cfg = GstWebRTCBinAppConfig(video_url=VIDEO_SOURCE)

        conn = AhoyConnector(
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            pipeline_config=cfg,
        )

        await conn.connect_coro()

        # note that a new coroutine is created and awaited together with webrtc_coro to manipulate the video
        conn_task = asyncio.create_task(conn.webrtc_coro())
        pipeline_task = asyncio.create_task(manipulate_video_coro(conn))
        await asyncio.gather(conn_task, pipeline_task)

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


async def main():
    # run it as the default endpoint
    try:
        cfg = GstWebRTCBinAppConfig(video_url=VIDEO_SOURCE)

        conn = AhoyConnector(
            server=AHOY_DIRECTOR_URL,
            api_key=API_KEY,
            pipeline_config=cfg,
        )

        await conn.connect_coro()

        await conn.webrtc_coro()

    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, exiting...")
        return


if __name__ == "__main__":
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())
