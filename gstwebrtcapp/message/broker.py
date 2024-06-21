import os
import signal
import subprocess

from gstwebrtcapp.utils.base import LOGGER


class MosquittoBroker:
    def __init__(
        self,
        port: int = 1883,
    ):
        self.port = port
        self.process = None
        self.is_running = False

    def run(self) -> None:
        if not os.path.exists("/etc/mosquitto/mosquitto.conf"):
            self._generate_default_conf_file()
        cmd = ["mosquitto", "-c", "/etc/mosquitto/mosquitto.conf", "-p", str(self.port)]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        while self.process.poll() is None:
            if not self.is_running:
                self.is_running = True
                LOGGER.info(f"INFO: Mosquitto broker has been started")

        self.stop()

    def stop(self) -> None:
        if self.process and self.process.returncode is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait()
            except ProcessLookupError:
                pass

    def _generate_default_conf_file(self) -> None:
        with open("/etc/mosquitto/mosquitto.conf", 'w') as config_file:
            config_lines = [
                f"pid_file /run/mosquitto/mosquitto.pid",
                "persistence true",
                f"persistence_location /var/lib/mosquitto/",
                f"log_dest stdout",
                f"connection_messages True",
                f"listener {self.port}",
                f"allow_anonymous True",
                f"log_dest file /var/log/mosquitto/mosquitto.log",
                f"include_dir /etc/mosquitto/conf.d",
            ]
            config_file.write('\n'.join(config_lines))
