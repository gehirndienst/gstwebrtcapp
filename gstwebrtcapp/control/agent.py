"""
agent.py

Description:
    Base Agent class for the gstreamerwebrtcapp to control the pipeline.

Author:
    - Nikita Smirnov <nsm@informatik.uni-kiel.de>

License:
    GPLv3 License

"""

from abc import ABCMeta, abstractmethod
from enum import Enum

from control.controller import Controller


class AgentType(Enum):
    ABSTRACT = "ABSTRACT"
    CC = "CC"
    DRL = "DRL"
    RECORDER = "RECORDER"


class Agent(metaclass=ABCMeta):
    def __init__(self, controller: Controller) -> None:
        assert controller is not None, "Agent must have a controller!"
        self.controller = controller
        self.type = AgentType.ABSTRACT

    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass
