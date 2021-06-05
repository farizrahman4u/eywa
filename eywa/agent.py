from typing import *

class Agent(object):

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        raise NotImplementedError

    def get_response(self, input: str) -> str:
        return input

    def get_state(self) -> Any:
        raise NotImplementedError

    def set_state(self, state: Any) -> None:
        raise NotImplementedError

    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, config: Dict) -> "Agent":
        raise NotImplementedError

    def __call__(self, input: str) -> str:
        return self.get_response(input)
