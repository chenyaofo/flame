from abc import abstractmethod

from flame._utils import check
from flame.engine import Engine


class Handler(object):
    def __init__(self, name: str):
        self.name = check(name, "name", str)

    @abstractmethod
    def attach(self, engine: Engine):
        pass
