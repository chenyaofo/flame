from abc import abstractmethod

from flame._utils import check
from flame.engine import Engine


class Handler(object):
    def __init__(self, name):
        self.name = check(name, "name", str)

    @abstractmethod
    def attach(self, engine: Engine):
        pass
