import typing

from flame._utils import check
from flame.engine import Engine, Event, BaseContext

eps = 1e-15


class Metric(object):
    def reset(self):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def compute(self):
        return None
