import typing

from flame.engine import Engine, Event, BaseContext
from flame.metrics.metric import Metric

T = typing.TypeVar("T")


class ReduceMetric(Metric):
    def __init__(self, name: str, reduce: typing.Callable[[T, T], T] = None,
                 initial_value: T = None, context_map: typing.Callable[[BaseContext], typing.Any] = None,
                 reset_event: Event = Event.PHASE_STARTED, update_event: Event = Event.ITER_COMPLETED):
        super(ReduceMetric, self).__init__(name, False, context_map, reset_event, update_event, False)
        self.reduce = reduce
        self.initial_value = initial_value
        self._value = initial_value

    def reset(self, *args, **kwargs) -> None:
        self._value = self.initial_value

    def update(self, value) -> None:
        self._value = self.reduce(self._value, value)

    @property
    def value(self) -> T:
        return self._value
