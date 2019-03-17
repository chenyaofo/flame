import torch
import typing
from flame.engine import Engine, Event, BaseContext
from flame.metrics.metric import Metric


class AverageMetric(Metric):
    def __init__(self, name: str, context_map: typing.Callable[[BaseContext], typing.Any] = None,
                 reset_event: Event = Event.PHASE_STARTED, update_event: Event = Event.ITER_COMPLETED):
        super(AverageMetric, self).__init__(name, False, context_map, reset_event, update_event, False)
        self.n = 0
        self._value = 0.

    def reset(self, *args, **kwargs) -> None:
        self.n = 0
        self._value = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self.n += 1
            self._value += value.item()
        elif isinstance(value, (int, float)):
            self.n += 1
            self._value += value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))

    @property
    def value(self) -> float:
        if self.n == 0:
            return 0.0
        return self._value / self.n
