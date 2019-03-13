from flame.engine import Engine, Event, BaseContext
from flame.metrics.metric import Metric


class ReduceMetric(Metric):
    def __init__(self, name, reduce=None, initial_value=None, context_map=None,
                 reset_event=Event.PHASE_STARTED, update_event=Event.ITER_COMPLETED):
        super(ReduceMetric, self).__init__(name, False, context_map, reset_event, update_event, False)
        self.reduce = reduce
        self.initial_value = initial_value
        self._value = initial_value

    def reset(self, *args, **kwargs):
        self._value = self.initial_value

    def update(self, value):
        self._value = self.reduce(self._value, value)

    @property
    def value(self):
        return self._value
