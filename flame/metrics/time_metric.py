import time

from flame.metrics.metric import Metric
from flame.engine import Engine, Event


class TimeMetric(Metric):
    def __init__(self, name: str, reset_event: Event = Event.PHASE_STARTED, update_event: Event = Event.ITER_COMPLETED):
        super(TimeMetric, self).__init__(name, False, lambda x: x, reset_event, update_event, False)

        self.start_time = None
        self.reset()

    def reset(self, *args, **kwargs) -> None:
        self.start_time = time.time()

    @property
    def value(self) -> float:
        return time.time() - self.start_time
