import time

from flame.metrics.metric import Metric
from flame.engine import Engine, Event


class TimeMetric(Metric):
    def __init__(self, name, reset_event=Event.PHASE_STARTED, update_event=Event.ITER_COMPLETED):
        super(TimeMetric, self).__init__(name, False, lambda x: x, reset_event, update_event, False)

        self.start_time = None
        self.reset()

    def reset(self, *args, **kwargs):
        self.start_time = time.time()

    @property
    def value(self):
        return time.time() - self.start_time
