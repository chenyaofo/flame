import torch
import typing
from flame.engine import Engine, Event, BaseContext
from flame.metrics.metric import Metric
from flame.distributed import is_dist_avail_and_init, all_reduce_array


class AverageMetric(object):
    def __init__(self):
        self._is_distributed = is_dist_avail_and_init()
        self.reset()

    def reset(self,) -> None:
        self._n = 0
        self._value = 0.
        self._reset_buf()

    def _reset_buf(self):
        self._n_buf = 0
        self._value_vuf = 0.
        self._is_synced = True

    def sync(self):
        if self._is_synced:
            return
        n = self._n_buf
        value = self._value_vuf
        if self._is_distributed:
            n, value = all_reduce_array(n, value)
        self._n += n
        self._value += value
        self._reset_buf()

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self._value_vuf += value.item()
        elif isinstance(value, (int, float)):
            self._value_vuf += value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self._n_buf += 1
        self._is_synced = False

    def compute(self) -> float:
        self.sync()
        return self._value / (self._n+1e-15)
