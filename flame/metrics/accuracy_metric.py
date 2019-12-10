import torch
import typing

from .metric import Metric, eps
from flame.engine import Engine, Event, BaseContext
from flame._utils import check
from flame.distributed import is_dist_avail_and_init, all_reduce_array


class Accuracy(object):
    def __init__(self):
        self._is_distributed = is_dist_avail_and_init()
        self.reset()

    def reset(self):
        self._n_correct = 0.0
        self._n_total = 0.0
        self._reset_buffer()

    @property
    def rate(self) -> float:
        self.sync()
        return self._n_correct / (self._n_total+eps)

    @property
    def n_correct(self) -> float:
        self.sync()
        return self._n_correct

    @property
    def n_total(self) -> float:
        self.sync()
        return self._n_total

    def _reset_buffer(self):
        self._n_correct_since_last_sync = 0.0
        self._n_total_since_last_sync = 0.0
        self._is_synced = True

    def update(self,  n_correct, n_total):
        self._n_correct_since_last_sync += n_correct
        self._n_total_since_last_sync += n_total
        self._is_synced = False

    def sync(self):
        if self._is_synced:
            return
        n_correct = self._n_correct_since_last_sync
        n_total = self._n_total_since_last_sync
        if self._is_distributed:
            n_correct, n_total = all_reduce_array(n_correct, n_total, reduction="sum")

        self._n_correct += n_correct
        self._n_total += n_total

        self._reset_buffer()


class AccuracyMetric(object):
    def __init__(self, name: str, topk: typing.Iterable[int] = (1,),):
        self.topk = sorted(list(topk))
        self.reset()

    def reset(self) -> None:
        self.accuracies = [Accuracy() for _ in self.topk]

    def update(self, targets, outputs) -> None:
        maxk = max(self.topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))

        for accuracy, k in zip(self.accuracies, self.topk):
            correct_k = correct[:k].sum().item()
            accuracy.update(correct_k, batch_size)

    def at(self, topk: int) -> Accuracy:
        if topk not in self.topk:
            raise ValueError(f"topk={topk} is not in registered topks={self.topk}")
        accuracy = self.accuracies[self.topk.index(topk)]
        accuracy.sync()
        return accuracy
