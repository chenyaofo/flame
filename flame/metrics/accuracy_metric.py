import torch
import typing

from .metric import Metric
from flame.engine import Engine, Event, BaseContext
from flame._utils import check


class Accuracy(object):
    def __init__(self, rate, n_correct, n_total):
        self.rate = rate
        self.n_correct = n_correct
        self.n_total = n_total

    def __str__(self):
        return f"Accuracy={self.rate * 100:.4f}%({self.n_correct}/{self.n_total})"


class AccuracyMetric(Metric):
    def __init__(self, name: str, topk: typing.Iterable[int] = (1,), grad_enabled: bool = False,
                 context_map: typing.Callable[[BaseContext], typing.Any] = None,
                 reset_event: Event = Event.PHASE_STARTED, update_event: Event = Event.ITER_COMPLETED):
        super(AccuracyMetric, self).__init__(name, grad_enabled, context_map, reset_event, update_event, True)
        topk = check(topk, "topk", None, lambda topk: len(topk) == len(set(topk)),
                     "The parameter topk must not contain repeating element")
        self.topk = sorted(list(topk))

        self.accuracies = None

        self.reset()

    def reset(self, *args, **kwargs) -> None:
        self.accuracies = [Accuracy(rate=0.0, n_correct=0, n_total=0) for _ in self.topk]

    def update(self, targets, outputs) -> None:
        with torch.set_grad_enabled(self.grad_enabled):
            maxk = max(self.topk)
            batch_size = targets.size(0)

            _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1))

            for accuracy, k in zip(self.accuracies, self.topk):
                accuracy.n_total += batch_size

                correct_k = correct[:k].sum().item()
                accuracy.n_correct += correct_k
                accuracy.rate = accuracy.n_correct / accuracy.n_total

    @property
    def value(self):
        return self

    def __getitem__(self, i) -> Accuracy:
        return self.accuracies[self.topk.index(i)]
