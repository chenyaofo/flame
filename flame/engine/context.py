from functools import partial
from dataclasses import dataclass, field

import torch

from flame.engine.phase import Phase

context = partial(dataclass, init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=False)


@context
class BaseContext:
    epoch: int = field(default=0)
    max_epoch: int = field(default=None)
    phase: Phase = field(default=None)
    iteration: int = field(default=0)
    max_iteration: int = field(default=None)
    net: torch.nn.Module = field(default=None)
    device: torch.device = field(default=None)
    entrypoints: dict = field(default_factory=dict)

    inputs: object = field(default=None)

    def set_infinity_epoch(self):
        self.max_epoch = float("inf")

    def set_infinity_iteration(self):
        self.max_iteration = float("inf")

    def is_in_phase(self, phase):
        return phase == self.phase

    def is_register_phase(self, phase):
        for _, item in self.__dict__.items():
            if isinstance(item, Phase):
                if item == phase:
                    return True
        return False

    @property
    def is_first_epoch(self):
        return self.epoch == 1

    @property
    def is_last_epoch(self):
        return self.epoch == self.max_epoch

    @property
    def is_first_iteration(self):
        return self.iteration == 1

    @property
    def is_last_iteration(self):
        return self.iteration == self.max_iteration

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass
