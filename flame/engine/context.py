from functools import partial
from dataclasses import dataclass, is_dataclass, fields, asdict, Field, MISSING

import torch

from flame.engine.phase import Phase
from flame.engine.scope import Scope
from .utils import Timer

context = partial(dataclass, init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=False)


def context_field(*, default=MISSING, default_factory=MISSING,
                  init: bool = True, repr: bool = True, hash: bool = None, compare: bool = True,
                  scope: Scope = Scope.GLOBAL, requierd_serialization: bool = False, metadata: dict = None):
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError('cannot specify both default and default_factory')
    if metadata is not None:
        return Field(default, default_factory, init, repr, hash, compare,
                     metadata.update({"requierd_serialization": requierd_serialization,
                                      "scope": scope.name}))
    else:
        return Field(default, default_factory, init, repr, hash, compare,
                     metadata={"requierd_serialization": requierd_serialization,
                               "scope": scope.name})


@context
class BaseContext:
    epoch: int = context_field(default=0, requierd_serialization=True)
    max_epoch: int = context_field(default=None, requierd_serialization=True)
    phase: Phase = context_field(default=None, requierd_serialization=True)
    iteration: int = context_field(default=0, requierd_serialization=True)
    max_iteration: int = context_field(default=None, requierd_serialization=True)
    device: torch.device = context_field(default=None, requierd_serialization=False)
    entrypoints: dict = context_field(default_factory=dict, requierd_serialization=True)

    timer: Timer = context_field(default=None, requierd_serialization=False)

    inputs: object = context_field(default=None)

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

    def state_dict(self) -> dict:
        context_fields = fields(self)
        context_dict = asdict(self)

        state_dict = dict()

        for field in context_fields:
            name = field.name
            requierd_serialization = field.metadata["requierd_serialization"]
            if requierd_serialization:
                if hasattr(context_dict[name], "state_dict"):
                    state_dict[name] = context_dict[name].state_dict()
                else:
                    state_dict[name] = context_dict[name]
        return state_dict

    def load_state_dict(self, state_dict: dict):
        context_fields = fields(self)
        context_dict = self.__dict__

        for field in context_fields:
            name = field.name
            requierd_serialization = field.metadata["requierd_serialization"]
            if requierd_serialization:
                if hasattr(context_dict[name], "load_state_dict"):
                    context_dict[name].load_state_dict(state_dict[name])
                else:
                    context_dict[name] = state_dict[name]

    def _clean_variable(self,scope_name):
        context_fields = fields(self)
        context_dict = self.__dict__

        for field in context_fields:
            name = field.name
            scope = field.metadata["scope"]
            if scope == scope_name:
                context_dict[name] = None

    def clean_epoch_variable(self):
        self._clean_variable(Scope.EPOCH.name)

    def clean_phase_variable(self):
        self._clean_variable(Scope.PHASE.name)

    def clean_iteration_variable(self):
        self._clean_variable(Scope.ITERATION.name)
