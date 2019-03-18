from enum import Enum


class Scope(Enum):
    GLOBAL = "global"
    EPOCH = "epoch"
    PHASE = "phase"
    ITERATION = "iteration"
