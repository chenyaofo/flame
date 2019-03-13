from enum import Enum


class Event(Enum):
    ENGINE_STARTED = 'engine_stated'
    ENGINE_COMPLETED = 'engine_complete'
    EPOCH_STARTED = 'epoch_started'
    EPOCH_COMPLETED = 'epoch_completed'
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_complelted"
    ITER_STARTED = 'iter_started'
    ITER_COMPLETED = 'iter_completed'
    EXCEPTION_RAISED = "exception_raised"
