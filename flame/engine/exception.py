class EngineException(BaseException):
    pass


class EngineStopException(EngineException):
    pass


class EpochStopException(EngineException):
    pass


class PhaseStopException(EngineException):
    pass


class IterationStopException(EngineException):
    pass
