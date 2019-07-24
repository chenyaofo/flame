import typing
import logging


class LogExceptionHook(object):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(self, exc_type, exc_value, traceback):
        self.logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, traceback))


T = typing.TypeVar("T")


def check(value: T, name: str, declared_type: typing.Any = None,
          condition: typing.Optional[typing.Callable[[T], bool]] = None, message: typing.Optional[str] = None) -> T:
    if declared_type is not None:
        if not isinstance(value, declared_type):
            raise TypeError("The parameter {} should be {}, but got {}."
                            .format(name, declared_type, type(value)))
    if condition is not None:
        if not condition(value):
            if message is None:
                raise ValueError()
            else:
                raise ValueError("{}, but got {}.".format(message, value))
    return value
