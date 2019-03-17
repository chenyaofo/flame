import sys
import typing
from IPython.core import ultratb


class DebugExceptionHook(object):
    def __init__(self, logger):
        self.instance = None
        self.logger = logger

    def __call__(self, exc_type, exc_value, traceback):
        frame = sys._getframe()
        self.logger.debug("In line {}, {}, {}".format(frame.f_lineno,
                                                      frame.f_code.co_name,
                                                      frame.f_code.co_filename))
        self.logger.debug("Encounter an uncaught exception {}: {}.".format(exc_type.__name__, exc_value))
        if self.instance is None:
            self.logger.debug("Enter debug mode with ipython, please enter 'help' for help.")
            self.instance = ultratb.FormattedTB(mode='Plain',
                                                color_scheme='Linux', call_pdb=1)
        return self.instance(exc_type, exc_value, traceback)


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
