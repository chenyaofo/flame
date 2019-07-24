import os
import sys
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


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger
