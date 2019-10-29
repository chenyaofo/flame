import os
import sys
import time
import typing
import logging
import argparse
import pyhocon
import glob
import zipfile


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

    logger.propagate = False
    return logger


def get_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train a network")
    parser.add_argument("-c", "--config", type=str, nargs='?',
                        help="the path to config file.", default=None, required=False)
    parser.add_argument("-o", "--output_directory", type=str, nargs='?',
                        help="the path to store experiment files.",
                        default="!default", required=False)
    parser.add_argument("-d", "--debug", action="store_true",
                        help="the flag for debug mode.", default=False)
    parser.add_argument("--code-snapshot", action="store_true",
                        help="the flag to create code snapshot.", default=False)
    parser.add_argument("--replace", type=str,
                        help="replace config item in hocon config(--config)", default=None)

    args, _ = parser.parse_known_args(argv)
    return args


def get_hocon_conf(commandline_arg: str) -> pyhocon.config_tree.ConfigTree:
    if commandline_arg is None:
        conf = None
    else:
        conf = pyhocon.ConfigFactory.parse_file(commandline_arg)
    return conf


def get_output_directory(commandline_arg: str, debug: bool) -> str:
    if commandline_arg is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if debug:
            output_directory = os.path.join("output", "{}_debug".format(timestamp))
        else:
            output_directory = os.path.join("output", timestamp)
    else:
        if commandline_arg == "!default":
            output_directory = None
        else:
            output_directory = commandline_arg

    return output_directory


def replace_hocon_item(hocon: pyhocon.ConfigTree, commandline_arg: str, logger=None):
    if commandline_arg is None:
        return
    for item in commandline_arg.split("&"):
        key, value = item.split("=")
        try:
            old_value = hocon[key]
            if isinstance(old_value, list):
                hocon.put(key, eval(value))
            else:
                hocon.put(key, value)
            logger.info("Replace hocon config item, {} = {} -> {}".format(key, old_value, value))
        except KeyError:
            raise KeyError("The key({}) is not defined in hocon config.".format(key))


def create_code_snapshot(name: str,
                         include_suffix: typing.List[str],
                         source_directory: str,
                         store_directory: str,
                         hocon: pyhocon.ConfigTree) -> None:
    if store_directory is None:
        return
    with zipfile.ZipFile(os.path.join(store_directory, "{}.zip".format(name)), "w") as f:
        for suffix in include_suffix:
            for file in glob.glob(os.path.join(source_directory, "**", "*{}".format(suffix)), recursive=True):
                f.write(file, os.path.join(name, file))
        if hocon is not None:
            f.writestr(
                os.path.join(name, "config", "default.hocon"),
                pyhocon.HOCONConverter.to_hocon(hocon, indent=4)
            )
