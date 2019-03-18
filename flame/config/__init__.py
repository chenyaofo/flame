import os
import time
import argparse
import pyhocon

__all__ = [
    "get_args",
    "get_hocon_conf",
    "get_output_directory"
]


def get_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train a network")
    parser.add_argument("-c", "--config", type=str, nargs='?', help="the path to config file.", default=None,
                        required=False)
    parser.add_argument("-o", "--output_directory", type=str, nargs='?',
                        help="the path to store experiment files.", default="!default", required=False)
    parser.add_argument("-d", "--debug", action="store_true", help="the flag for debug mode.", default=False)
    parser.add_argument("--replace", type=str, help="replace config item in hocon config(--config)", default=None)

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


def replace_hocon_item(hocon: pyhocon.ConfigTree, commandline_arg: str):
    if commandline_arg is None:
        return
    for item in commandline_arg.split("&"):
        key, value =item.split("=")
        try:
            old_value = hocon[key]
            if isinstance(old_value, list):
                hocon.put(key, eval(value))
            else:
                hocon.put(key, value)
        except KeyError:
            raise KeyError("The key({}) is not defined in hocon config.".format(key))
