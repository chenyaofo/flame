import os
import time
import argparse
import pyhocon

__all__ = [
    "get_args",
    "get_hocon_conf",
    "get_working_directory"
]


def get_args(argv):
    parser = argparse.ArgumentParser(description="train a network")
    parser.add_argument("-c", "--config", type=str, nargs='?', help="the path to config file.", default=None,
                        required=False)
    parser.add_argument("-e", "--experiment", type=str, nargs='?', help="the path to store experiment files.",
                        default="!default",
                        required=False)
    parser.add_argument("-d", "--debug", action="store_true", help="the flag for debug mode.", default=False)

    return parser.parse_known_args(argv)


def get_hocon_conf(commandline_arg):
    if commandline_arg is None:
        conf = None
    else:
        conf = pyhocon.ConfigFactory.parse_file(commandline_arg)
    return conf


def get_working_directory(commandline_arg, debug):
    if commandline_arg is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if debug:
            working_directory = os.path.join("experiments", "{}_debug".format(timestamp))
        else:
            working_directory = os.path.join("experiments", timestamp)
    else:
        if commandline_arg == "!default":
            working_directory = None
        else:
            working_directory = commandline_arg

    return working_directory
