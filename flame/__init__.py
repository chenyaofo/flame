import os
import sys

__version__ = "0.1.0-alpha0"

from .config import get_args, get_hocon_conf, get_working_directory
from .logging import get_logger
from ._utils import DebugExceptionHook

args, _ = get_args(sys.argv)
working_directory = get_working_directory(args.experiment, args.debug)
conf = get_hocon_conf(args.config)

debug = args.debug

if working_directory is not None:
    os.makedirs(working_directory, exist_ok=False)

logger = get_logger("flame", working_directory, "default.log", debug)

if debug:
    sys.excepthook = DebugExceptionHook(logger)

from flame import engine
from flame import handlers
from flame import metrics
from flame import utils
