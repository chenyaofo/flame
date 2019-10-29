import os
import sys

__all__ = [
    "__version__",
    "args",
    "output_directory",
    "hocon",
    "debug",
    "logger",
    "engine",
    "handlers",
    "metrics",
    "utils",
]

__version__ = "0.1.0"

from ._utils import get_args, get_hocon_conf, get_output_directory, replace_hocon_item
from ._utils import get_logger
from ._utils import LogExceptionHook
from ._utils import create_code_snapshot

args = get_args(sys.argv)
output_directory = get_output_directory(args.output_directory, args.debug)

debug = args.debug

if output_directory is not None:
    os.makedirs(output_directory, exist_ok=False)
    with open(os.path.join(output_directory, ".flame"), "w") as f:
        f.write(".flame")

logger = get_logger("flame", output_directory, "default.log", debug)

if output_directory is not None:
    logger.info("The default output directory is {}.".format(output_directory))

hocon = get_hocon_conf(args.config)
if hocon is not None:
    logger.info("Initialize flame.hocon from {}.".format(args.config))
replace_hocon_item(hocon, args.replace, logger)

sys.excepthook = LogExceptionHook(logger)

if args.code_snapshot:
    create_code_snapshot("code-snapshot", ["*.py"], os.getcwd(), output_directory, hocon)

from . import engine
from . import handlers
from . import metrics
from . import utils
