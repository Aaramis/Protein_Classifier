import logging
import sys
import subprocess
import os
from version import __version__
from typing import Any
from enum import Enum, auto
try:
    from rich import print as rprint
except ImportError:
    def rprint(message):
        print(message)

class LogStatus(Enum):
    INFO = auto()
    DEBUG = auto()
    WARNING = auto()
    CRITICAL = auto()

LOG_LEVELS = {
    LogStatus.INFO: logging.info,
    LogStatus.DEBUG: logging.debug,
    LogStatus.WARNING: logging.warning,
    LogStatus.CRITICAL: logging.critical
}

def configure_logging(log_file: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s:: %(module)s:%(lineno)d:: %(levelname)s:: %(message)s',
        datefmt="%d-%m-%Y %H:%M:%S",
        filemode='w'
    )

def log_version_info() -> None:
    """Log version information."""
    last_commit = os.popen('git rev-parse HEAD').read().strip()
    write_logs(msg=f"Last commit: {last_commit}", status=LogStatus.INFO)
    write_logs(msg=f"Generate Reference version: {__version__}", status=LogStatus.INFO)

def log_command_line() -> None:
    """Log the command line."""
    cmd_line = f"Running command: python {subprocess.list2cmdline(sys.argv)}"
    write_logs(msg=cmd_line, status=LogStatus.INFO)

def initialize_logging(options: Any) -> None:
    log_file = os.path.join(options.log_path, f'{".log"}')
    configure_logging(log_file)
    log_version_info()
    log_command_line()

def write_logs(msg: str, status: LogStatus, echo: bool = False) -> None:
    """Print and log messages for reproducibility."""
    log_function = LOG_LEVELS.get(status, logging.info)
    log_function(msg)
    
    if echo:
        print_message(msg, status)

def print_message(msg: str, status: LogStatus) -> None:
    """Print messages for reproducibility."""
    levels = {
        LogStatus.INFO: ["", ""],
        LogStatus.DEBUG: ["[blue]", "[/blue]"],
        LogStatus.WARNING: ["[dark_orange]", "[/dark_orange]"],
        LogStatus.CRITICAL: ["[red]", "[/red]"]
    }
    rprint(f"o [{status}] {levels[status][0]}{msg}{levels[status][1]}")
