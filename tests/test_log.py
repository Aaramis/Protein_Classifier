import sys
import os

sys.path.append(f"{os.getcwd()}/src")

import logging
from log import (
    check_directory,
    LogStatus,
    write_logs,
    configure_logging,
    log_version_info,
    log_command_line,
)
from config import parse_args

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def test_initialize_logging(caplog):
    """ unit test for all functions in log.py file """

    stream_handler.stream = sys.stdout
    options = parse_args()

    # Check if the log file is created
    check_directory(options.log_path, "log directory")
    assert os.path.exists(options.log_path)

    # Check configure Logging
    log_file = os.path.join(options.log_path, f'{".log"}')
    configure_logging(log_file)
    assert logging.root.level == logging.DEBUG

    # Check log_version
    log_version_info()
    assert len(caplog.text.split("\n")) == 3

    # Check log_command_line
    log_command_line()
    assert len(caplog.text.split("\n")) == 4

    # Check write_logs
    write_logs("1", LogStatus.WARNING, False)
    assert len(caplog.text.split("\n")) == 5
    write_logs("1", LogStatus.CRITICAL, False)
    assert len(caplog.text.split("\n")) == 6
