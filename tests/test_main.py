import sys
import os

sys.path.append(f"{os.getcwd()}/src")

import tempfile
import pytest

# from unittest.mock import patch
# from config import parse_args
# from main import main
# from log import (
#     LogStatus,
#     configure_logging,
#     log_version_info,
#     log_command_line,
#     initialize_logging,
#     write_logs,
#     print_message,
# )


@pytest.fixture
def temp_directories():
    data_path = tempfile.mkdtemp()
    log_path = tempfile.mkdtemp()
    output_path = tempfile.mkdtemp()

    return data_path, log_path, output_path


@pytest.fixture
def temp_log_directory():
    log_path = tempfile.mkdtemp()
    return log_path


# def test_configure_logging(temp_log_directory):
#     log_file = os.path.join(temp_log_directory, 'test.log')
#     print(log_file)
#     configure_logging(log_file)
#     assert os.path.exists(log_file)

# def test_log_version_info(caplog):
#     log_version_info()
#     assert "Last commit:" in caplog.text
#     assert "Generate Reference version:" in caplog.text

# def test_log_command_line(caplog):
#     log_command_line()
#     assert "Running command:" in caplog.text

# def test_initialize_logging(temp_directories, caplog):
#     data_path, log_path, output_path = temp_directories
#     options = argparse.Namespace(data_path=data_path, log_path=log_path, output_path=output_path)

#     initialize_logging(options)

#     log_file = os.path.join(log_path, 'test.log')
#     assert os.path.exists(log_file)
#     assert "Last commit:" in caplog.text
#     assert "Running command:" in caplog.text

# def test_write_logs(caplog):
#     msg = "Test message"
#     write_logs(msg, LogStatus.INFO, echo=True)
#     assert msg in caplog.text

# def test_print_message(capfd):
#     msg = "Test message"
#     print_message(msg, LogStatus.INFO)
#     captured = capfd.readouterr()
#     assert msg in captured.out

# def test_main(temp_directories):
#     data_path, log_path, output_path = temp_directories

#     with patch('sys.argv', ['main.py', '--data_path', data_path, '--log_path', log_path, '--output_path', output_path]):
#         args = parse_args()
#         main()

#     # Add assertions based on your program's behavior
#     assert os.path.exists(log_path)
#     assert os.path.exists(output_path)
