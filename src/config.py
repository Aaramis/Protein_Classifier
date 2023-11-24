import argparse
import os
import torch
from log import write_logs, check_directory, LogStatus


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='Protein Classifier Configuration')

    # Command line arguments for directory adnf iles
    main_args = parser.add_argument_group('Directory arguments')
    main_args.add_argument('--data_path',
                           type=str,
                           default=os.path.abspath('./random_split'),
                           help='Path to the data directory')
    main_args.add_argument('--log_path',
                           type=str,
                           default=os.path.abspath('./log'),
                           help='Path to the log directory')
    main_args.add_argument('--output_path', "-o",
                           type=str,
                           default=os.path.abspath('./output'),
                           help='Path to the output directory')
    main_args.add_argument('--model_path',
                           type=str,
                           default=os.path.abspath('./models'),
                           help='Path to the model')
    main_args.add_argument('--model_name',
                           type=str,
                           default='prot_cnn_model.pt',
                           help='Model name')

    # Command line arguments for hyperparameters
    model_args = parser.add_argument_group('Models hyperparameters arguments')

    model_args.add_argument("--train", action="store_true",
                            help="Flag to trigger training.")
    model_args.add_argument("--test", action="store_true",
                            help="Flag to trigger testing.")
    model_args.add_argument('--seq_max_len',
                            type=int, default=120,
                            help='Maximum length of sequences')
    model_args.add_argument('--batch_size',
                            type=int, default=1,
                            help='Batch size for training and evaluation')
    model_args.add_argument('--num_workers',
                            type=int, default=1,
                            help='Number of data loader workers')
    model_args.add_argument('--num_classes',
                            type=int, default=16652,
                            help='Number of classes (family labels)')
    model_args.add_argument('--lr',
                            type=float, default=1e-2,
                            help='Learning rate')
    model_args.add_argument('--momentum',
                            type=float, default=0.9,
                            help='SGD momentum')
    model_args.add_argument('--weight_decay',
                            type=float, default=1e-2,
                            help='Weight decay for regularization')
    model_args.add_argument('--gpus',
                            type=int, default=1,
                            help='Number of GPUs to use for training')
    model_args.add_argument('--epochs',
                            type=int, default=25,
                            help='Number of training epochs')

    # Command line arguments for plots
    plots_args = parser.add_argument_group('Plots arguments')
    plots_args.add_argument('--display_plots',
                            type=int, default=0,
                            help='Display plots')
    plots_args.add_argument('--save_plots',
                            type=int, default=0,
                            help='Save plots in output folder')

    return parser.parse_args()


def check_positive_integer(value: int, name: str) -> None:
    """Check if the value is a positive integer."""
    if value <= 0 or not isinstance(value, int):
        write_logs(f"Invalid {name}: {value}. It should be a positive integer.",
                   LogStatus.CRITICAL, True)


def check_float_range(value: float, name: str, min_val: float = 0, max_val: float = 1) -> None:
    """Check if the float value is within the specified range."""
    if value < min_val or value > max_val:
        write_logs(f"Invalid {name}: {value}. It should be between {min_val} and {max_val}.",
                   LogStatus.CRITICAL, True)


def check_arguments(options: argparse.Namespace) -> None:
    """Validate the arguments"""
    check_directory(options.data_path, 'data directory')
    check_directory(options.log_path, 'log directory')
    check_directory(options.output_path, 'output directory')
    check_directory(options.model_path, 'model directory')

    # Check if the specified GPUs are available
    if options.gpus > 0 and not torch.cuda.is_available():
        available_gpu_count = torch.cuda.device_count()
        if not available_gpu_count:
            write_logs("GPU not available, training will be performed on CPU.", LogStatus.WARNING, True)
        elif options.gpus != available_gpu_count:
            write_logs("GPU are not optimized", LogStatus.WARNING, True)

    check_positive_integer(options.batch_size, 'batch size')
    check_positive_integer(options.num_workers, 'num workers')
    check_positive_integer(options.num_classes, 'number of classes')
    check_positive_integer(options.seq_max_len, 'number of classes')
    check_float_range(options.lr, 'learning rate', 0, 1)
    check_float_range(options.momentum, 'momentum', 0, 1)
    check_positive_integer(options.epochs, 'number of epochs')
    check_float_range(options.display_plots, 'display plots', 0, 1)
    check_float_range(options.save_plots, 'save plots', 0, 1)

    # Check if log_path and output_path are writable
    try:
        for path in [options.log_path, options.output_path]:
            if path:
                test_file = os.path.join(path, 'test_write_permission.txt')
                with open(test_file, 'w') as test_fp:
                    test_fp.write("Testing write permission.")
                os.remove(test_file)
    except IOError:
        write_logs("No write permission to the specified directory. EXIT",
                   LogStatus.CRITICAL, True)
