# config.py

import argparse
import os
import torch
from log import write_logs, LogStatus

def parse_args():
    
    parser = argparse.ArgumentParser(description='Protein Classifier Configuration')

    # Command line arguments for directory
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

    # Command line arguments for hyperparameters
    model_args = parser.add_argument_group('Models hyperparameters arguments')

    model_args.add_argument('--seq_max_len'
                            , type=int, default=120
                            , help='Maximum length of sequences')
    model_args.add_argument('--batch_size'
                            , type=int, default=1
                            , help='Batch size for training and evaluation')
    model_args.add_argument('--num_workers'
                            , type=int, default=0
                            , help='Number of data loader workers')
    model_args.add_argument('--num_classes'
                            , type=int, default=16652
                            , help='Number of classes (family labels)')
    model_args.add_argument('--lr'
                            , type=float, default=1e-2
                            , help='Learning rate')
    model_args.add_argument('--momentum'
                            , type=float, default=0.9
                            , help='SGD momentum')
    model_args.add_argument('--weight_decay'
                            , type=float, default=1e-2
                            , help='Weight decay for regularization')
    model_args.add_argument('--gpus'
                            , type=int, default=0
                            , help='Number of GPUs to use for training')
    model_args.add_argument('--epochs'
                            , type=int, default=25
                            , help='Number of training epochs')

    
    return parser.parse_args()

def check_arguments(options):
    """Validate the arguments"""

    if options.data_path and not os.path.isfile(options.data_path):
        write_logs(f"'{options.data_path}' does not exist. EXIT"
                   , LogStatus.CRITICAL
                   , True)

    if options.log_path:
        os.makedirs(options.log_path, exist_ok=True)

    # # Check GPU availability
    # if options.gpus > 0 and not torch.cuda.is_available():
    #     msg = {
    #         "msg": "GPU not available, training will be performed on CPU.",
    #         "status": "warning",
    #         "echo": True
    #     }
    #     logs.append(msg)