import os
import logging
import timeit
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from config import parse_args, check_arguments
from log import initialize_logging, write_logs, LogStatus
from data.data_loader import (
    reader,
    build_labels,
    get_amino_acid_frequencies,
    build_vocab,
    SequenceDataset,
    create_data_loaders,
)
from data.plot import (
    visualize_aa_frequencies,
    visualize_sequence_lengths,
    visualize_family_sizes,
)
from model.protein_cnn import ProtCNN
from model.utils import (
    load_model,
    eval_model,
    predict_one_sequence,
    predict_csv,
)


def measure_performance(func, *args, **kwargs):
    # Use timeit to measure the execution time
    execution_time = timeit.timeit(lambda: func(*args, **kwargs), number=1)
    print(f"Execution time: {execution_time:.6f} seconds")


def load_data(args):
    train_data, train_targets = reader("train", args.data_path)
    fam2label = build_labels(train_targets)

    if args.num_classes != len(fam2label):
        args.num_classes == len(fam2label)
        msg = f"Num_classes doesn't match numbers of labels. New num_classes {len(fam2label)}"
        write_logs(msg, LogStatus.WARNING, True)

    amino_acid_counter = get_amino_acid_frequencies(train_data)
    word2id = build_vocab(train_data)

    train_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_path, "train")
    dev_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_path, "dev")
    test_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_path, "test")

    dataloaders = create_data_loaders(train_dataset, dev_dataset, test_dataset, args.batch_size, args.num_workers)

    return train_data, fam2label, word2id, amino_acid_counter, dataloaders


def visualize_plots(data, aa_counter, args):
    visualize_family_sizes(data, args.save_plots, args.display_plots, args.output_path)
    visualize_sequence_lengths(data, args.save_plots, args.display_plots, args.output_path)
    visualize_aa_frequencies(aa_counter, args.save_plots, args.display_plots, args.output_path)


def main():
    # Initialization
    args = parse_args()
    initialize_logging(args)
    check_arguments(args)

    # Load data
    train_data, fam2label, word2id, amino_acid_counter, dataloaders = load_data(args)

    # Visualizations
    if args.save_plots or args.display_plots:
        visualize_plots(train_data, amino_acid_counter, args)

    # Train
    if args.train:
        prot_cnn = ProtCNN(args.num_classes)
        tb_logger = TensorBoardLogger("lightning_logs/", name="my_experiment")
        pl.seed_everything(0)
        trainer = pl.Trainer(accelerator="auto", max_epochs=args.epochs, logger=tb_logger)
        trainer.fit(prot_cnn, dataloaders['train'], dataloaders['dev'])
        torch.save(prot_cnn.state_dict(), os.path.join(args.model_path, args.model_name))

    # Evaluation
    if args.eval:
        model = load_model(args.model_path, args.model_name)
        eval_model(model, dataloaders['test'])

    # Predict 1 sequence
    if args.predict and args.sequence:
        model = load_model(args.model_path, args.model_name)
        predict_one_sequence(args, model, args.sequence, word2id, fam2label, True)

    # Predict multiple sequences
    if args.predict and args.csv:
        model = load_model(args.model_path, args.model_name)
        predict_csv(args, model, word2id, fam2label)

    logging.shutdown()


if __name__ == "__main__":
    main()
