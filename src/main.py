# main.py

from config import parse_args, check_arguments
from log import initialize_logging
from data.data_loader import (
    reader,
    build_labels,
    get_amino_acid_frequencies,
    build_vocab,
    SequenceDataset,
)
from data.visualizer import (
    visualize_aa_frequencies,
    visualize_sequence_lengths,
    visualize_family_sizes,
)

import logging
import timeit


def measure_performance(func, *args, **kwargs):
    # Utiliser timeit pour mesurer le temps d'exécution
    execution_time = timeit.timeit(lambda: func(*args, **kwargs), number=1)
    print(f"Execution time: {execution_time:.6f} seconds")


def load_data(args):
    # Charger les données
    train_data, train_targets = reader("train", args.data_path)
    fam2label = build_labels(train_targets)
    amino_acid_counter = get_amino_acid_frequencies(train_data)
    word2id = build_vocab(train_data)
    train_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, args.data_path, "train"
    )
    dev_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, args.data_path, "dev"
    )
    test_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, args.data_path, "test"
    )

    return (
        train_data,
        train_targets,
        fam2label,
        amino_acid_counter,
        word2id,
        train_dataset,
        dev_dataset,
        test_dataset,
    )


def main():
    # Initialisation
    args = parse_args()
    initialize_logging(args)
    check_arguments(args)

    # Chargement des données
    (
        train_data,
        train_targets,
        fam2label,
        amino_acid_counter,
        word2id,
        train_dataset,
        dev_dataset,
        test_dataset,
    ) = load_data(args)

    # Visualisations
    visualize_family_sizes(
        train_targets, args.save_plots, args.display_plots, args.output_path
    )
    visualize_sequence_lengths(
        train_data, args.save_plots, args.display_plots, args.output_path
    )
    visualize_aa_frequencies(
        amino_acid_counter, args.save_plots, args.display_plots, args.output_path
    )

    logging.shutdown()


if __name__ == "__main__":
    main()
