import logging
from argparse import Namespace
from pandas import DataFrame
from src.config import parse_args, check_arguments
from src.log import initialize_logging, write_logs, LogStatus
from src.data.data_loader import (
    reader,
    reader_T5,
    build_labels,
    get_amino_acid_frequencies,
    build_vocab,
    SequenceDataset,
)
from src.data.plot import (
    visualize_aa_frequencies,
    visualize_sequence_lengths,
    visualize_family_sizes,
)
from src.model.protein_cnn import ProtCNN
from src.model.utils_cnn import load_model
from src.model.utils_t5 import (
    save_model_T5,
    evaluate_t5,
    predict_one_sequence_t5,
    predict_csv_t5,
)
from src.model.train_cnn import ModelManager
from src.model.train_t5 import train_per_residue, load_model_T5


def load_data(args: Namespace):
    train_data, train_targets = reader("train", args.data_path)
    fam2label = build_labels(train_targets)

    if args.num_classes != len(fam2label):
        args.num_classes == len(fam2label)
        msg = f"Num_classes doesn't match numbers of labels. New num_classes {len(fam2label)}"
        write_logs(msg, LogStatus.WARNING, True)

    amino_acid_counter = get_amino_acid_frequencies(train_data)
    word2id = build_vocab(train_data)

    return fam2label, word2id, amino_acid_counter


def visualize_plots(aa_counter: DataFrame, args: Namespace):

    data, _ = reader(args.split, args.data_path)

    visualize_family_sizes(data, args)
    visualize_sequence_lengths(data, args)
    visualize_aa_frequencies(aa_counter, args)


def case_cnn(args: Namespace, word2id, fam2label):
    prot_cnn = ProtCNN(args.momentum, args.weight_decay, args.lr, args.num_classes)
    model_manager = ModelManager(prot_cnn, word2id, fam2label, args)

    # Train
    if args.train:
        train_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_path, "train")
        train_dataloader = train_dataset.create_dataloader(train_dataset, args.batch_size, args.num_workers)

        dev_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_path, "dev")
        dev_dataloader = dev_dataset.create_dataloader(dev_dataset, args.batch_size, args.num_workers)

        model_manager.train(train_dataloader, dev_dataloader)

    # Evaluation
    if args.eval:
        model = load_model(args.model_path, args.model_name)
        test_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, args.data_path, "test")
        test_dataloader = test_dataset.create_dataloader(test_dataset, args.batch_size, args.num_workers)
        model_manager.evaluate(model, test_dataloader)

    # Predict 1 sequence
    if args.predict and args.sequence:
        model = load_model(args.model_path, args.model_name)
        model_manager.predict_one_sequence(model, args.sequence, True)

    # Predict multiple sequences
    if args.predict and args.csv:
        model = load_model(args.model_path, args.model_name)
        model_manager.predict_csv(model)


def case_T5(args: Namespace, fam2label):
    all_data = reader_T5(args.data_path, fam2label)
    model_file = f"{args.model_path}/{args.model_name}"

    # Train
    if args.train:
        # Get train and validation data
        my_train = all_data[all_data.dataset == "train"].reset_index(drop=True)
        my_valid = all_data[all_data.dataset == "dev"].reset_index(drop=True)

        tokenizer, model, history = train_per_residue(
            my_train,
            my_valid,
            num_labels=args.num_classes,
            batch=args.batch_size,
            accum=1,
            epochs=args.epochs,
            seed=42,
            gpu=args.gpus,
            pretrained_model=args.pretrained_model,
            dropout=args.dropout,
        )

        save_model_T5(model, model_file)

    if args.eval:
        tokenizer, model = load_model_T5(model_file, args.num_classes, args.pretrained_model, args.dropout)
        my_test = all_data[all_data.dataset == "test"].reset_index(drop=True)
        evaluate_t5(model, tokenizer, my_test)

    # Predict 1 sequence
    if args.predict and args.sequence:
        tokenizer, model = load_model_T5(model_file, args.num_classes, args.pretrained_model, args.dropout)
        predict_one_sequence_t5(model, tokenizer, args.sequence, fam2label, True)

    # Predict multiple sequences
    if args.predict and args.csv:
        tokenizer, model = load_model_T5(model_file, args.num_classes, args.pretrained_model, args.dropout)
        predict_csv_t5(model, tokenizer, args.csv, args.output_path, fam2label)


def main():

    # Initialization
    args = parse_args()
    initialize_logging(args)
    check_arguments(args)

    # Load data
    fam2label, word2id, amino_acid_counter = load_data(args)

    # Visualizations
    if args.save_plots or args.display_plots:
        visualize_plots(amino_acid_counter, args)

    # CNN
    if args.model_type == "CNN":
        case_cnn(args, word2id, fam2label)

    # T5
    if args.model_type == 'T5':
        case_T5(args, fam2label)

    logging.shutdown()


if __name__ == "__main__":
    main()
