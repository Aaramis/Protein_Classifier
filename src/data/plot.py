import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace
from pathlib import Path
from src.log import write_logs, LogStatus


def visualize_family_sizes(data: pd.Series, args: Namespace) -> None:

    write_logs(f"Ploting {args.split} family sizes", LogStatus.INFO, True)
    sorted_targets = data.groupby(data).size().sort_values(ascending=False)

    plt.figure()

    sns.histplot(sorted_targets.values, kde=True, log_scale=True)
    plt.title(f"Distribution of family sizes for the {args.split} split")
    plt.xlabel("Family size (log scale)")
    plt.ylabel("# Families")

    if args.save_plots:
        save_path = Path(args.output_path) / "plot" / f"{args.split}_family_sizes.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if args.display_plots:
        plt.show()


def visualize_sequence_lengths(train_data: pd.Series, args: Namespace) -> None:

    write_logs(f"Ploting {args.split} sequence lengths", LogStatus.INFO, True)
    sequence_lengths = train_data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()

    plt.figure()

    sns.histplot(sequence_lengths.values, kde=True, log_scale=True, bins=60)
    plt.axvline(mean, color="r", linestyle="-", label=f"Mean = {mean:.1f}")
    plt.axvline(median, color="g", linestyle="-", label=f"Median = {median:.1f}")
    plt.title(f"Distribution of sequence lengths for {args.split} split")
    plt.xlabel("Sequence' length (log scale)")
    plt.ylabel("# Sequences")
    plt.legend(loc="best")

    if args.save_plots:
        save_path = Path(args.output_path) / "plot" / f"{args.split}_sequence_lengths.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if args.display_plots:
        plt.show()


def visualize_aa_frequencies(amino_acid_counter: pd.DataFrame, args: Namespace) -> None:

    write_logs(f"Ploting {args.split} aa frequencies", LogStatus.INFO, True)
    f, ax = plt.subplots(figsize=(8, 5))

    pal = sns.husl_palette(len(amino_acid_counter))
    sns.barplot(
        x="AA",
        y="Frequency",
        data=amino_acid_counter.sort_values(by=["Frequency"], ascending=False),
        ax=ax,
        palette=pal,
        hue="AA",
    )

    plt.title("Distribution of AAs' frequencies in the 'train' split")
    plt.xlabel("Amino acid codes")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")

    if args.save_plots:
        save_path = Path(args.output_path) / "plot" / f"{args.split}_aa_frequencies.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if args.display_plots:
        plt.show()
