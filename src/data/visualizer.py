from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualize_family_sizes(train_targets: pd.Series, save: bool, display: bool, output_dir: Union[str, Path]) -> None:

    sorted_targets = train_targets.groupby(train_targets).size().sort_values(ascending=False)

    plt.figure()

    sns.histplot(sorted_targets.values, kde=True, log_scale=True)
    plt.title("Distribution of family sizes for the 'train' split")
    plt.xlabel("Family size (log scale)")
    plt.ylabel("# Families")

    if save:
        save_path = Path(output_dir) / "plot" / "family_sizes.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if display:
        plt.show()


def visualize_sequence_lengths(
    train_data: pd.Series,
    save: bool = True,
    display: bool = True,
    output_dir: Union[str, Path] = None,
) -> None:
    sequence_lengths = train_data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()

    plt.figure()

    sns.histplot(sequence_lengths.values, kde=True, log_scale=True, bins=60)
    plt.axvline(mean, color="r", linestyle="-", label=f"Mean = {mean:.1f}")
    plt.axvline(median, color="g", linestyle="-", label=f"Median = {median:.1f}")
    plt.title("Distribution of sequence lengths")
    plt.xlabel("Sequence' length (log scale)")
    plt.ylabel("# Sequences")
    plt.legend(loc="best")

    if save:
        save_path = Path(output_dir) / "plot" / "sequence_lengths.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    if display:
        plt.show()


def visualize_aa_frequencies(
    amino_acid_counter: pd.DataFrame,
    save: bool = True,
    display: bool = True,
    output_dir: Union[str, Path] = None,
) -> None:
    f, ax = plt.subplots(figsize=(8, 5))

    pal = sns.husl_palette(len(amino_acid_counter))
    sns.barplot(
        x="AA",
        y="Frequency",
        data=amino_acid_counter.sort_values(by=["Frequency"], ascending=False),
        ax=ax,
        palette=pal,
        hue="AA",
        legend=False,
    )

    plt.title("Distribution of AAs' frequencies in the 'train' split")
    plt.xlabel("Amino acid codes")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")

    if save:
        save_path = Path(output_dir) / "plot" / "aa_frequencies.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if display:
        plt.show()
