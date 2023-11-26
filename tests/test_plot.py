import sys
import os

sys.path.append(f"{os.getcwd()}")

from src.data.plot import (
    visualize_family_sizes,
    visualize_sequence_lengths,
    visualize_aa_frequencies,
)
from src.data.data_loader import (
    reader,
    get_amino_acid_frequencies,
)

from typing import Tuple

from pathlib import Path
import pytest
from PIL import Image
import pandas as pd


fake_data_path = Path.cwd() / "tests" / "data"


class Args(object):

    def __init__(self, output_path: str) -> None:
        self.split = "train"
        self.save_plots = 1
        self.display_plots = 0
        self.output_path = output_path


@pytest.fixture
def mock_data() -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    train_targets = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'C'])
    train_data = pd.Series(['AAC', 'BB', 'CC', 'AAD', 'BB', 'AAA', 'CCD'])
    amino_acid_counter = pd.DataFrame({'AA': ['A', 'B', 'C', 'D'], 'Frequency': [4, 3, 2, 1]})
    return train_targets, train_data, amino_acid_counter


@pytest.fixture
def fake_data() -> Tuple[pd.Series, pd.Series]:
    return reader("train", fake_data_path)


def test_visualize_family_sizes(mock_data: Tuple[pd.Series, pd.Series, pd.DataFrame], tmp_path: Path) -> None:
    train_targets, _, _ = mock_data
    save_path = tmp_path / "plot" / "train_family_sizes.png"
    args = Args(tmp_path)
    visualize_family_sizes(train_targets, args)

    assert save_path.is_file()

    img = Image.open(save_path)
    assert img.mode == 'RGBA'


def test_visualize_sequence_lengths(mock_data: Tuple[pd.Series, pd.Series, pd.DataFrame], tmp_path: Path) -> None:
    _, train_data, _ = mock_data
    save_path = tmp_path / "plot" / "train_sequence_lengths.png"
    args = Args(tmp_path)
    visualize_sequence_lengths(train_data, args)

    assert save_path.is_file()

    img = Image.open(save_path)
    assert img.mode == 'RGBA'


def test_visualize_aa_frequencies(tmp_path: Path, fake_data: Tuple[pd.Series, pd.Series]) -> None:
    sequence, _ = fake_data
    amino_acid_counter = get_amino_acid_frequencies(sequence)
    save_path = tmp_path / "plot" / "train_aa_frequencies.png"
    args = Args(tmp_path)
    visualize_aa_frequencies(amino_acid_counter, args)

    assert save_path.is_file()

    img = Image.open(save_path)
    assert img.size == (800, 500)
    assert img.mode == 'RGBA'
