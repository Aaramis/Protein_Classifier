import sys
import os

sys.path.append(f"{os.getcwd()}")

import pytest
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from src.data.data_loader import (
    reader,
    build_labels,
    get_amino_acid_frequencies,
    build_vocab,
    SequenceDataset,
    create_data_loaders,
)

fake_data_path = f"{os.getcwd()}/tests/data"


@pytest.fixture
def fake_data() -> Tuple[pd.Series, pd.Series]:
    return reader("train", fake_data_path)


def test_reader(fake_data: Tuple[pd.Series, pd.Series]) -> None:
    sequence, targets = fake_data

    assert isinstance(sequence, pd.Series)
    assert isinstance(targets, pd.Series)

    assert len(sequence) == 13390
    assert len(targets) == 13390


def test_build_labels(fake_data: Tuple[pd.Series, pd.Series]) -> None:
    _, targets = fake_data
    fam2label = build_labels(targets)
    # Checks the number of labels, including '<unk>'.
    assert len(fam2label) == len(targets.unique()) + 1
    assert isinstance(fam2label, Dict)


def test_get_amino_acid_frequencies(fake_data: Tuple[pd.Series, pd.Series]) -> None:
    sequence, _ = fake_data
    amino_acid_counter = get_amino_acid_frequencies(sequence)
    assert len(amino_acid_counter) == 22
    assert isinstance(amino_acid_counter, pd.DataFrame)
    assert amino_acid_counter.shape[1] == 2
    assert amino_acid_counter.columns.tolist() == ["AA", "Frequency"]


def test_build_vocab(fake_data: Tuple[pd.Series, pd.Series]) -> None:
    sequence, _ = fake_data
    word2id = build_vocab(sequence)
    assert len(word2id) == 22
    assert isinstance(word2id, Dict)


def test_sequence_dataset(fake_data: Tuple[pd.Series, pd.Series]) -> None:
    sequence, targets = fake_data
    word2id = build_vocab(sequence)
    fam2label = build_labels(targets)

    max_len = 10

    dataset = SequenceDataset(word2id, fam2label, max_len, fake_data_path, "train")

    assert isinstance(dataset, SequenceDataset)
    assert len(dataset) == 13390

    sample = dataset[0]
    assert "sequence" in sample
    assert "target" in sample
    assert sample["sequence"].shape == (len(word2id), max_len)


def test_create_data_loaders(fake_data: Tuple[pd.Series, pd.Series]) -> None:
    sequence, targets = fake_data
    word2id = build_vocab(sequence)
    fam2label = build_labels(targets)

    batch_size = 1
    num_workers = 1
    seq_max_len = 10

    train = SequenceDataset(word2id, fam2label, seq_max_len, fake_data_path, "train")
    dev = SequenceDataset(word2id, fam2label, seq_max_len, fake_data_path, "dev")
    test = SequenceDataset(word2id, fam2label, seq_max_len, fake_data_path, "test")

    dataloaders = create_data_loaders(train, dev, test, batch_size, num_workers)

    assert isinstance(dataloaders, dict)
    assert 'train' in dataloaders and isinstance(dataloaders['train'], DataLoader)
    assert 'dev' in dataloaders and isinstance(dataloaders['dev'], DataLoader)
    assert 'test' in dataloaders and isinstance(dataloaders['test'], DataLoader)

    assert dataloaders['train'].batch_size == batch_size
    assert dataloaders['dev'].batch_size == batch_size
    assert dataloaders['test'].batch_size == batch_size

    assert dataloaders['train'].num_workers == num_workers
    assert dataloaders['dev'].num_workers == num_workers
    assert dataloaders['test'].num_workers == num_workers
