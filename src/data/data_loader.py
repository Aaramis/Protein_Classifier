from collections import Counter
from typing import Dict, Tuple, Union
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from src.log import write_logs, LogStatus


def reader(partition: str, data_path: Union[str, Path]) -> Tuple[pd.Series, pd.Series]:
    write_logs(f"Loading Data {partition}", LogStatus.INFO, True)
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))

    all_data = pd.concat(data)
    return all_data["sequence"], all_data["family_accession"]


def build_labels(targets: pd.Series) -> Dict[Union[str, int], int]:
    write_logs("Building Labels", LogStatus.INFO, False)
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0
    write_logs(f"There are {len(fam2label)} labels.", LogStatus.INFO, True)
    return fam2label


def get_amino_acid_frequencies(data: pd.Series) -> pd.DataFrame:
    write_logs("Getting Amino Acid Frequencies", LogStatus.INFO, False)
    aa_counter = Counter()
    for sequence in data:
        aa_counter.update(sequence)
    return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})


def build_vocab(data: pd.Series) -> Dict[str, int]:
    write_logs("Building Vocab", LogStatus.INFO, False)
    voc = set()
    rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
    for sequence in data:
        voc.update(sequence)
    unique_AAs = sorted(voc - rare_AAs)
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    return word2id


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        word2id: Dict[str, int],
        fam2label: Dict[Union[str, int], int],
        max_len: int,
        data_path: Union[str, Path],
        split: str,
    ):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        self.data, self.label = reader(split, data_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])
        return {'sequence': seq, 'target': label}

    def preprocess(self, text: str) -> torch.Tensor:
        seq = [self.word2id.get(word, self.word2id['<unk>']) for word in text[:self.max_len]]
        seq += [self.word2id['<pad>']] * (self.max_len - len(seq))
        seq = torch.from_numpy(np.array(seq))
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id))
        one_hot_seq = one_hot_seq.permute(1, 0)
        return one_hot_seq

    def create_dataloader(self, data: object, batch: int, workers: int) -> DataLoader:
        return torch.utils.data.DataLoader(
                data,
                batch_size=batch,
                shuffle=True,
                num_workers=workers,
            )
