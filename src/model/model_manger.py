from typing import Dict, Union
import os
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.log import write_logs, check_directory, LogStatus
from src.model.utils import find_key_by_value
from argparse import Namespace


class ModelManager:
    def __init__(self, model, word2id: Dict[str, int], fam2label: Dict[Union[str, int], int], args: Namespace):
        self.model = model
        self.word2id: Dict[str, int] = word2id
        self.fam2label: Dict[Union[str, int], int] = fam2label
        self.seq_max_len: int = args.seq_max_len
        self.batch_size: int = args.batch_size
        self.num_workers: int = args.num_workers
        self.num_classes: int = args.num_classes
        self.epochs: int = args.epochs
        self.model_name: str = args.model_name
        self.model_path: str = args.model_path
        self.output_path: str = args.output_path
        self.data_path: str = args.data_path
        self.csv: str = args.csv

    def train(self, train_loader: DataLoader, dev_loader: DataLoader) -> None:
        tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/", name="my_experiment")
        pl.seed_everything(0)
        trainer = pl.Trainer(accelerator="auto", max_epochs=self.epochs, logger=tb_logger)
        trainer.fit(self.model, train_loader, dev_loader)

        torch.save(self.model.state_dict(), os.path.join(self.model_path, self.model_name))

    def evaluate(self, model, test_loader: DataLoader) -> None:
        write_logs("Starting Evaluation", LogStatus.INFO, True)
        model.eval()
        all_preds = []
        all_labels = []

        try:

            with torch.no_grad(), tqdm(total=len(test_loader)) as pbar:
                for batch in test_loader:
                    x, y = batch['sequence'], batch['target']
                    y_hat = model(x)
                    preds = torch.argmax(y_hat, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    pbar.update(1)

            accuracy = accuracy_score(all_labels, all_preds)
            write_logs(f"Test Accuracy: {accuracy:.4f}", LogStatus.INFO, True)

        except KeyboardInterrupt:
            write_logs("Interrupted evaluation", LogStatus.CRITICAL, True)

    def predict_one_sequence(self, model, seq, display: bool) -> Union[str, None]:

        write_logs(f"Starting Prediction of {seq}", LogStatus.INFO, display)

        seq = [self.word2id.get(word, self.word2id['<unk>']) for word in seq[:self.seq_max_len]]
        seq += [self.word2id['<pad>']] * (self.seq_max_len - len(seq))
        seq = torch.from_numpy(np.array(seq))
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id))
        one_hot_seq = one_hot_seq.permute(1, 0)
        one_hot_seq = one_hot_seq.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            prediction = model(one_hot_seq)

        predicted_class_index = torch.argmax(prediction, dim=1).item()
        predicted_class_index = find_key_by_value(self.fam2label, predicted_class_index)

        write_logs(f"Predicted class {predicted_class_index}", LogStatus.INFO, display)
        return predicted_class_index

    def predict_csv(self, model) -> None:

        write_logs(f"Starting Prediction of {self.csv}", LogStatus.INFO, True)

        df = pd.read_csv(self.csv, index_col=None, usecols=["sequence"])
        lst_pred = [self.predict_one_sequence(model, seq, False) for seq in df['sequence']]
        df["pred"] = pd.DataFrame(lst_pred)

        check_directory(f"{self.output_path}/prediction", "prediction directory")
        file = f"{self.output_path}/prediction/prediction.csv"
        df.to_csv(file, index=None)

        if os.path.isfile(file):
            write_logs(f"Prediction available here -> {file}", LogStatus.INFO, True)
