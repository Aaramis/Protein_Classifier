from model.protein_cnn import ProtCNN
from log import write_logs, check_directory, LogStatus
import torch
import os
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def get_last_layer(model: dict) -> Optional[str]:

    last_layer_key = None
    for key in model.keys():
        if 'model' in key:
            last_layer_key = key
    return last_layer_key


def get_num_classes(model: dict, last_layer: Optional[str]) -> int:

    if last_layer is not None:
        num_classes = model[last_layer].size()
        return list(num_classes)[0]
    return 0


def load_model(model_path: str, model_name: str) -> Optional[ProtCNN]:

    write_logs("Loading model", LogStatus.INFO, False)

    if model_path and model_name:
        file_path = f"{model_path}/{model_name}"
    else:
        write_logs("Model not findable", LogStatus.CRITICAL, False)
        return None

    pretrained_state_dict = torch.load(file_path)
    last_layer = get_last_layer(pretrained_state_dict)
    num_classes = get_num_classes(pretrained_state_dict, last_layer)

    if num_classes:
        prot_cnn = ProtCNN(num_classes)
        prot_cnn.load_state_dict(pretrained_state_dict)
        return prot_cnn

    write_logs("Number of classes incorrect", LogStatus.CRITICAL, False)
    return None


def test_model(model: ProtCNN, test_dataloader) -> None:

    write_logs("Starting Evauation", LogStatus.INFO, True)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad(), tqdm(total=len(test_dataloader)) as pbar:
        for batch in test_dataloader:
            x, y = batch['sequence'], batch['target']
            y_hat = model(x)
            preds = torch.argmax(y_hat, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            pbar.update(1)

    accuracy = accuracy_score(all_labels, all_preds)
    write_logs(f"Test Accuracy: {accuracy:.4f}", LogStatus.INFO, True)


def evaluate_one_sequence(args, model, sequence, word2id, display):

    write_logs(f"Starting Prediction of {sequence}", LogStatus.INFO, display)

    seq = [word2id.get(word, word2id['<unk>']) for word in sequence[:args.seq_max_len]]
    seq += [word2id['<pad>']] * (args.seq_max_len - len(seq))
    seq = torch.from_numpy(np.array(seq))
    one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(word2id))
    one_hot_seq = one_hot_seq.permute(1, 0)
    one_hot_seq = one_hot_seq.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prediction = model(one_hot_seq)

    predicted_class_index = torch.argmax(prediction, dim=1).item()
    write_logs(f"Predicted class {predicted_class_index}", LogStatus.INFO, display)

    return predicted_class_index


def evaluate_csv(args, model, word2id):

    write_logs(f"Starting Prediction of {args.csv}", LogStatus.INFO, True)

    df = pd.read_csv(args.csv, index_col=None, usecols=["sequence"])
    lst_pred = [evaluate_one_sequence(args, model, seq, word2id, False)for seq in df['sequence']]
    df["pred"] = pd.DataFrame(lst_pred)

    check_directory(f"{args.output_path}/prediction", "prediciton directory")
    file = f"{args.output_path}/prediction/prediction.csv"
    df.to_csv(file, index=None)

    if os.path.isfile(file):
        write_logs(f"Prediction available here -> {file}", LogStatus.INFO, True)
