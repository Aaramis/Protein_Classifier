from model.protein_cnn import ProtCNN
from log import write_logs, LogStatus
import torch
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import accuracy_score


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

    write_logs("Starting Test", LogStatus.INFO, True)
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
