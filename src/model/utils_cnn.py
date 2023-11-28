from src.model.protein_cnn import ProtCNN
from collections import OrderedDict
from src.log import write_logs, LogStatus
import torch
import os
from typing import Optional, Union, Dict


def get_last_layer(model: OrderedDict) -> Optional[str]:
    """
    Get the last layer key in the model.

    Args:
        model (OrderedDict): The model's state dictionary.

    Returns:
        Optional[str]: The key of the last layer or None if not found.
    """
    last_layer_key = None
    for key in model.keys():
        if 'model' in key:
            last_layer_key = key
    return last_layer_key


def get_num_classes(model: OrderedDict, last_layer: Optional[str]) -> int:
    """
    Get the number of classes from the last layer of the model.

    Args:
        model (OrderedDict): The model's state dictionary.
        last_layer (Optional[str]): The key of the last layer.

    Returns:
        int: The number of classes or 0 if last_layer is None.
    """
    if last_layer is not None:
        num_classes = model[last_layer].size()
        return list(num_classes)[0]
    return 0


def load_model(model_path: str, model_name: str) -> Optional[ProtCNN]:
    """
    Load a PyTorch model from a file.

    Args:
        model_path (str): The path to the directory containing the model file.
        model_name (str): The name of the model file.

    Returns:
        Optional[ProtCNN]: The loaded model or None if unsuccessful.
    """
    write_logs("Loading model", LogStatus.INFO, False)

    file_path = f"{model_path}/{model_name}"

    if not os.path.isfile(file_path):
        write_logs(f"'{file_path}' is not a valid. EXIT", LogStatus.CRITICAL, True)
        exit()

    try:
        pretrained_state_dict = torch.load(file_path)
    except Exception as e:
        write_logs(f"Error loading model: {str(e)}", LogStatus.ERROR, False)
        return None

    last_layer = get_last_layer(pretrained_state_dict)
    num_classes = get_num_classes(pretrained_state_dict, last_layer)

    if num_classes:
        prot_cnn = ProtCNN(num_classes=num_classes)
        prot_cnn.load_state_dict(pretrained_state_dict)
        return prot_cnn

    write_logs("Number of classes incorrect", LogStatus.CRITICAL, False)
    return None


def find_key_by_value(dictionary: Dict[Union[str, int], int], value: int) -> Optional[Union[str, int]]:
    """
    Find a key in a dictionary by its value.

    Args:
        dictionary (Dict[Union[str, int], int]): The dictionary to search.
        value (int): The value to find.

    Returns:
        Optional[Union[str, int]]: The key corresponding to the given value, or None if not found.
    """
    for key, val in dictionary.items():
        if val == value:
            return key
    return None
