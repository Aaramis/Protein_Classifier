import torch
import os
import numpy as np
import pandas as pd
import random
from transformers import set_seed
from datasets import Dataset
from src.log import write_logs, LogStatus, check_directory
from src.model.utils_cnn import find_key_by_value
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from typing import List, Dict
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration


def predict_csv_t5(model, tokenizer, csv_path: str, output_path: str, fam2label: Dict[str, int]) -> None:
    """
    Perform predictions on sequences from a CSV file and save the results to a new CSV file.

    Args:
        model (torch.nn.Module): The T5 model for prediction.
        tokenizer: The tokenizer for processing input sequences.
        csv_path (str): Path to the CSV file containing sequences.
        output_path (str): Path to the output directory.
        fam2label (Dict[str, int]): A dictionary mapping family names to label indices.
    """
    write_logs(f"Starting Prediction of {csv_path}", LogStatus.INFO, True)

    # Load the CSV file
    df = pd.read_csv(csv_path, index_col=None, usecols=["sequence"])

    # Predict for each sequence in the DataFrame
    lst_pred = [predict_one_sequence_t5(model, tokenizer, seq, fam2label, False) for seq in df['sequence']]

    # Add predictions to the DataFrame
    df["pred"] = pd.DataFrame(lst_pred)

    # Create the prediction directory if it doesn't exist
    check_directory(f"{output_path}/prediction", "prediction directory")

    # Save the predictions to a CSV file
    file = f"{output_path}/prediction/prediction.csv"
    df.to_csv(file, index=None)

    if os.path.isfile(file):
        write_logs(f"Prediction available here -> {file}", LogStatus.INFO, True)


def predict_one_sequence_t5(model, tokenizer, seq: str, fam2label: Dict[str, int], display: bool) -> str:
    """
    Perform prediction on a single sequence using the T5 model.

    Args:
        model (torch.nn.Module): The T5 model for prediction.
        tokenizer: The tokenizer for processing input sequences.
        seq (str): Input sequence for prediction.
        fam2label (Dict[str, int]): A dictionary mapping family names to label indices.
        display (bool): Whether to display log messages.

    Returns:
        str: Predicted class name.
    """
    write_logs(f"Starting Prediction of {seq}", LogStatus.INFO, display)

    # Tokenize the input sequence
    tokenized_seq = tokenizer(seq, max_length=1024, padding=True, truncation=True, return_tensors="pt")

    # Perform the prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_seq)

    # Extract the predicted logits from the model outputs
    logits = outputs.logits

    # Choose the predicted token with the highest probability as the output
    predicted_class_index = torch.argmax(logits[0][0], dim=-1)

    # Convert the predicted index to the corresponding class name
    predicted_class_name = find_key_by_value(fam2label, predicted_class_index.item())

    write_logs(f"Predicted class {predicted_class_name}", LogStatus.INFO, display)
    return predicted_class_name


def evaluate_t5(model, tokenizer, df_test: pd.DataFrame) -> None:
    """
    Evaluate the T5 model on a test dataset.

    Args:
        model (torch.nn.Module): The T5 model for evaluation.
        tokenizer: The tokenizer for processing input sequences.
        df_test (pd.DataFrame): DataFrame containing test sequences and labels.
    """
    # Check if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    write_logs("Starting Evaluation", LogStatus.INFO, True)
    model.eval()
    all_preds = []
    all_labels = []

    try:
        with torch.no_grad(), tqdm(total=len(df_test)) as pbar:
            for _, row in df_test.iterrows():
                sequence = row['sequence']
                target = row['label']

                # Tokenize and convert to tensor
                inputs = tokenizer(sequence, max_length=1024, padding=True, truncation=True, return_tensors="pt")
                inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits[0][0], dim=-1).numpy()

                all_labels.append(target)
                all_preds.append(preds)
                pbar.update(1)

        accuracy = accuracy_score(all_labels, all_preds)
        write_logs(f"Test Accuracy: {accuracy:.4f}", LogStatus.INFO, True)

    except KeyboardInterrupt:
        write_logs("Interrupted evaluation", LogStatus.CRITICAL, True)


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def create_dataset(tokenizer: PreTrainedTokenizer, seqs: List[str], labels: List[int]) -> Dataset:
    """
    Create a dataset from sequences and labels using a tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer for processing input sequences.
        seqs (List[str]): List of input sequences.
        labels (List[int]): List of corresponding labels.

    Returns:
        Dataset: A Hugging Face Dataset object.
    """
    tokenized = tokenizer(seqs, max_length=1024, padding=True, truncation=True)
    dataset = Dataset.from_dict(tokenized)

    labels = [[lab] for lab in labels]

    dataset = dataset.add_column("labels", labels)

    return dataset


def save_model_T5(model: T5ForConditionalGeneration, filepath: str) -> None:
    """
    Save finetuned parameters of a T5 model.

    Args:
        model (T5ForConditionalGeneration): The T5 model.
        filepath (str): Path to save the model parameters.
    """
    # Create a dictionary to hold the non-frozen parameters
    non_frozen_params = {}

    # Iterate through all the model parameters
    for param_name, param in model.named_parameters():
        # If the parameter has requires_grad=True, add it to the dictionary
        if param.requires_grad:
            non_frozen_params[param_name] = param

    # Save only the finetuned parameters
    torch.save(non_frozen_params, filepath)
