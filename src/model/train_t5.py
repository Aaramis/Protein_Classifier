import os
import re
import torch
import numpy as np
from transformers import TrainingArguments, Trainer, T5EncoderModel, T5Tokenizer, DataCollatorForTokenClassification
from src.model.protein_t5 import LoRAConfig, modify_with_lora, ClassConfig, T5EncoderForTokenClassification
from src.model.utils_t5 import set_seeds, create_dataset
from src.model.config_t5 import ds_config
from evaluate import load


def load_model_T5(filepath, num_labels, pretrained_model, dropout):
    # Load a new model
    model, tokenizer = PT5_classification_model(dropout, num_labels, pretrained_model)

    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model


def PT5_classification_model(dropout, num_labels, pretrained_model):
    # Load PT5 and tokenizer
    model = T5EncoderModel.from_pretrained(pretrained_model)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)

    # Create a new Classifier model with PT5 dimensions
    class_config = ClassConfig(dropout=dropout, num_labels=num_labels)
    class_model = T5EncoderForTokenClassification(model.config, class_config)

    # Set encoder and embedding weights to checkpoint weights
    class_model.shared = model.shared
    class_model.encoder = model.encoder

    # Delete the checkpoint model
    model = class_model
    del class_model

    # Print the number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ProtT5_Classifier\nTrainable Parameter: " + str(params))

    # Add model modification lora
    config = LoRAConfig()

    # Add LoRA layers
    model = modify_with_lora(model, config)

    # Freeze Embeddings and Encoder (except LoRA)
    for (param_name, param) in model.shared.named_parameters():
        param.requires_grad = False
    for (param_name, param) in model.encoder.named_parameters():
        param.requires_grad = False

    for (param_name, param) in model.named_parameters():
        if re.fullmatch(config.trainable_param_names, param_name):
            param.requires_grad = True

    # Print trainable Parameter
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("ProtT5_LoRA_Classifier\nTrainable Parameter: " + str(params) + "\n")

    return model, tokenizer


def train_per_residue(
    train_df,        # training data
    valid_df,        # validation data
    num_labels=3,    # number of classes
    batch=4,         # for training
    accum=2,         # gradient accumulation
    val_batch=16,    # batch size for evaluation
    epochs=10,       # training epochs
    lr=3e-4,         # recommended learning rate
    seed=42,         # random seed
    deepspeed=True,  # if the GPU is large enough, disable deepspeed for training speedup
    gpu=1,           # GPU selection (1 for the first GPU)
    dropout=0.2,     # Dropout rate
    pretrained_model="Rostlab/prot_t5_base_mt_uniref50"
):
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu-1)

    # Set all random seeds
    set_seeds(seed)

    # Load model
    model, tokenizer = PT5_classification_model(dropout, num_labels, pretrained_model)

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    train_df["sequence"] = train_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    valid_df["sequence"] = valid_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    # Add spaces between each amino acid for PT5 to correctly use them
    train_df['sequence'] = train_df.apply(lambda row: " ".join(row["sequence"]), axis=1)
    valid_df['sequence'] = valid_df.apply(lambda row: " ".join(row["sequence"]), axis=1)

    # Create Datasets
    train_set = create_dataset(tokenizer, list(train_df['sequence']), list(train_df['label']))
    valid_set = create_dataset(tokenizer, list(valid_df['sequence']), list(valid_df['label']))

    # Huggingface Trainer arguments
    args = TrainingArguments(
        "./",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=ds_config if deepspeed else None,
    )

    # Metric definition for validation data
    def compute_metrics(eval_pred):
        metric = load("accuracy")
        predictions, labels = eval_pred

        labels = labels.reshape((-1,))

        predictions = np.argmax(predictions, axis=2)
        predictions = predictions.reshape((-1,))

        predictions = predictions[labels != -100]
        labels = labels[labels != -100]

        return metric.compute(predictions=predictions, references=labels)

    # For token classification, we need a data collator here to pad correctly
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    return tokenizer, model, trainer.state.log_history
