# ProtCNN - Protein Classifier Refactoring and Enhancement

## I. Introduction

Welcome to the ProtCNN project! This project involves building and improving a protein classifier using PyTorch. The classifier's task is to assign the corresponding [Pfam](https://en.wikipedia.org/wiki/Pfam) family to each protein. The provided model is inspired by [ProtCNN](https://www.biorxiv.org/content/10.1101/626507v3.full), and the dataset used is from the [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split).

## II. Getting Start

To start working on this project, follow these steps:

* Clone the repository: ```git clone <repository_url>```
* Create a virtual environment: ```python -m venv venv ```
* Activate the virtual environment: ```source venv/bin/activate``` (Linux/Mac) or ```venv\Scripts\activate``` (Windows)
* Install dependencies: ```pip install -r requirements.txt```
* Build the Docker image: ```docker build -t protein_classifier .```
* Run the Docker container: ```docker run -it protein_classifier```


## III. Optional Arguments


| Argument          | Description                      |
|-------------------|----------------------------------|
| --help -h         | Show help message and exit       |

### Directory Arguments

| Argument          | Description                      | Default Value       |
|-------------------|----------------------------------|---------------------|
| --data_path       | Path to the data directory       | ./random_split      |
| --log_path        | Path to the log directory        | ./log               |
| --output_path,-o  | Path to the output directory     | ./output            |
| --model_path      | Path to the model                | ./models            |


### Training Arguments

| Argument          | Description                                | Default Value       |
|-------------------|--------------------------------------------|---------------------|
| --train           | Flag to trigger training                   | _Flag_              |
| --seq_max_len     | Maximum length of sequences                | 120                 |
| --batch_size      | Batch size for training and evaluation     | 1                   |
| --num_workers     | Number of data loader workers              | 1                   |
| --num_classes     | Number of classes (family labels)          | 16652               |
| --lr              | Learning rate                              | 1e-2                |
| --momentum        | SGD momentum                               | 0.9                 |
| --weight_decay    | Weight decay for regularization            | 1e-2                |
| --epochs          | Number of training epochs                  | 25                  |
| --gpus            | Number of GPUs to use for training         | 1                   |

### Evaluation and Prediction arguments

| Argument          | Description                                | Default Value       |
|-------------------|--------------------------------------------|---------------------|
| --eval            | Flag to trigger evaluation                 | _Flag_              |
| --predict         | Flag to trigger prediction                 | _Flag_              |
| --model_name      | Model name to use                          | prot_cnn_model.pt   |
| --sequence        | Sequence to predict                        | None                |
| --csv             | csv file with sequences to predict         | None                |

### Plots Arguments

| Argument          | Description                                | Default Value       |
|-------------------|--------------------------------------------|---------------------|
| --display_plots   | Display plots                              | 0                   |
| --save_plots      | Save plots in the output folder            | 0                   |
| --split           | Type of split to plot                      | train               |


## IV. Usage and CLI commands

### Display and save plots

This command lets you quickly visualize the distribution of family sizes, sequence lengths and amino acid frequencies for the train dataset. Plots will be saved by default in ```./output/plot```

```
python main.py --save_plots 1 --display_plots 1 --split train
```

### Training

Please download the data available [here](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split). By default, the model will be saved in ``./output/models/[model_name]``. If you want to optimize the protCNN model's hyperparameters, some arguments are available for direct CLI interaction.

Example :

```
python main.py --train --data_path ./tests/data --epochs 25  --lr 0.8  --model_name prot_cnn_model.pt
```

If you wish, you can follow the training using tensorboard. To do so, write the following command line in another terminal, then go to the following [address](http://localhost:6006/)

```
tensorboard --logdir lightning_logs/
```

### Evaluating

You can test the quality of a model using the ``--eval`` flag, which will measure the accuracy of the ``./randomsplit/test`` folder.

Example :

```
python main.py --eval --data_path ./tests/data --model_name prot_cnn_model.pt
```

### Predict the familly 

To predict the family of a protein, you can directly predict a sequence or a csv file using the ``--predict`` command. The result will be displayed in the terminal and saved in ``./output/predict/prediction.csv``.


Example for one sequence:

```
python main.py --predict --model_path ./models  --model_name prot_cnn_model.pt  --sequence LTDYDNIRNCCKEATVCPKCWKFMVLAVKILDFLLDDMFGFN
```

Example for multiple sequences:

```
python main.py --predict --model_path ./models  --model_name prot_cnn_model.pt  --csv ./tests/data/test.csv
```


https://elifesciences.org/articles/82819