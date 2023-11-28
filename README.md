# Protein Classifier Refactoring and Enhancement

## I. Introduction

Welcome to this Protein Classifier project. This project involves building and improving a protein classifier using PyTorch. The classifier's task is to assign the corresponding [Pfam](https://en.wikipedia.org/wiki/Pfam) family to each protein. The provided model is inspired by [ProtCNN](https://www.biorxiv.org/content/10.1101/626507v3.full), and the dataset used is from the [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split).

For the bonus section, another model has been added, inspired by the ProtTrans [article](https://pubmed.ncbi.nlm.nih.gov/34232869/). This new architecture uses a pretrained transformer available on [huggingface](https://huggingface.co/). By default, the model used is '[prot_t5_base_mt_uniref50](https://huggingface.co/Rostlab/prot_t5_base_mt_uniref50)', but other more powerful models such as '[prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)' can undoubtedly be used. The code is largely inspired by the work of the ProtTrans project, whose code is available on [github](https://github.com/agemagician/ProtTrans#prediction).

## II. Getting Start

To start working on this project, follow these steps:

### Create environment

* Clone the repository: ```git clone git@github.com:Aaramis/Protein_Classifier.git```
* Create a virtual environment: ```python -m venv .venv```
* Activate the virtual environment: ```source .venv/bin/activate``` (Linux/Mac) or ```.venv\Scripts\activate``` (Windows)
* Install dependencies: ```pip install -r requirements.txt```
* Build the Docker image: ```docker build -t protein_classifier .```
* Run the Docker container: ```docker run -it protein_classifier```

### Download data

Please download the data available [here](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split). 


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

### Models Arguments

| Argument          | Description                                | Default Value       |
|-------------------|--------------------------------------------|---------------------|
| --model_type      | Type of model to use : CNN or T5           | str                 |
| --train           | Flag to trigger training                   | _Flag_              |
| --eval            | Flag to trigger evaluation                 | _Flag_              |
| --model_name      | Model name to use                          | prot_cnn_model.pt   |

### Hyperparameters Arguments


| Argument          | Description                                | Default Value       |
|-------------------|--------------------------------------------|---------------------|
| --seq_max_len     | Maximum length of sequences                | 120                 |
| --batch_size      | Batch size for training and evaluation     | 1                   |
| --epochs          | Number of training epochs                  | 25                  |
| --gpus            | Number of GPUs to use for training         | 1                   |
| --num_workers     | Number of data loader workers              | 1                   |
| --num_classes     | Number of classes (family labels)          | 16652               |


### CNN Arguements

| Argument          | Description                                | Default Value       |
|-------------------|--------------------------------------------|---------------------|
| --lr              | Learning rate                              | 1e-2                |
| --momentum        | SGD momentum                               | 0.9                 |
| --weight_decay    | Weight decay for regularization            | 1e-2                |


### T5 Arguments

| Argument          | Description                 | Default Value                     |
|-------------------|-----------------------------|-----------------------------------|
| --accum           | Gradient Accumulation       | 2                                 |
| --dropout         | Dropout Rate                | 0.2                               |
| --pretained_model | Pretrained model name       | "Rostlab/prot_t5_base_mt_uniref50"|

### Evaluation and Prediction arguments

| Argument          | Description                                | Default Value       |
|-------------------|--------------------------------------------|---------------------|
| --predict         | Flag to trigger prediction                 | _Flag_              |
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

### CNN

#### Training

By default, the model will be saved in ``./output/models/[model_name]``. If you want to optimize the protCNN model's hyperparameters, some arguments are available for direct CLI interaction.

Example :

```
python main.py --model_type CNN --train --data_path ./tests/data --epochs 25  --lr 0.8  --model_name prot_cnn_model.pt
```

If you wish, you can follow the training using tensorboard. To do so, write the following command line in another terminal, then go to the following [address](http://localhost:6006/)

```
tensorboard --logdir lightning_logs/
```

### Evaluating

You can test the quality of a model using the ``--eval`` flag, which will measure the accuracy of the ``./randomsplit/test`` folder.

Example :

```
python main.py --model_type CNN --eval --data_path ./tests/data --model_path ./models --model_name prot_cnn_model.pt
```

### Predict the familly 

To predict the family of a protein, you can directly predict a sequence or a csv file using the ``--predict`` command. The result will be displayed in the terminal and saved in ``./output/predict/prediction.csv``.


Example for one sequence:

```
python main.py --model_type CNN --predict --model_path ./models  --model_name prot_cnn_model.pt  --sequence LTDYDNIRNCCKEATVCPKCWKFMVLAVKILDFLLDDMFGFN
```

Example for multiple sequences:

```
python main.py --model_type CNN --predict --model_path ./models  --model_name prot_cnn_model.pt  --csv ./tests/data/test.csv
```


### T5

#### Training

By default, the model will be saved in ``./output/models/[model_name]``. If you want to optimize the protCNN model's hyperparameters, some arguments are available for direct CLI interaction.

Example :

```
python main.py --model_type T5 --train --data_path ./tests/data --epochs 25  --accum 2  --model_path ./models --model_name PT5_secstr_finetuned.pth --pretrained_model Rostlab/prot_t5_base_mt_uniref50
```

### Evaluating

You can test the quality of a model using the ``--eval`` flag, which will measure the accuracy of the ``./randomsplit/test`` folder.

Example :

```
python main.py --model_type T5 --eval --data_path ./tests/data --model_path ./models --model_name PT5_secstr_finetuned.pth --pretrained_model Rostlab/prot_t5_base_mt_uniref50
```

### Predict the familly 

To predict the family of a protein, you can directly predict a sequence or a csv file using the ``--predict`` command. The result will be displayed in the terminal and saved in ``./output/predict/prediction.csv``.


Example for one sequence:

```
python main.py --model_type T5 --predict --model_path ./models --model_name PT5_secstr_finetuned.pth --pretrained_model Rostlab/prot_t5_base_mt_uniref50  --model_name prot_cnn_model.pt  --sequence LTDYDNIRNCCKEATVCPKCWKFMVLAVKILDFLLDDMFGFN
```

Example for multiple sequences:

```
python main.py --model_type T5 --predict --model_path ./models --model_name PT5_secstr_finetuned.pth --pretrained_model Rostlab/prot_t5_base_mt_uniref50 --model_name prot_cnn_model.pt  --csv ./tests/data/test.csv
```