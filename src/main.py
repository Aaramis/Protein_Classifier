# main.py

from config import parse_args, check_arguments
from log import initialize_logging
# from data_loader import get_dataloaders
# from model import ProtCNN
# from train import train_protein_classifier

def main():

    args = parse_args()
    initialize_logging(args)
    check_arguments(args)

if __name__ == "__main__":
    main()
