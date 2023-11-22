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
    # ... (initialisation du modèle, des dataloaders, etc.)
    # dataloaders = get_dataloaders(args.data_path, args.seq_max_len, args.batch_size, args.num_workers)
    # model = ProtCNN(args.num_classes)

    # train_protein_classifier(model, dataloaders, args.epochs, args.gpus)

if __name__ == "__main__":
    main()
