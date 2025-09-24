


import logging
import sys
from train.train import train_model
from utils.logging import setup_logging
from validation.validate import validate_yolo_model

def print_usage():
    print("Usage:")
    print("  python main.py validate <model> <dataset>")
    print("  python main.py train <dataset>")

if __name__ == "__main__":
    setup_logging("app/training.log", logging.INFO)

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "validate":
        if len(sys.argv) != 4:
            print_usage()
            sys.exit(1)
        model_id = sys.argv[2]
        validate_dataset = sys.argv[3]
        validate_yolo_model(model_id=model_id, validate_dataset=validate_dataset)
    elif command == "train":
        if len(sys.argv) != 3:
            print_usage()
            sys.exit(1)
        train_dataset = sys.argv[2]
        train_model(train_dataset=train_dataset, epochs=1)
    else:
        print_usage()
        sys.exit(1)
    