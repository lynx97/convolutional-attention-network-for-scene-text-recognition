import argparse
from model import SSCAN
from predict import Predict
from inference import Inference
import os
import time

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Train or test the NRTR model.')

    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Define if we train the model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Define if we test the model"
    )
    parser.add_argument(
        "-tf",
        "--train_file",
        type=str,
        nargs="?",
        help="file to contain label each line is: image_name & label, strip with \t",
        required=True
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        default='./save/'
    )
    parser.add_argument(
        "-ex",
        "--examples_path",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
        required=True
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=64
    )
    parser.add_argument(
        "-it",
        "--iteration_count",
        type=int,
        nargs="?",
        help="How many iteration in training",
        default=1000
    )
    parser.add_argument(
        "-vs",
        "--vocab_size",
        type=int,
        nargs="?",
        help="vocabulary size",
        default=64
    )

    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        default=True,
        help="Define if we try to load a checkpoint file from the save folder"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    print("Restore", args.restore)

    if args.train:
        ssan = SSCAN(
            args.batch_size,
            args.model_path,
            args.examples_path,
            args.vocab_size,
            args.train_file,
            args.restore
        )

        ssan.train(args.iteration_count)
    if args.test:
        ocr_test = Predict(
            args.batch_size,
            args.model_path,
            args.examples_path,
            args.vocab_size,
            args.train_file,
            True
        )
        ocr_test.pred()
if __name__ == '__main__':
    main()
