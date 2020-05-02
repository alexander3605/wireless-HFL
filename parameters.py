import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='')
    parser.add_argument('--is_folder', type=bool, default=False, help='')
    parser.add_argument('--device', type=str, default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"), help='')
    parser.add_argument('--n_experiments', type=int, default=1, help='')
    args = parser.parse_args()

    return args