
import os
import torch
import logging
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args

def main(args):


if __name__ == '__main__':
    args = parse_args()
    main(args)

    model = build_model()

    train_dataset = build_dataset(, 'train')
    val_dataset = build_dataset(, 'train')