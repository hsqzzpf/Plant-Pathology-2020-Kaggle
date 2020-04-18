# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model',
                        type=int,
                        help='models to choose',
                        default=0)

    parser.add_argument('--batch_size',
                        type=int,
                        help='batch_size',
                        default=64)

    parser.add_argument('--data_root',
                        type=str,
                        help='path to data folder',
                        default='../data/')

    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=5)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)


    return parser
