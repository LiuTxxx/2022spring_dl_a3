from torch import optim
import torch.nn as nn

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from .dataset import PalindromeDataset
from .lstm import LSTM


class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def accuracy(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i, (inputs, labels) in enumerate(data_loader):
            if i == 100:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(config, print_res):

    # Initialize the model that we are going to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size).to(device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    acc_arr = []
    loss_arr = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        # Add more code here ...
        optimizer.zero_grad()
        output = model(batch_inputs)
        loss = criterion(output, batch_targets)
        loss_arr.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if print_res and step % 100 == 0:
            print("Step [{}/{}] Loss: {:.4f}, Acc: {:.4f}"
                  .format(step + 1, config.train_steps, loss.item(), accuracy(model, data_loader)))
#         if step % 100 == 0:
#             print(step)
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print(f'Done training with input length T = {config.input_length}.')
    return accuracy(model, data_loader)


def get_config():
    config = Dict()
    config.input_length = 10
    config.input_dim = 1
    config.num_classes = 10
    config.num_hidden = 128
    config.batch_size = 128
    config.lr = 0.001
    config.train_steps = 10000
    config.max_norm = 10.0
    return config


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = get_config()
    # Train the model
    train(config)