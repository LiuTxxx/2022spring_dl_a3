from matplotlib import pyplot as plt
from torch import optim
import torch.nn as nn

import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from .dataset import PalindromeDataset
from .vanilla_rnn import VanillaRNN


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


def train(t, print_res, config = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model that we are going to use
    model = VanillaRNN(t, 1, 128, 10, 128).to(device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(t + 1)
    data_loader = DataLoader(dataset, 128, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    acc_arr = []
    loss_arr = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        # Add more code here ...
        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.zero_grad()
        output = model(batch_inputs)
        loss = criterion(output, batch_targets)
        loss_arr.append(loss.item())
        loss.backward()
        optimizer.step()

        if print_res and step % 3000 == 0:
            acc_arr.append(accuracy(model, data_loader))
            print("Step [{}/{}] Loss: {:.4f}, Acc: {:.4f}"
                  .format(step + 1, 10000, loss.item(), accuracy(model, data_loader)))

        if step == 10000:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
#     if print_res:
#         name = "RNN: "
#         idx_arr = 10 * np.arange(len(acc_arr))
#         plt.title(name + 'loss')
#         plt.plot(loss_arr)
#         plt.show()
#         plt.title(name + 'train accuracy')
#         plt.plot(idx_arr, acc_arr)
#         plt.show()

    print(f'Done training with input length T = {t}.')
    return accuracy(model, data_loader)


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

    config = parser.parse_args()
    # Train the model
    train(10, False, config)
