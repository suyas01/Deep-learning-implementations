import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# I had to set the environment variable "KMP_DUPLICATE_LIB_OK" to "TRUE" to prevent
# errors related to the loading of duplicate libraries. This issue can occur when 
# multiple versions of the Intel Math Kernel Library (MKL) are loaded simultaneously 
# by different libraries (e.g., NumPy, SciPy) that depend on MKL for numerical computations. 
# so that sloves the problem I had before


from __future__ import division
from __future__ import print_function
#__future__ imports ensure compatibility with Python 3 behavior in Python 2 code so there will be no conflict
# between each other should someone wants to use it 

import time
import argparse
import numpy as np
from torch.utils.data import  DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
 
from utils import load_data, accuracy
from models import GCN
import matplotlib.pyplot as plt
# These are the necessary libraries that must be imported
 

# Now the task is to perform argument parsing and setup for the GCN project 
# Training settings
parser = argparse.ArgumentParser()

# Argument to disable CUDA training (GPU acceleration)
parser.add_argument('--disable_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
# Argument to enable fast mode which skips validation during the training pass
parser.add_argument('--fast_mode', action='store_true', default=False,
                    help='Validate during training pass.')
# Argument to set the random seed for reproducibility
parser.add_argument('--random_seed', type=int, default=42, help='Random seed.')
# Argument to set the number of epochs for training
parser.add_argument('--num_epochs', type=int, default=300,
                    help='Number of epochs to train.')
# Argument to set the initial learning rate for the optimizer
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Initial learning rate.')
# Argument to set the weight decay (L2 regularization) for the optimizer
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
# Argument to set the number of hidden units in the model
parser.add_argument('--hidden_units', type=int, default=16,
                    help='Number of hidden units.')
# Argument to set the dropout rate (probability of dropping units)
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

# Check if CUDA is available and not disabled
args.cuda = not args.disable_cuda and torch.cuda.is_available()

# Set the manual seed for random number generation for reproducibility
torch.manual_seed(args.random_seed)

 
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
 
# Model and optimizer setup
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden_units,
            nclass=labels.max().item() + 1,
            dropout=args.dropout_rate)

# Initialize the optimizer with model parameters, learning rate, and weight decay
optimizer = optim.Adam(model.parameters(),
                       lr=args.learning_rate, weight_decay=args.weight_decay)

# If CUDA is available and enabled, move the model and data to the GPU
if args.cuda:
    model.cuda()              # Move the model to the GPU
    features = features.cuda()  # Move the features tensor to the GPU
    adj = adj.cuda()          # Move the adjacency matrix to the GPU
    labels = labels.cuda()    # Move the labels to the GPU
    idx_train = idx_train.cuda()  # Move the training indices to the GPU
    idx_val = idx_val.cuda()      # Move the validation indices to the GPU
    idx_test = idx_test.cuda()    # Move the test indices to the GPU

# Initialize a list to store the loss values during training
Loss_list = []

# Explanation:
# Using CUDA (GPU acceleration) can significantly speed up the training and evaluation process,
# especially for large models and datasets. By moving the model and data to the GPU, 
# advantage of parallel processing capabilities of the GPU can be taken, leading to faster computation times.
 
 
accval = []  # List to store validation accuracies

def train(epoch):
    t = time.time()  # Record the start time of the epoch
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear the gradients of all optimized parameters
    output = model(features, adj)  # Perform a forward pass through the model
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # Calculate the training loss
    acc_train = accuracy(output[idx_train], labels[idx_train])  # Calculate the training accuracy
    loss_train.backward()  # Backpropagate the loss
    optimizer.step()  # Update the model parameters

    if not args.fast_mode:
        model.eval()  # Set the model to evaluation mode
        output = model(features, adj)  # Perform another forward pass through the model

    # Calculate the validation loss and accuracy
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # Print training and validation metrics
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    # Append metrics to their respective lists for later analysis
    Loss_list.append(loss_train.item())  # Store training loss
    Accuracy_list.append(acc_train.item())  # Store training accuracy
    lossval.append(loss_val.item())  # Store validation loss
    accval.append(acc_val.item())  # Store validation accuracy

# Explanation:
# The train function executes one epoch of training. It performs a forward pass to compute 
# the model's output, calculates the loss and accuracy, backpropagates the loss to update 
# the model's parameters, and computes validation metrics if fast mode is disabled.
# Metrics are printed and stored for further analysis and plotting.
 
 
 

# Test function
def test():
    model.eval()  # Set the model to evaluation mode
    output = model(features, adj)  # Perform a forward pass through the model
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])  # Calculate the test loss
    acc_test = accuracy(output[idx_test], labels[idx_test])  # Calculate the test accuracy

    # Print test set results
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    # Detach and convert the test accuracy and loss to numpy arrays
    acc = acc_test.detach().numpy()
    loss = loss_test.detach().numpy()

    # Debug: Print the types of loss_test and acc_test to ensure correctness
    print(type(loss_test))
    print(type(acc_test))

    # Define two arrays (additional arrays can be defined here if needed)
 
 
# Train the model
t_total = time.time()  # Record the start time of training

# Iterate over the specified number of epochs
for epoch in range(args.epochs):
    train(epoch)  # Perform training for the current epoch

print("Optimization Finished!")  # Print a message indicating the end of training
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))  # Print the total time elapsed during training

# Plotting training and validation metrics
plt.plot([i for i in range(len(lossval))], lossval)  # Plot validation loss
plt.plot([i for i in range(len(accval))], accval)    # Plot validation accuracy
plt.show()  # Display the plots

# Testing the model
test()  # Perform testing to evaluate the model's performance on the test set