			Graph Convolutional Network (GCN) Implementation with PyTorch

This project implements a Graph Convolutional Network (GCN) using PyTorch. 
The implementation includes four main components: `main.py`, `model.py`, `layers.py`, and `utils.py`. 
Below is a explanation of the implementation 


## Introduction

Graph Convolutional Networks (GCNs) are a type of neural network designed to operate on graph-structured data. 
This implementation uses the Cora dataset to demonstrate the functionality of GCNs. 
The main goal is to perform node classification on the graph data.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib



## Project Structure

- `main.py`: Main script to run the GCN training and testing.
- `model.py`: Defines the GCN model.
- `layers.py`: Contains the implementation of the graph convolutional layer.
- `utils.py`: Utility functions for data loading and preprocessing.

## Usage

To run the GCN training and testing, use the following command:

```bash
python main.py
```

You can customize various training parameters using command-line arguments:

- `--disable_cuda`: Disables CUDA training (default: False).
- `--fast_mode`: Enables fast mode which skips validation during training (default: False).
- `--random_seed`: Sets the random seed for reproducibility (default: 42).
- `--num_epochs`: Sets the number of epochs for training (default: 300).
- `--learning_rate`: Sets the initial learning rate for the optimizer (default: 0.01).
- `--weight_decay`: Sets the weight decay (L2 regularization) for the optimizer (default: 5e-4).
- `--hidden_units`: Sets the number of hidden units in the model (default: 16).
- `--dropout_rate`: Sets the dropout rate (default: 0.5).

## Detailed Explanation

i. main.py

The `main.py` script is the entry point for the GCN training and testing. It performs the following tasks:

1. **Argument Parsing and Setup**: Parses command-line arguments and sets up training parameters.
2. **Data Loading**: Loads the Cora dataset using the `load_data` function from `utils.py`.
3. **Model and Optimizer Setup**: Initializes the GCN model and optimizer.
4. **Training and Validation**: Defines the `train` function to execute one epoch of training and validation.
5. **Testing**: Defines the `test` function to evaluate the model on the test set.
6. **Plotting**: Plots the training and validation metrics.
7. **Running the Training Loop**: Executes the training loop for the specified number of epochs and prints the results.

ii. model.py

The `model.py` script defines the GCN model. It contains the following components:

1. **GCN Class**: A subclass of `torch.nn.Module` that defines the GCN architecture.
    - **Layers**: Defines two graph convolutional layers (`gc1` and `gc2`).
    - **Dropout**: Applies dropout for regularization.
    - **Forward Method**: Defines the forward pass of the model, which includes graph convolution, ReLU activation, dropout, and log softmax.

iii. layers.py

The `layers.py` script contains the implementation of the graph convolutional layer. It includes:

1. **GraphConvolution Class**: A subclass of `torch.nn.Module` that implements a single graph convolutional layer.
    - **Initialization**: Initializes the layer with input and output feature sizes and optional bias.
    - **Parameter Initialization**: Initializes the weight and bias parameters.
    - **Forward Method**: Defines the forward pass of the layer, performing matrix multiplication and adding bias if present.

iv. utils.py

The `utils.py` script contains utility functions for data loading and preprocessing. It includes:

1. **encode_onehot Function**: Converts categorical labels to one-hot encodings.
2. **load_data Function**: Loads the Cora dataset, builds the graph, normalizes features and adjacency matrix, and defines train, validation, and test indices.
3. **normalize Function**: Row-normalizes a sparse matrix.
4. **sparse_mx_to_torch_sparse_tensor Function**: Converts a scipy sparse matrix to a torch sparse tensor.

