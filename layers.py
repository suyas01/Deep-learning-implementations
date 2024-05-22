import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# Define the GraphConvolution class
class GraphConvolution(Module):
  
    def __init__(self, in_features, out_features, bias=True):
        # Initialize the module
        super(GraphConvolution, self).__init__()
        # Define the number of input and output features
        self.in_features = in_features
        self.out_features = out_features
        # Initialize the weight parameter
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Initialize the bias parameter if bias is True
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # Reset the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the weights with values sampled from a uniform distribution
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # Initialize the biases if they are not None
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Perform matrix multiplication between input and weights
        support = torch.mm(input, self.weight)
        # Perform sparse matrix multiplication between adjacency matrix and support
        output = torch.spmm(adj, support)
        # Add bias if it is not None
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# This is a simple GCN layer

    def __repr__(self):
        # Return a string representation of the layer
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
