import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        # Define the first graph convolutional layer
        self.gc1 = GraphConvolution(nfeat, nhid)

        # Define the second graph convolutional layer
        self.gc2 = GraphConvolution(nhid, nclass)

        # Dropout for regularization
        self.dropout = dropout

    def forward(self, x, adj):
        # Perform the first graph convolution followed by a ReLU activation
        x = F.relu(self.gc1(x, adj))

        # Apply dropout for regularization
        x = F.dropout(x, self.dropout, training=self.training)

        # Perform the second graph convolution
        x = self.gc2(x, adj)

        # Apply a log softmax function to the output
        return F.log_softmax(x, dim=1)