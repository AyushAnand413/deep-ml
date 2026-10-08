import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        # TODO: call the parent constructor and register an nn.Linear as self.linear
        super().__init__() #initializes the parent nn.Module
        self.linear = nn.Linear(in_features,out_features)
        #nn.Linear(3, 1) creates a layer with three weights and one bias means 3 feat and 1 output pytorch automatically creates it weights and bias
        #self.linear saves this layer inside the model so we can use it later 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return the output of the linear layer
        return self.linear(x)
    # __init__() creates and stores the layer.
    # forward() uses the stored layer to calculate the output.
