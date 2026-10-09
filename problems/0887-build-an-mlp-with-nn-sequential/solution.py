import torch
import torch.nn as nn

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    # TODO: return a Sequential of Linear -> ReLU -> Linear
    #nn.Sequential is a container nn.Module that runs its child modules in the order they were passed. It's the simplest way to compose layers when the architecture is a straight pipeline with no branching.
    #our task is to connect the three components indim hiddendim and outdim in seq 
    #so we can take in dim and use nn.linear to create a fully connected layer

    return nn.Sequential(
        nn.Linear(in_dim,hidden_dim), #first layter
        nn.ReLU(), #relu on that
        nn.Linear(hidden_dim,out_dim), #final layer
    )
