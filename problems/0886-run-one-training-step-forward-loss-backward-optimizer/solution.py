import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.
    #mse 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #it creates the optimizer with given lr  
    #clear grad accumulator 
    optimizer.zero_grad()
    predict = model(x) # forward pass 
    loss = F.mse_loss(predict, y) #loss sqrd diff bw pred and target
    loss.backward() #backprop
    optimizer.step() #update wieghts and bias

    return loss.item()