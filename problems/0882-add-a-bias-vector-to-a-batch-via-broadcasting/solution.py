import torch

def add_bias(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: add b to every row of x using broadcasting
    return x+b
    #pytorch in if x= (n,d) and b is (d,) then x+b = (n,d)