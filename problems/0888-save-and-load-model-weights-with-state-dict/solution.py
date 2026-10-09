import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst
    #we have src - the source model whose weights you want to copy
    #dst - the destination model whose weights you want to overwrite
    #A PyTorch model's state_dict() contains its learnable parameters, such as weights and biases.
    buffer = io.BytesIO()
    torch.save(src.state_dict(),buffer)
    buffer.seek(0)
    dst.load_state_dict(torch.load(buffer, weights_only=True))