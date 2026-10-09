import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_batches = 0
    first_batch_X_shape = None

    for batch_X, batch_y in loader:
        num_batches += 1

        if num_batches == 1:
            first_batch_X_shape = tuple(batch_X.shape)

    return num_batches, first_batch_X_shape