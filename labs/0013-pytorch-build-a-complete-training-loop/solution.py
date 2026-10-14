import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    """
    - model = NeuralNetwork(): Creates the neural network.
    - model.train(): Switches the model to training mode.
    - model.eval(): Switches the model to evaluation/testing mode.
    - model.train() does not update weights by itself.

    What changes in each mode?
    - Dropout:
    - model.train() → Randomly turns off some neurons.
    - model.eval() → Dropout is disabled.
    - Batch Normalization:
    - model.train() → Uses current batch statistics and updates running statistics.
    - model.eval() → Uses stored running statistics.
    - If the model has neither Dropout nor Batch Normalization, switching modes generally does not change its output.
    """
    history = []
    
    # Your code here
    #optimizer
    optimizer = optim.Adam(model.parameters(),lr=lr)
    #loss fn
    criterion = nn.CrossEntropyLoss()

    n = len(X_train)
    for epoch in range(epochs):
        model.train() #dropout some neurons to prevent overfiiting
        indices = torch.randperm(n)
        total_loss=0.0
        for start in range(0,n,batch_size):
            batch_indices = indices[start:start + batch_size]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            #clean accumulated grads
            optimizer.zero_grad()
            #forward pass
            pred = model(X_batch)
            #losss
            loss = criterion(pred,y_batch)
            #backward prop
            loss.backward()
            #update pramas
            #here we can use optstep
            optimizer.step()

            ## Add the batch loss multiplied by the number of samples in that batch
            total_loss += loss.item() * len(X_batch)
        
        # Calculate the average training loss across all training samples
        train_loss = total_loss/n;

        model.eval()
        #now we need to do validation
        #as we arent training so no need to stiore grad so
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred,y_val).item()
            
            predictions = val_pred.argmax(dim=1)  # Get the predicted class with the highest score
            val_accuracy = (predictions == y_val).float().mean().item()  # Calculate the fraction of correct predictions
    
        history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
        })
    return history
