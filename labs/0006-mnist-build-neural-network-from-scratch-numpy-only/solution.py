# --- cell 1 ---
import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        # TODO: Initialize weights and biases
        # Hint: Use np.random.randn() with proper scaling   
        self.w1 = np.random.randn(input_size,hidden_size) * np.sqrt(2.0/input_size)
        #it creates weights of shape (784,128) and we use he initilization to keep it small
        self.b1 = np.zeros((1,hidden_size)) # (1,128)
        self.w2 = np.random.randn(hidden_size,output_size) * np.sqrt(2.0/input_size) #(128,10)
        self.b2 = np.zeros((1,output_size)) #(1,10)
    
    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # TODO: Implement forward pass
        # Layer 1: X @ W1 + b1, then ReLU
        # Layer 2: hidden @ W2 + b2, then Softmax
        z1 = X @ self.w1 + self.b1 # layer 1
        h1 = np.maximum(0,z1) # relu
        z2 = h1 @ self.w2 + self.b2 # layer 2
        #softmax :- convert these scores of (N,10) into prob
        z2_shifted = z2 - np.max(z2 , axis=1 , keepdims=True)
        exp_z2 = np.exp(z2_shifted)
        probs = exp_z2 / np.sum(exp_z2, axis=1 , keepdims=True)
        #we also need these values for backprop so we cache it 
        self.cache = {
            'X':X,
            'z1':z1,
            'h1':h1,
            'z2':z2
        }
        return probs
        
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        # TODO: Implement backward pass
        # 1. Compute loss
        # 2. Compute gradient of loss w.r.t. softmax output
        # 3. Backprop through linear layer 2
        # 4. Backprop through ReLU
        # 5. Backprop through linear layer 1
        # 6. Update all parameters using gradients
        N = X.shape[0]
        #probs[np.arange(N), y] selects only the probability assigned to each ex correct digit
        loss = -np.mean(np.log(probs[np.arange(N),y] + 1e-12))
        #output gradient
        dz2 = probs.copy()
        dz2[np.arange(N),y] -= 1
        dz2 /= N  # we do this because our loss is avg loss across the batch so grads also need to be
        
        dw2 = self.cache['h1'].T @ dz2
        db2 = np.sum(dz2 , axis=0, keepdims=True)
        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (self.cache['z1'] > 0)
        
        dw1 = self.cache['X'].T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        
        return loss
    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)