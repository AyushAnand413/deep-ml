import math


#Cosine Annealing uses the shape of half a cosine wave to gradually decrease the learning rate
class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        # Initialize initial_lr, T_max, and min_lr
        self.initial_lr = initial_lr
        self.T_max = T_max #max no of epoch in CosineAnnealing
        self.min_lr = min_lr

    def get_lr(self, epoch):
        # Calculate and return the learning rate for the given epoch, rounded to 4 decimal places
        
        epoch = max(0,min(epoch,self.T_max)) #this keeps epoch bw 0 to T_max
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos((epoch / self.T_max) * math.pi))

        return round(lr,4)