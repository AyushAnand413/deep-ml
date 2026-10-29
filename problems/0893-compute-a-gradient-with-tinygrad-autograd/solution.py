from tinygrad import Tensor

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked tensor for x, compute f(x), run backprop, return df/dx as a float
    x = Tensor(x_value, requires_grad= True)#create a tracked Tensor
    y = x * x + 3 * x + 2 #
    y.backward()

    return x.grad.item()