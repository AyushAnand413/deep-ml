import jax
import jax.numpy as jnp

def df_dx(x):
    """Derivative of f(x) = x**3 + 2x at float x, via jax.grad. Returns float."""
    def f(x):
        return x**3 + 2*x
    
    gradient_fn = jax.grad(f)
    return float(gradient_fn(float(x)))
