import cupy as cp
import numpy as np
from typing import Union

ArrayType = Union[np.ndarray, cp.ndarray]

# Select module. By default, it uses GPU if available.
def get_array_module(use_gpu: bool = True):
    if use_gpu:
        try:
            return cp if cp.cuda.is_available() else np
        except Exception:
            return np
    else:
        return np

xp = get_array_module()

def set_array_module(module):
    """
    Set the array module to use in all activation functions.
    This allows you to override the default behavior (e.g., force CPU).
    """
    global xp
    xp = module

def sigmoid(x: ArrayType) -> ArrayType:
    return 1 / (1 + xp.exp(-x))

def sigmoid_derivative(x: ArrayType) -> ArrayType:
    return x * (1 - x)

def relu(z: ArrayType) -> ArrayType:
    return xp.maximum(0, z)

def relu_derivative(z: ArrayType) -> ArrayType:
    return xp.where(z > 0, 1, 0)

def leaky_relu(z: ArrayType, alpha: float = 0.01) -> ArrayType:
    return xp.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z: ArrayType, alpha: float = 0.01) -> ArrayType:
    return xp.where(z > 0, 1, alpha)

def tanh(z: ArrayType) -> ArrayType:
    return xp.tanh(z)

def tanh_derivative(z: ArrayType) -> ArrayType:
    return 1 - xp.tanh(z)**2

def softmax(z: ArrayType) -> ArrayType:
    # With numerical stability
    exp_z = xp.exp(z - xp.max(z, axis=1, keepdims=True))
    return exp_z / xp.sum(exp_z, axis=1, keepdims=True)

def linear(z: ArrayType) -> ArrayType:
    return z

def linear_derivative(z: ArrayType) -> ArrayType:
    return xp.ones_like(z)
