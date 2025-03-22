import cupy as cp

def sigmoid(x: cp.ndarray) -> cp.ndarray:
    return 1 / (1 + cp.exp(-x))

def sigmoid_derivative(x: cp.ndarray) -> cp.ndarray:
    return x * (1 - x)

def relu(z: cp.ndarray) -> cp.ndarray:
    return cp.maximum(0, z)

def relu_derivative(z: cp.ndarray) -> cp.ndarray:
    return cp.where(z > 0, 1, 0)

def leaky_relu(z: cp.ndarray, alpha: float = 0.01) -> cp.ndarray:
    return cp.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z: cp.ndarray, alpha: float = 0.01) -> cp.ndarray:
    return cp.where(z > 0, 1, alpha)

def tanh(z: cp.ndarray) -> cp.ndarray:
    return cp.tanh(z)

def tanh_derivative(z: cp.ndarray) -> cp.ndarray:
    return 1 - cp.tanh(z)**2

def softmax(z: cp.ndarray) -> cp.ndarray:
    # With numerical stability
    exp_z = cp.exp(z - cp.max(z, axis=1, keepdims=True))
    return exp_z / cp.sum(exp_z, axis=1, keepdims=True)

def linear(z: cp.ndarray) -> cp.ndarray:
    return z

def linear_derivative(z: cp.ndarray) -> cp.ndarray:
    return cp.ones_like(z)
