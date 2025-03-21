import numpy as np
from activation_functions import *
from sklearn.metrics import accuracy_score

class NeuralNetwork:

    def __init__(
        self,
        layer_dims: list[int],
        activations: list[str] = None,
        epoch: int = 20,
        learning_rate: float = 0.01,
        seed: int = 42
    ):
        self.layer_dims = layer_dims
        self.activations = activations
        self.epoch = epoch
        self.learning_rate = learning_rate
        np.random.seed(seed)

    def _initialize_parameters(self):

        parameters = {}

        L = len(self.layer_dims) - 1 # Exclude input layer

        for i in range(1, L + 1):

            parameters[f'W{i}'] = np.random.randn(self.layer_dims[i-1], self.layer_dims[i]) * 0.1
            parameters[f'B{i}'] = np.zeros((1, self.layer_dims[i]))

        self.parameters = parameters

    @staticmethod
    def _cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:

        errors = y_true - y_pred
        return 0.5 * np.sum(errors ** 2)

    def _forward_propagation(self, X: np.ndarray) -> tuple:

        cache = {}
        A = X
        L = len(self.layer_dims) - 1

        cache['A0'] = X

        for l in range(1, L+1):
            Z = np.dot(A, self.parameters[f'W{l}']) + self.parameters[f'B{l}']

            if self.activations[l-1] == 'sigmoid':
                A = sigmoid(Z)

            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        self.cache = cache
        return cache[f'A{L}']

    def _backward_propagation(self, y: np.ndarray) -> None:

        gradients = {}
        L = len(self.layer_dims) - 1

        # For output layer
        AL = self.cache[f'A{L}']

        error = y - AL
        derivative_activation = sigmoid_derivative(AL)

        dZL = derivative_activation * error

        gradients[f'A{L}'] = dZL

        for l in range(L-1, 0, -1):

            AL = self.cache[f'A{l}']
            prev_grads = np.dot(gradients[f'A{l+1}'], self.parameters[f'W{l+1}'].T)
            dZL = sigmoid_derivative(AL) * prev_grads

            gradients[f'A{l}'] = dZL

        # Update weights
        for l in range(1, L + 1):

            delta_weight = self.learning_rate * np.dot(self.cache[f'A{l-1}'].T, gradients[f'A{l}'])
            self.parameters[f'W{l}'] += delta_weight

            self.parameters[f'B{l}'] += self.learning_rate * np.sum(gradients[f'A{l}'], axis=0, keepdims=True)


    def train(self, X: np.ndarray, y: np.ndarray) -> None:

        self._initialize_parameters()

        for epoch in range(1, self.epoch + 1):

            y_pred = self._forward_propagation(X)
            cost = self._cost(y, y_pred)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{self.epoch}, Cost: {cost}")

            self._backward_propagation(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward_propagation(X)

if __name__ == '__main__':
    X = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    y = np.array([
        [1,0],
        [0,1],
        [1,1],
        [0,0]
    ])

    model = NeuralNetwork(layer_dims=[3, 2, 2], activations=['sigmoid', 'sigmoid'], epoch=10_000, learning_rate=0.1)
    model.train(X, y)


    y_pred = model.predict(X)
    threshold = 0.5

    print('Predictions:', y_pred)
    y_pred = (y_pred > threshold).astype(int)
    print('Predictions:', y_pred)
    print(f'Accuracy: {accuracy_score(y_true=y, y_pred=y_pred)*100}')
