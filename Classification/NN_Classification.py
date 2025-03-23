from activation_functions import *

xp = get_array_module()

class NeuralNetworkClassifier:

    def __init__(
        self,
        layer_dims: list[int],
        activations: list[str] = None,
        epoch: int = 20,
        batch_size: int = 64,
        max_norm: float = 5.0,
        learning_rate: float = 0.01,
        decay_rate: float = 0.96,
        decay_step: int = 10,
        verbose: int = 0,
        use_gpu: bool = True,
        seed: int = 42
    ):
        self.layer_dims = layer_dims
        self.activations = activations
        self.epoch = epoch
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.use_gpu = use_gpu
        self.verbose = verbose

        if self.use_gpu:
            try:
                if cp.cuda.is_available():
                    self.xp = cp

                    # Enable memory pooling to reduce allocation overhead.
                    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
                    cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

                    if self.verbose:
                        print("Using GPU:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
                else:
                    if self.verbose:
                        print("GPU not available. Falling back to CPU.")

                    self.xp = np

            except Exception as e:

                if self.verbose:
                    print(f"Error while setting up GPU: {e}. Falling back to CPU.")

                self.xp = np
        else:
            self.xp = np

            if self.verbose:
                print("Using CPU (forced by user).")

        set_array_module(self.xp)
        self.xp.random.seed(seed)

    def _initialize_parameters(self):

        parameters = {}

        L = len(self.layer_dims) - 1 # Exclude input layer

        for i in range(1, L + 1):
            parameters[f'W{i}'] = self.xp.random.randn(self.layer_dims[i-1], self.layer_dims[i]) * self.xp.sqrt(2. / self.layer_dims[i-1])
            parameters[f'B{i}'] = self.xp.zeros((1, self.layer_dims[i]))

        self.parameters = parameters

    def _cost(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        m = y_true.shape[0]
        epsilon = 1e-8
        return -self.xp.sum(y_true * self.xp.log(y_pred + epsilon)) / m

    def _exponential_decay(self, current_epoch: int) -> float:
        return self.learning_rate * (self.decay_rate ** (current_epoch / self.decay_step))

    def _get_activation_functions(self, activation_name: str = 'sigmoid') -> tuple:

        match activation_name:
            case 'sigmoid':
                return sigmoid, sigmoid_derivative
            case 'relu':
                return relu, relu_derivative
            case 'leaky_relu':
                return leaky_relu, leaky_relu_derivative
            case 'tanh':
                return tanh, tanh_derivative
            case 'softmax':
                return softmax, None
            case 'linear':
                return linear, linear_derivative
            case _:
                raise ValueError(f"Unsupported activation function: {activation_name}")

    def _forward_propagation(self, X:ArrayType) -> tuple:

        cache = {}
        A = X
        L = len(self.layer_dims) - 1

        cache['A0'] = X

        for l in range(1, L+1):
            Z = self.xp.dot(A, self.parameters[f'W{l}']) + self.parameters[f'B{l}']

            activation_func, _ = self._get_activation_functions(self.activations[l-1])
            A = activation_func(Z)

            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        self.cache = cache
        return cache[f'A{L}']

    def _backward_propagation(self, y:ArrayType, lr: float) -> None:

        gradients = {}
        L = len(self.layer_dims) - 1
        AL = self.cache[f'A{L}']

        # For output layer
        if self.activations[-1] == 'softmax':
            dZL = AL - y

        else:
            error = y - AL
            _, derivative_func = self._get_activation_functions(self.activations[-1])
            derivative_activation = derivative_func(self.cache[f'Z{L}'])

            dZL = derivative_activation * error

        gradients[f'dZ{L}'] = dZL

        # For remaining layers
        for l in range(L-1, 0, -1):

            _, derivative_func = self._get_activation_functions(self.activations[l-1])
            prev_grads = self.xp.dot(gradients[f'dZ{l+1}'], self.parameters[f'W{l+1}'].T)

            if derivative_func is None:
                raise ValueError(f"Activation '{self.activations[l-1]}' in hidden layer {l} does not support a derivative. Please choose another activation for hidden layers.")

            dZL = derivative_func(self.cache[f'Z{l}']) * prev_grads

            gradients[f'dZ{l}'] = dZL

        # Update weights
        for l in range(1, L + 1):

            dW = self.xp.dot(self.cache[f'A{l-1}'].T, gradients[f'dZ{l}'])
            dB = self.xp.sum(gradients[f'dZ{l}'], axis=0, keepdims=True)

            # Apply gradient clipping
            norm_dW = self.xp.linalg.norm(dW)
            if norm_dW > self.max_norm:
                dW = dW * (self.max_norm / norm_dW)

            norm_dB = self.xp.linalg.norm(dB)
            if norm_dB > self.max_norm:
                dB = dB * (self.max_norm / norm_dB)

            self.parameters[f'W{l}'] -= lr * dW
            self.parameters[f'B{l}'] -= lr * dB


    def fit(self, X:ArrayType, y:ArrayType) -> None:

        try:
            if self.xp.cuda.is_available():
                print("Using GPU")
                device = self.xp.cuda.Device(0)
                with device:
                    print("GPU Device Name:", self.xp.cuda.runtime.getDeviceProperties(0)['name'].decode())
            else:
                print("Using CPU")
        except AttributeError:
            print("Using CPU")

        X = self.xp.asarray(X)
        y = self.xp.asarray(y)

        self._initialize_parameters()
        m = X.shape[0]

        for epoch in range(1, self.epoch + 1):

            current_lr = self._exponential_decay(epoch)

            permutation = self.xp.asarray(self.xp.random.permutation(m), dtype=self.xp.intp)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, m, self.batch_size):

                X_batch = X_shuffled[i: i+self.batch_size]
                y_batch = y_shuffled[i: i+self.batch_size]

                y_pred = self._forward_propagation(X_batch)
                cost = self._cost(y_batch, y_pred)

                self._backward_propagation(y_batch, lr=current_lr)

            if epoch % 10 == 0 and self.verbose:
                print(f"Epoch {epoch}/{self.epoch}, Cost: {cost}", end='\r')

        print(' '*100, end='\r')

    def predict(self, X:ArrayType) ->ArrayType:

        X = self.xp.asarray(X)
        probabilities = self._forward_propagation(X)
        predictions = self.xp.argmax(probabilities, axis=1)
        return predictions if self.xp is np else cp.asnumpy(predictions)
