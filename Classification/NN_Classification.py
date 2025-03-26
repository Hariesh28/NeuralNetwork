import time
import pickle
import matplotlib.pyplot as plt
from activation_functions import *

xp = get_array_module()

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "softmax": (softmax, None),
    "linear": (linear, linear_derivative)
}

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
        dropout_prob: float = 0.0,
        l2_lambda: float = 0.001,
        threshold: float = 0.5,
        early_stopping: bool = False,
        patience: int = 5,
        verbose: int = 0,
        use_gpu: bool = True,
        seed: int = 42
    ):
        self.layer_dims = layer_dims
        self.activations = activations if activations is not None else ["relu"]*(len(layer_dims)-2) + ["softmax"]
        self.epoch = epoch
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.use_gpu = use_gpu
        self.threshold = threshold
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.l2_lambda = l2_lambda
        self.dropout_prob = dropout_prob
        self.cache = {}

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

        if self.layer_dims[-1] == 1:
            cost = -self.xp.mean(
                y_true * self.xp.log(y_pred + epsilon) +
                (1 - y_true) * self.xp.log(1 - y_pred + epsilon)
            )

        else:
            cost = -self.xp.sum(y_true * self.xp.log(y_pred + epsilon)) / m

        l2_cost = 0
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            l2_cost += self.xp.sum(self.parameters[f'W{l}'] ** 2)

        cost += (self.l2_lambda / (2 * m)) * l2_cost
        return cost

    def _exponential_decay(self, current_epoch: int) -> float:
        return self.learning_rate * (self.decay_rate ** (current_epoch / self.decay_step))

    def _get_activation_functions(self, activation_name: str = "sigmoid") -> tuple:

        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation function: {activation_name}")

        return ACTIVATIONS[activation_name]

    def _to_numpy(self, arr):
        return np.array([x.get() if hasattr(x, "get") else x for x in arr], dtype=np.float64)

    def _forward_propagation(self, X:ArrayType, training: bool = True) -> tuple:

        cache = {}
        A = X
        L = len(self.layer_dims) - 1

        cache['A0'] = X

        for l in range(1, L+1):
            Z = self.xp.dot(cache[f'A{l-1}'], self.parameters[f'W{l}']) + self.parameters[f'B{l}']

            activation_func, _ = self._get_activation_functions(self.activations[l-1])
            A = activation_func(Z)

            if training and self.dropout_prob > 0 and l != L:

                # Dropout mask
                D = self.xp.random.rand(*A.shape) > self.dropout_prob
                A = A * D
                A = A / (1.0 - self.dropout_prob) # Scale activations
                cache[f'D{l}'] = D

            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        self.cache = cache
        return cache[f'A{L}']

    def _backward_propagation(self, y:ArrayType, lr: float) -> None:

        m = y.shape[0]
        gradients = {}
        L = len(self.layer_dims) - 1
        AL = self.cache[f'A{L}']

        # For output layer
        if self.activations[-1] == 'softmax':
            dZL = AL - y

        else:
            error = AL - y
            _, derivative_func = self._get_activation_functions(self.activations[-1])
            derivative_activation = derivative_func(self.cache[f'Z{L}'])

            dZL = derivative_activation * error

        gradients[f'dZ{L}'] = dZL

        # For remaining layers
        for l in range(L-1, 0, -1):

            _, derivative_func = self._get_activation_functions(self.activations[l-1])
            dA = self.xp.dot(gradients[f'dZ{l+1}'], self.parameters[f'W{l+1}'].T)

            if derivative_func is None:
                raise ValueError(f"Activation '{self.activations[l-1]}' in hidden layer {l} does not support a derivative. Please choose another activation for hidden layers.")

            # If dropout was applied, scale dA with same mask
            if f'D{l}' in self.cache:
                dA = dA * self.cache[f'D{l}'] / (1.0 - self.dropout_prob)

            dZ = derivative_func(self.cache[f'Z{l}']) * dA

            gradients[f'dZ{l}'] = dZ

        # Update weights
        for l in range(1, L + 1):

            dW = self.xp.dot(self.cache[f'A{l-1}'].T, gradients[f'dZ{l}']) + (self.l2_lambda / m) * self.parameters[f'W{l}']
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

        epoch_list, train_loss_list, train_acc_list, lr_list, epoch_time_list = [], [], [], [], []
        best_loss = float("inf")
        no_improve = 0

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

            start_time = time.time()
            current_lr = self._exponential_decay(epoch)
            lr_list.append(current_lr)

            permutation = self.xp.asarray(self.xp.random.permutation(m), dtype=self.xp.intp)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, m, self.batch_size):

                X_batch = X_shuffled[i: i+self.batch_size]
                y_batch = y_shuffled[i: i+self.batch_size]

                self._forward_propagation(X_batch, training=True)
                self._backward_propagation(y_batch, lr=current_lr)

            y_train_pred = self._forward_propagation(X, training=True)
            train_loss = self._cost(y, y_train_pred)
            train_loss_list.append(train_loss)

            if self.layer_dims[-1] == 1:
                predictions = (y_train_pred >= self.threshold).astype(int)
                predictions = predictions.flatten()
                true_labels = y.flatten().astype(int)
            else:
                predictions = self.xp.argmax(y_train_pred, axis=1)
                true_labels = self.xp.argmax(y, axis=1)

            train_accuracy = self.xp.mean(predictions == true_labels)
            train_acc_list.append(train_accuracy)

            epoch_list.append(epoch)
            epoch_duration = time.time() - start_time
            epoch_time_list.append(epoch_duration)

            if self.verbose:
                print(f"Epoch {epoch}/{self.epoch} - Loss: {train_loss:.4f} - Acc: {train_accuracy:.4f} "
                      f"- LR: {current_lr:.6f} "
                      f"- Time: {epoch_duration:.2f}s")

            # Early stopping check.
            if self.early_stopping:
                if train_loss < best_loss:
                    best_loss = train_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break

        self.data = {
        "epochs": np.array(epoch_list, dtype=np.float64),
        "train_loss": self._to_numpy(train_loss_list),
        "train_accuracy": self._to_numpy(train_acc_list),
        "learning_rate": self._to_numpy(lr_list),
        "epoch_duration": np.array(epoch_time_list, dtype=np.float64)
    }

    def predict(self, X:ArrayType) ->ArrayType:

        X = self.xp.asarray(X)

        if X.shape[1] != self.layer_dims[0]:
            raise ValueError(f"Expected input with {self.layer_dims[0]} features, but got {X.shape[1]}.")

        probabilities = self._forward_propagation(X, training=False)

        if self.layer_dims[-1] == 1:
            predictions = (probabilities >= self.threshold).astype(int)
        else:
            predictions = self.xp.argmax(probabilities, axis=1)

        return predictions if self.xp is np else cp.asnumpy(predictions)

    def predict_proba(self, X: ArrayType) -> ArrayType:

        X = self.xp.asarray(X)

        if X.shape[1] != self.layer_dims[0]:
            raise ValueError(f"Expected input with {self.layer_dims[0]} features, but got {X.shape[1]}.")

        probabilities = self._forward_propagation(X, training=False)
        return probabilities if self.xp is np else cp.asnumpy(probabilities)

    def get_training_data(self) -> dict:
        return self.data

    def save(self, filename: str = "model_state.pkl") -> None:
        """Save the model state to a file."""

        state = {k: v for k, v in self.__dict__.items() if k not in ["xp", "cache", "data"]}
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filename: str):
        """Load the model state from a file and reinitialize unpicklable attributes."""

        with open(filename, "rb") as f:
            state = pickle.load(f)

        # Create an instance without calling __init__
        obj = cls.__new__(cls)
        obj.__dict__.update(state)

        # Reinitialize the xp attribute based on use_gpu flag
        if obj.use_gpu:
            try:
                if cp.cuda.is_available():
                    obj.xp = cp
                else:
                    obj.xp = np
            except Exception:
                obj.xp = np
        else:
            obj.xp = np

        # Update the activation functions to use the same backend
        set_array_module(obj.xp)
        return obj


    def plot_training_metrics(self):

        # Ensure the training data is available
        if not hasattr(self, 'data'):
            raise ValueError("Model has not been trained yet. No data available for plotting.")

        epoch_list = self.data["epochs"]
        train_loss_list = self.data["train_loss"]
        train_acc_list = self.data["train_accuracy"]
        lr_list = self.data["learning_rate"]
        epoch_time_list = self.data["epoch_duration"]

        # Plotting the training metrics
        plt.figure(figsize=(12, 8))

        # Training Loss
        plt.subplot(2, 2, 1)
        plt.plot(epoch_list, train_loss_list, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss over Epochs")

        # Training Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(epoch_list, train_acc_list, marker='o', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Training Accuracy")
        plt.title("Training Accuracy over Epochs")

        # Learning Rate
        plt.subplot(2, 2, 3)
        plt.plot(epoch_list, lr_list, marker='o', color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Decay")

        # Epoch Duration
        plt.subplot(2, 2, 4)
        plt.plot(epoch_list, epoch_time_list, marker='o', color='brown')
        plt.xlabel("Epoch")
        plt.ylabel("Epoch Time (s)")
        plt.title("Epoch Duration")

        plt.tight_layout()
        plt.show()
