import numpy as np
from tensorflow.keras.datasets import mnist  # only used for loading data
import matplotlib.pyplot as plt

def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

# Neural Network

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.params = self._initialize_weights()

    def _initialize_weights(self):
        params = {}
        for i in range(1, len(self.layer_sizes)):
            params[f"W{i}"] = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(2. / self.layer_sizes[i-1])
            params[f"b{i}"] = np.zeros((1, self.layer_sizes[i]))
        return params

    def forward(self, X):
        cache = {"A0": X}
        L = len(self.layer_sizes) - 1

        for i in range(1, L):
            Z = np.dot(cache[f"A{i-1}"], self.params[f"W{i}"]) + self.params[f"b{i}"]
            A = relu(Z)
            cache[f"Z{i}"], cache[f"A{i}"] = Z, A

        # output layer (softmax)
        ZL = np.dot(cache[f"A{L-1}"], self.params[f"W{L}"]) + self.params[f"b{L}"]
        AL = softmax(ZL)
        cache[f"Z{L}"], cache[f"A{L}"] = ZL, AL
        return AL, cache

    def backward(self, y_true, cache):
        grads = {}
        L = len(self.layer_sizes) - 1
        m = y_true.shape[0]
        AL = cache[f"A{L}"]

        dZL = AL - y_true
        grads[f"dW{L}"] = np.dot(cache[f"A{L-1}"].T, dZL) / m
        grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True) / m

        for i in reversed(range(1, L)):
            dA = np.dot(dZL, self.params[f"W{i+1}"].T)
            dZ = dA * relu_derivative(cache[f"Z{i}"])
            grads[f"dW{i}"] = np.dot(cache[f"A{i-1}"].T, dZ) / m
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
            dZL = dZ

        return grads

    def update_params(self, grads):
        for key in self.params.keys():
            if key.startswith('W') or key.startswith('b'):
                layer = key[1:]
                self.params[key] -= self.learning_rate * grads[f"d{key}"]

    def train(self, X, y, epochs=20, batch_size=64):
        losses = []
        for epoch in range(epochs):
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                AL, cache = self.forward(X_batch)
                loss = cross_entropy_loss(y_batch, AL)
                grads = self.backward(y_batch, cache)
                self.update_params(grads)

            losses.append(loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        return losses

    def predict(self, X):
        AL, _ = self.forward(X)
        return np.argmax(AL, axis=1)

# Training and Evaluation

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train.reshape(X_train.shape[0], -1)/255.0, X_test.reshape(X_test.shape[0], -1)/255.0

    y_train_oh = one_hot_encode(y_train)
    y_test_oh = one_hot_encode(y_test)

    nn = NeuralNetwork([784, 128, 64, 10], learning_rate=0.01)
    losses = nn.train(X_train, y_train_oh, epochs=20, batch_size=64)

    # Evaluate
    preds = nn.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # Plot loss
    plt.plot(losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
