import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ReLU and softmax functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8))

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target.reshape(-1, 1)  # Labels

# One-hot encode the labels
encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # He initialization for ReLU activation function
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)
    
    # Forward pass
    def forward(self, X):
        # Input to hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = relu(self.hidden_layer_input)

        # Hidden layer to output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = softmax(self.output_layer_input)
        return self.output
    
    # Backpropagation
    def backward(self, X, y, learning_rate):
        # Output error and delta
        output_error = self.output - y  # Difference between predicted and actual
        output_delta = output_error

        # Hidden layer error and delta
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * relu_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output -= self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden -= X.T.dot(hidden_delta) * learning_rate
        self.bias_output -= np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0) * learning_rate
    
    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 100 == 0:
                loss = cross_entropy_loss(y, self.output)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

    # Predict using the trained network
    def predict(self, X):
        self.forward(X)
        return np.argmax(self.output, axis=1)

# Initialize the neural network (4 input features, 10 hidden neurons, 3 output classes)
nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3)

# Train the neural network
nn.train(X_train, y_train, epochs=1000, learning_rate=0.001)

# Test the network
predictions = nn.predict(X_test)

# Convert one-hot encoded test labels back to class indices
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate the accuracy
accuracy = np.mean(predictions == y_test_labels)
print(f'\nTest Accuracy: {accuracy * 100:.2f}%')

# Print some predictions
print("\nSample Predictions:")
for i in range(5):
    print(f'Predicted: {predictions[i]}, Actual: {y_test_labels[i]}')
