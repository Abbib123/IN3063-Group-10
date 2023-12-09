import numpy as np
from tensorflow.keras.datasets import mnist


def load_and_preprocess_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten the images and normalize
    X_train_flattened = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test_flattened = X_test.reshape(X_test.shape[0], -1) / 255.0

    # One-hot encode the labels
    y_train_encoded = np.eye(10)[y_train]
    y_test_encoded = np.eye(10)[y_test]

    return X_train_flattened, y_train_encoded, X_test_flattened, y_test_encoded

# Sigmoid Activation Layer
class SigmoidLayer:
    def forward(self, x):
        self.cache = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dA):
        x = self.cache
        sigmoid_x = 1 / (1 + np.exp(-x))
        return dA * sigmoid_x * (1 - sigmoid_x)

# ReLU Activation Layer
class ReLULayer:
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dA):
        x = self.cache
        return dA * (x > 0)

class SoftmaxLayer:
    def forward(self, x):
        # Numerically stable softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache = softmax_output
        return softmax_output

    def backward(self, dA, y):
        # dA is the gradient of the loss with respect to the softmax output
        softmax_output = self.cache
        m = dA.shape[0]
        dz = softmax_output - y
        dz /= m  # Normalize by the number of samples
        return dz

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.sigmoid = SigmoidLayer()
        self.relu = ReLULayer()
        self.softmax = SoftmaxLayer()

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu.forward(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax.forward(self.z2)  # Use softmax activation in the output layer
        return self.a2

    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu.backward(self.z1)
        dW1 = np.dot(x.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def compute_loss(self, predictions, y):
        epsilon = 1e-9  # Small constant to prevent log(0)
        loss = -np.sum(y * np.log(predictions + epsilon)) / y.shape[0]
        return loss
    
    
    def train(self, x, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward propagation
            predictions = self.forward(x)
            
            # Compute loss (categorical cross-entropy)
            loss = self.compute_loss(predictions, y)

            # Backpropagation
            self.backward(x, y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    

X_train, y_train, X_test, y_test = load_and_preprocess_mnist()


# Create and train the neural network
input_size = 784
hidden_size = 128
output_size = 10  # Set to the number of classes in your problem
nn = TwoLayerNN(input_size, hidden_size, output_size)
nn.train(X_train, y_train, learning_rate=0.1, epochs=1000)

# Predictions on test set
test_predictions = nn.forward(X_test)
test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {test_accuracy:.4f}")