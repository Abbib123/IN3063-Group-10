import numpy as np
import tensorflow as tf

# Define activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(Z):
    # Numerical stability fix: shift values inside the exp function
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e_Z / np.sum(e_Z, axis=0, keepdims=True)

def softmax_backward(Y, softmax_output):
    # Y is one-hot encoded true labels, softmax_output is the output of the softmax function
    dZ = softmax_output - Y
    return dZ

# Initialize network parameters
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

# Forward pass
def forward_pass(X, parameters, keep_prob=0.8, training=True):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)  # or sigmoid(Z1), depending on your choice
    
    # Implementing Inverted Dropout
    if training and keep_prob < 1.0:
        D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob
        A1 = np.multiply(A1, D1)
        A1 /= keep_prob
    else:
        D1 = None
    
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "D1": D1}
    return A2, cache

# Backward pass
def backward_pass(X, Y, cache, parameters, keep_prob=0.8):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    D1 = cache["D1"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = softmax_backward(y_train_one_hot, A2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(W2.T, dZ2)
    
    if D1 is not None:
        dA1 = np.multiply(dA1, D1)  # Apply the dropout mask
        dA1 /= keep_prob           # Scale the values back up
    
    dZ1 = np.multiply(dA1, relu_derivative(cache["Z1"]))  # or sigmoid_derivative for sigmoid
    
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters 
    
def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2 + 1e-15)) / m
    cost = np.squeeze(cost)  # ensures the cost is the dimension we expect.
    return cost

def convert_to_one_hot(labels, num_classes):
    one_hot = np.eye(num_classes)[labels].T
    return one_hot

def evaluate(X, Y, parameters,training=False):
    A2, _ = forward_pass(X.T, parameters, training=training)
    predictions = np.argmax(A2, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy


# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Define network architecture and training parameters
input_size = 784  # 28*28
hidden_size = 128
output_size = 10
learning_rate = 0.1
num_iterations = 1000  # Example 
   
# Initialize parameters
parameters = initialize_parameters(input_size, hidden_size, output_size)

# Convert y_train and y_test to one-hot encoding
num_classes = 10
y_train_one_hot = convert_to_one_hot(y_train, num_classes)
y_test_one_hot = convert_to_one_hot(y_test, num_classes)


keep_prob = 0.8

# Training loop
for i in range(num_iterations):
    # Forward pass with dropout
    A2, cache = forward_pass(X_train.T, parameters, keep_prob=keep_prob, training=True)

    # Compute cost
    cost = compute_cost(A2, y_train_one_hot)

    # Backward pass with dropout
    grads = backward_pass(X_train.T, y_train_one_hot, cache, parameters, keep_prob=keep_prob)

    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)

    # Print cost every 100 iterations (or as needed)
    if i % 100 == 0:
        print(f"Iteration {i}: Cost {cost}")
        

# Evaluate the network on the test set
test_accuracy = evaluate(X_test, y_test_one_hot, parameters)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")