import numpy as np

def relu(x):
  return np.maximum(0, x)

def relu_derivative(x):
  return (x > 0).astype(float)

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, hidden_size2=None, learning_rate=0.02, huber_delta=1.0):
    self.lr = learning_rate
    self.huber_delta = huber_delta

    # if second hidden size not provided, mirror the first
    self.h1 = hidden_size
    self.h2 = hidden_size if hidden_size2 is None else hidden_size2

    # He initialization for ReLU layers: N(0, sqrt(2/fan_in))
    self.W1 = np.random.randn(input_size, self.h1) * np.sqrt(2.0 / max(1, input_size))
    self.b1 = np.zeros((1, self.h1))

    self.W2 = np.random.randn(self.h1, self.h2) * np.sqrt(2.0 / max(1, self.h1))
    self.b2 = np.zeros((1, self.h2))

    # Output layer remains linear; use He scaling based on hidden2 size
    self.W3 = np.random.randn(self.h2, 1) * np.sqrt(2.0 / max(1, self.h2))
    self.b3 = np.zeros((1, 1))

  def forward(self, X):
    self.Z1 = np.dot(X, self.W1) + self.b1
    self.A1 = relu(self.Z1)

    self.Z2 = np.dot(self.A1, self.W2) + self.b2
    self.A2 = relu(self.Z2)

    self.Z3 = np.dot(self.A2, self.W3) + self.b3  # linear output
    return self.Z3

  def compute_loss(self, y_true, y_pred):
    # Huber loss is less sensitive to outliers than MSE
    e = y_true - y_pred
    abs_e = np.abs(e)
    delta = self.huber_delta
    quadratic = 0.5 * (e ** 2)
    linear = delta * (abs_e - 0.5 * delta)
    loss = np.where(abs_e <= delta, quadratic, linear)
    return np.mean(loss)
  
  def compute_mae(self, y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

  def backward(self, X, y_true, y_pred):
    m = X.shape[0]
    # Huber gradient wrt predictions
    e = y_pred - y_true
    delta = self.huber_delta
    dZ3 = np.clip(e, -delta, delta) / m  # derivative of Huber wrt y_pred

    # Gradients for output layer
    dW3 = np.dot(self.A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    # Backprop into second hidden layer
    dA2 = np.dot(dZ3, self.W3.T)
    dZ2 = dA2 * relu_derivative(self.Z2)
    dW2 = np.dot(self.A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Backprop into first hidden layer
    dA1 = np.dot(dZ2, self.W2.T)
    dZ1 = dA1 * relu_derivative(self.Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    self.dW1, self.db1 = dW1, db1
    self.dW2, self.db2 = dW2, db2
    self.dW3, self.db3 = dW3, db3

  def update_weights(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
        self.W3 -= self.lr * self.dW3
        self.b3 -= self.lr * self.db3

  def train(self, X, y, epochs=1000, verbose=True):
    for epoch in range(epochs):
      y_pred = self.forward(X)
      self.backward(X, y, y_pred)
      self.update_weights()

      if verbose and epoch % 100 == 0:
        loss = self.compute_loss(y, y_pred)
        print(f"Epoch {epoch}, Loss: {loss}")

  def predict(self, X):
        """Predict goal differential for new data."""
        return self.forward(X)