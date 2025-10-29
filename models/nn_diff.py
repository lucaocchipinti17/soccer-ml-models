import numpy as np

def relu(x):
  return np.maximum(0, x)

def relu_derivative(x):
  return (x > 0).astype(float)

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, learning_rate=0.02):
    self.lr = learning_rate
    self.W1 = np.random.randn(input_size, hidden_size) * 0.01
    self.b1 = np.zeros((1, hidden_size))

    self.W2 = np.random.randn(hidden_size, 1) * 0.01
    self.b2 = np.zeros((1, 1))

  def forward(self, X):
    self.Z1 = np.dot(X, self.W1) + self.b1
    self.A1 = relu(self.Z1)

    self.Z2 = np.dot(self.A1, self.W2) + self.b2
    return self.Z2

  def compute_loss(self, y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
  
  def compute_mae(self, y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

  def backward(self, X, y_true, y_pred):
    m = X.shape[0]
    dZ2 = (2/m) * (y_pred - y_true)
    dW2 = np.dot(self.A1.T, dZ2)             
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, self.W2.T)            
    dZ1 = dA1 * relu_derivative(self.Z1) 
    dW1 = np.dot(X.T, dZ1)                  
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    self.dW1, self.db1, self.dW2, self.db2 = dW1, db1, dW2, db2

  def update_weights(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2

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