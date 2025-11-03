import numpy as np

def relu(x):
  return np.maximum(0, x)

def relu_derivative(x):
  return (x > 0).astype(float)

def softmax(logits):
  # Stable softmax
  z = logits - np.max(logits, axis=1, keepdims=True)
  exp_z = np.exp(z)
  return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NeuralNetworkWinner:
  def __init__(self, input_size, hidden_size, hidden_size2=None, learning_rate=0.01,
               beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    self.lr = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.weight_decay = weight_decay

    self.h1 = hidden_size
    self.h2 = hidden_size if hidden_size2 is None else hidden_size2

    # He init for ReLU layers
    self.W1 = np.random.randn(input_size, self.h1) * np.sqrt(2.0 / max(1, input_size))
    self.b1 = np.zeros((1, self.h1))

    self.W2 = np.random.randn(self.h1, self.h2) * np.sqrt(2.0 / max(1, self.h1))
    self.b2 = np.zeros((1, self.h2))

    # Output: 3 classes [home_win, draw, away_win]
    self.W3 = np.random.randn(self.h2, 3) * np.sqrt(1.0 / max(1, self.h2))
    self.b3 = np.zeros((1, 3))

    # Adam state
    self.t = 0
    self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
    self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
    self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
    self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
    self.mW3 = np.zeros_like(self.W3); self.vW3 = np.zeros_like(self.W3)
    self.mb3 = np.zeros_like(self.b3); self.vb3 = np.zeros_like(self.b3)

  def forward(self, X):
    self.Z1 = np.dot(X, self.W1) + self.b1
    self.A1 = relu(self.Z1)

    self.Z2 = np.dot(self.A1, self.W2) + self.b2
    self.A2 = relu(self.Z2)

    self.logits = np.dot(self.A2, self.W3) + self.b3
    self.probs = softmax(self.logits)
    return self.probs

  def compute_loss(self, y_true_onehot, y_pred_probs, eps=1e-12):
    # Cross-entropy loss for one-hot labels
    p = np.clip(y_pred_probs, eps, 1.0)
    loss = -np.sum(y_true_onehot * np.log(p), axis=1)
    return np.mean(loss)

  def compute_accuracy(self, y_true_onehot, y_pred_probs):
    return np.mean(np.argmax(y_true_onehot, axis=1) == np.argmax(y_pred_probs, axis=1))

  def backward(self, X, y_true_onehot, y_pred_probs):
    m = X.shape[0]
    # Softmax + cross-entropy gradient: dL/dlogits = (probs - y)/m
    dZ3 = (y_pred_probs - y_true_onehot) / m
    dW3 = np.dot(self.A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    if self.weight_decay > 0.0:
      dW3 += self.weight_decay * self.W3

    dA2 = np.dot(dZ3, self.W3.T)
    dZ2 = dA2 * relu_derivative(self.Z2)
    dW2 = np.dot(self.A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    if self.weight_decay > 0.0:
      dW2 += self.weight_decay * self.W2

    dA1 = np.dot(dZ2, self.W2.T)
    dZ1 = dA1 * relu_derivative(self.Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    if self.weight_decay > 0.0:
      dW1 += self.weight_decay * self.W1

    self.dW1, self.db1 = dW1, db1
    self.dW2, self.db2 = dW2, db2
    self.dW3, self.db3 = dW3, db3

  def update_weights(self):
        self.t += 1
        b1, b2, eps, lr = self.beta1, self.beta2, self.eps, self.lr

        # W1, b1
        self.mW1 = b1 * self.mW1 + (1 - b1) * self.dW1
        self.vW1 = b2 * self.vW1 + (1 - b2) * (self.dW1 ** 2)
        mW1_hat = self.mW1 / (1 - b1 ** self.t)
        vW1_hat = self.vW1 / (1 - b2 ** self.t)
        self.W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + eps)

        self.mb1 = b1 * self.mb1 + (1 - b1) * self.db1
        self.vb1 = b2 * self.vb1 + (1 - b2) * (self.db1 ** 2)
        mb1_hat = self.mb1 / (1 - b1 ** self.t)
        vb1_hat = self.vb1 / (1 - b2 ** self.t)
        self.b1 -= lr * mb1_hat / (np.sqrt(vb1_hat) + eps)

        # W2, b2
        self.mW2 = b1 * self.mW2 + (1 - b1) * self.dW2
        self.vW2 = b2 * self.vW2 + (1 - b2) * (self.dW2 ** 2)
        mW2_hat = self.mW2 / (1 - b1 ** self.t)
        vW2_hat = self.vW2 / (1 - b2 ** self.t)
        self.W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + eps)

        self.mb2 = b1 * self.mb2 + (1 - b1) * self.db2
        self.vb2 = b2 * self.vb2 + (1 - b2) * (self.db2 ** 2)
        mb2_hat = self.mb2 / (1 - b1 ** self.t)
        vb2_hat = self.vb2 / (1 - b2 ** self.t)
        self.b2 -= lr * mb2_hat / (np.sqrt(vb2_hat) + eps)

        # W3, b3
        self.mW3 = b1 * self.mW3 + (1 - b1) * self.dW3
        self.vW3 = b2 * self.vW3 + (1 - b2) * (self.dW3 ** 2)
        mW3_hat = self.mW3 / (1 - b1 ** self.t)
        vW3_hat = self.vW3 / (1 - b2 ** self.t)
        self.W3 -= lr * mW3_hat / (np.sqrt(vW3_hat) + eps)

        self.mb3 = b1 * self.mb3 + (1 - b1) * self.db3
        self.vb3 = b2 * self.vb3 + (1 - b2) * (self.db3 ** 2)
        mb3_hat = self.mb3 / (1 - b1 ** self.t)
        vb3_hat = self.vb3 / (1 - b2 ** self.t)
        self.b3 -= lr * mb3_hat / (np.sqrt(vb3_hat) + eps)

  def train(self, X, y_onehot, epochs=1000, verbose=True):
    for epoch in range(epochs):
      probs = self.forward(X)
      self.backward(X, y_onehot, probs)
      self.update_weights()

      if verbose and epoch % 100 == 0:
        loss = self.compute_loss(y_onehot, probs)
        acc = self.compute_accuracy(y_onehot, probs)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}")

  def predict_proba(self, X):
        return self.forward(X)

  def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


