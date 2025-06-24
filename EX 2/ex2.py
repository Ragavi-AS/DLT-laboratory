import numpy as np

X = np.array([[1.5, 2.0, 1.2, 3.2, 2.7, 3.0, 0.5, 1.0]])
Y = np.array([[1, 1, 1, 0, 0, 0, 1, 1]])

X = (X - np.mean(X)) / np.std(X)

w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 1000
m = X.shape[1]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for i in range(epochs):
    Z = w * X + b
    A = sigmoid(Z)
    cost = -np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)) / m
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    w -= learning_rate * dw
    b -= learning_rate * db
    if i % 100 == 0:
        print(f"Epoch {i}: Cost = {cost:.4f}")

Z = w * X + b
A = sigmoid(Z)
predictions = (A > 0.5).astype(int)
accuracy = 100 - np.mean(np.abs(predictions - Y)) * 100

print("\nFinal Weights:", w)
print("Final Bias:", b)
print("Predictions:", predictions)
print("Actual Labels:", Y)
print(f"Accuracy: {accuracy:.2f}%")
