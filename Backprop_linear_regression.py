# Linear regression: f = w * x

import numpy as np
import pandas
import matplotlib.pyplot as plt

# Load data from 'data.csv' file
data = np.asarray(pandas.read_excel("data.xlsx"))
X, Y = data[:, 0], data[:, 1]

# initialising weight with random value
w = np.random.uniform(low=0, high=1)


# model output
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


# Training
learning_rate = 0.00001
n_iters = 1000

for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # calculate gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w}, loss = {l}')

plt.figure()
plt.plot(X, Y, '.b')
plt.plot(X, w * X, '-r')
plt.show()
