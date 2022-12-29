# Solving a least squares problem using numpy
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])  # These are points on the graphs

A = np.vstack([x, np.ones(len(x))]).T  # vertical stack followed by a transposition. ( y = Ap, A=[[x 1]] and p =[[w],
# [b]] )
print("The data matrix: \n", A)

w_np, b_np = np.linalg.lstsq(A, y, rcond=None)[0]  # this built in function uses the least square method
print("The obtained result: ", w_np, b_np)

import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, w_np * x + b_np, 'r', label='Fitted line')
plt.legend()
plt.show()

# Solving using gradient descent

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])  # These are points on the graphs


def fit_lse(x, y):
    lr = 0.1
    w = 1
    b = 0
    for i in range(10):
        grad_w = 0
        grad_b = 0

        for _x, _y in zip(x, y):
            grad_w += _x * (_x * w + b - _y)
            grad_b += (_x * w + b - _y)

        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b


w, b = fit_lse(x, y)
print('obtained result', w, b)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, w_np * x + b_np, 'r', label='Fitted line')
plt.title("Solved using numpy function")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, w * x + b, 'r', label='Fitted line')
plt.title("Solved using gradient descent")
plt.legend()
plt.show()


