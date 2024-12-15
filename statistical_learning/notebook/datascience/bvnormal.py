import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

N = 1000
r = 0.0
sigma = np.array([[1, r], [r, 1]])
B = np.linalg.cholesky(sigma)
x = B @ randn(2, N)

plt.scatter([x[0,:]], [x[1,:]], alpha=0.4, s=4)
plt.show()