import numpy as np
import matplotlib.pyplot as plt

# rand_nums = [np.random.random() for i in range(100)]

# plt.hist(rand_nums, bins=np.linspace(0, 1, num=10))
# plt.show()

w = np.random.normal(0.0, 1, 3).T

# print(w)
# print(w.T)

f = np.zeros((5, 3))
f[0, :] = w
print(f)