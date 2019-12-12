import numpy as np

image = np.arange(1, 10).reshape((3, 3))

kernels = np.arange(1, 9).reshape((2, 2, 2))

res = np.multiply(image, kernels)
print(res)