import numpy as np

res = np.argmax([[2, 1], [0, 4]], axis=None)
val = np.unravel_index(res, (2, 2))
print(val)