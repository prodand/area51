import numpy as np
import matplotlib.pyplot as plt

res = np.array([np.arange(1, 10).reshape((3, 3)) for i in range(1, 30)])

plt.ion()

line, = plt.plot([], [], 'bo')
# print(type(lines[0]))
# line = lines[0]
for i in range(1, 101):
    line.set_xdata(np.append(line.get_xdata(), i))
    line.set_ydata(np.append(line.get_ydata(), i * i))
    plt.draw()
