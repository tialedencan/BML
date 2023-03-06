import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,3,1])
y= np.array([1,2,2,1,1])
plt.plot(x, y, 'g', linewidth = 2, marker =".", markersize = 6)
plt.axis ([0,4,0,4])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Shape')
plt.show()
