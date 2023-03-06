# Napišite program koji ce kreirati sliku koja sadrži cetiri kvadrata crne odnosno
# bijele boje (vidi primjer slike 2.4 ispod). 

import numpy as np
import matplotlib.pyplot as plt

zero = np.zeros([50,50], int)
one = np.ones([50,50], int)
#print(zero)

first_row = np.hstack((zero,one))
secund_row = np.hstack((one,zero))

image = np.vstack((first_row,secund_row))

plt.imshow(image, cmap="gray")
plt.show()