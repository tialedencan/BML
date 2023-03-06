import numpy as np
import matplotlib.pyplot as plt

zero = np.zeros([50,50], int)
one = np.ones([50,50], int)


first_row = np.hstack((zero,one))
secund_row = np.hstack((one,zero))

image = np.vstack((first_row,secund_row))

plt.imshow(image, cmap="gray")
plt.show()