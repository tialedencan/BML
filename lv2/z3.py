import numpy as np
import matplotlib . pyplot as plt


img = plt.imread("road.jpg")
img = img [:,:,0].copy()
print(img.shape)
print(img.dtype)
plt.figure()

# a)
plt.imshow(img , cmap ="gray", alpha=0.2)


# b)
rotated_img = np.rot90(img)
plt.imshow(rotated_img)


# c)
height, width= img.shape

start_col = width // 4
end_col = width //2
sliced_img = img[:, start_col:end_col]

plt.imshow(sliced_img)

# 2.c)
sliced_img= img[:,110:320]
plt.imshow(sliced_img)


# d)
mirrored_img = np.fliplr(img)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax1.set_title("Original Image")
ax2.imshow(mirrored_img)
ax2.set_title("Mirrored Image")


plt.show()