# Napišite skriptu koja ´ce uˇcitati izgra ¯ denu mrežu iz zadatka 1. Nadalje, skripta
# treba uˇcitati sliku test.png sa diska. Dodajte u skriptu kod koji ´ce prilagoditi sliku za mrežu,
# klasificirati sliku pomo´cu izgra ¯ dene mreže te ispisati rezultat u terminal. Promijenite sliku
# pomo´cu nekog grafiˇckog alata (npr. pomo´cuWindows Paint-a nacrtajte broj 2) i ponovo pokrenite
# skriptu. Komentirajte dobivene rezultate za razliˇcite napisane znamenke.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

saved_model = tf.keras.models.load_model('Model/')

# Load the image and convert it to grayscale
image = Image.open('test1.png').convert('L')

# Resize the image to 28x28 pixels
image = image.resize((28, 28))

# Convert the image to a numpy array and normalize pixel values to be between 0 and 1
image_array = np.array(image) / 255.0

# Invert the image (black pixels become white and white pixels become black)
image_array = 1 - image_array

# Reshape the array to a 4D tensor with shape (1, 28, 28, 1) and make the prediction
#reshaped_image=image_array.reshape((1, 28, 28, 1))
image_reshaped = image_array.reshape(-1, 784).astype("float32") / 255
prediction = saved_model.predict(image_reshaped)
max_prediction=np.argmax(prediction)
# Print the predicted label
print(f"The predicted label is {prediction}")


plt.imshow(image_array)
plt.title(f'Stvarni broj:1, predikcija:{max_prediction}')
plt.show()
