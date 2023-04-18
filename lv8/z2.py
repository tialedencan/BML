# Napišite skriptu koja ce ucitati izgradenu mrežu iz zadatka 1 i MNIST skup
# podataka. Pomocu matplotlib biblioteke potrebno je prikazati nekoliko loše klasificiranih slika iz
# skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvid¯enu
# mrežom.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

saved_model = tf.keras.models.load_model('Model/')
print(saved_model.summary())

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()
x_test_reshaped = x_test.reshape(10000, 784).astype("float32") / 255

predictions = saved_model.predict(x_test_reshaped)#2d array, u svakom retku vjerojatnost pripadanja primjera svakoj klasi
y_predictions = np.argmax(predictions,axis=1)

wrong_classified_y_value = y_predictions[y_predictions != y_test]
wrong_classified_y_correct_value = y_test[y_predictions != y_test]
wrong_prediction_images=x_test[y_predictions != y_test]
for i in range(5):
    plt.imshow(wrong_prediction_images[i],cmap=plt.cm.binary)
    plt.title(f"Stvarna:{wrong_classified_y_correct_value[i]}, Predviđena:{wrong_classified_y_value[i]}")
    plt.show()