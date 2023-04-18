# from tensorflow import keras
# from keras import layers
# model = keras.Sequential()
# model.add( layers.Input( shape =(2, )))
# model.add( layers.Dense (3, activation =" relu "))
# model.add( layers.Dense (1, activation =" sigmoid "))
# model.summary()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()
#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)

# 1. Upoznajte se s ucitanim podacima. Koliko primjera sadrži skup za ucenje, a koliko skup za
# testiranje? Kako su skalirani ulazni podaci tj. slike? Kako je kodirana izlazne velicina?
print(f'Veličina i dimenzija primjera za ucenje: {x_train.shape}') 
print(f'Veličina i dimenzija primjera za testiranje: {x_test.shape}')
print(f'Izlaz skupa za testiranje: {y_test}')

#60000 primjera u skupu za treniranje i 10000 u skupu za testiranje, dimenzija slika je 28x28 
# izlazne velicine su array s oznakom koju znamenku predstavlja ulazna vrijednost, duljine br. ulaznih pr.

# 2. Pomocu matplotlib biblioteke prikažite jednu sliku iz skupa podataka za ucenje te ispišite
# njezinu oznaku u terminal.

# Preprocess the data (these are NumPy arrays)
x_train_reshaped = x_train.reshape(60000, 784).astype("float32") / 255
x_test_reshaped = x_test.reshape(10000, 784).astype("float32") / 255

plt.imshow(x_train[6],cmap=plt.cm.binary)
plt.show()
print(y_train[6])

# 3. Pomo´cu klase Sequential izgradite mrežu prikazanu na slici 8.5. Pomo´cu metode
# .summary ispišite informacije o mreži u terminal.

model = tf.keras.models.Sequential()
model.add( tf.keras.layers.Input(shape=(784,)))
model.add (tf.keras.layers.Dense(100,activation=tf.nn.relu))
model.add (tf.keras.layers.Dense(50,activation=tf.nn.relu))
model.add (tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.summary()

# 4. Pomo´cu metode .compile podesite proces treniranja mreže.

oh_encoder=OneHotEncoder()
y_train_encoded = oh_encoder.fit_transform(np.reshape(y_train,(-1,1))).toarray() #OHE treba 2d array, pa se koristi reshape (-1,1), tj (n,1),
y_test_encoded = oh_encoder.transform(np.reshape(y_test,(-1,1))).toarray() #-1 oznacava da sam otkrije koliko, mora toarray() 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Pokrenite uˇcenje mreže (samostalno definirajte broj epoha i veliˇcinu serije). Pratite tijek
# uˇcenja u terminalu.

history = model.fit(x_train_reshaped,y_train_encoded,epochs=4,batch_size=50)
#print(history.history)

# 6. Izvršite evaluaciju mreže na testnom skupu podataka pomo´cu metode .evaluate.

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test_reshaped, y_test_encoded, batch_size=128)
print("test loss, test acc:", results)

# 7. Izraˇcunajte predikciju mreže za skup podataka za testiranje. Pomo´cu scikit-learn biblioteke
# prikažite matricu zabune za skup podataka za testiranje.

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for test samples")
predictions = model.predict(x_test_reshaped)#2d array, u svakom retku vjerojatnost pripadanja primjera svakoj klasi
y_predictions=np.argmax(predictions,axis=1)
#print("test loss, test acc:", predictions)
print("predictions shape:", predictions.shape)
cm=confusion_matrix(y_true=y_test, y_pred=y_predictions)
#print (" Matrica zabune : " , cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# 8. Pohranite model na tvrdi disk.
model.save('Model/')

