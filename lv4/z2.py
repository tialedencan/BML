# Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoriˇcku
# varijable „Fuel Type“ kao ulaznu veliˇcinu. Pri tome koristite 1-od-K kodiranje kategoriˇckih
# veliˇcina. Radi jednostavnosti nemojte skalirati ulazne veliˇcine. Komentirajte dobivene rezultate.
# Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
# vozila radi?

from sklearn import datasets
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import MinMaxScaler
import pandas as pd
import sklearn . linear_model as lm
from sklearn . metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import matplotlib . pyplot as plt
from sklearn . preprocessing import OneHotEncoder
from sklearn.metrics import max_error
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

data['Fuel Type'] = data['Fuel Type'].astype('category')
data['Fuel Type'] = data['Fuel Type'].cat.codes
x = data[['Engine Size (L)','Cylinders','Fuel Type','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
y = data[['CO2 Emissions (g/km)']]

ohe = OneHotEncoder()

X_encoded = ohe.fit_transform(x[['Fuel Type']]).toarray()
#print(x['Fuel Type'])
#print(X_encoded)
x['Fuel Type']=X_encoded
print(x)

X_train , X_test , y_train , y_test = train_test_split (x, y, test_size = 0.2, random_state = 1)

linearModel = lm.LinearRegression()
linearModel.fit( X_train, y_train)  

y_test_p = linearModel.predict(X_test)

error = max_error(y_test,y_test_p)
print(error)
max_error_id = np.argmax(error)

model = data.iloc[max_error_id]
print(model)
