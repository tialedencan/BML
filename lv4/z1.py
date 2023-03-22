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
import math
import numpy as np

# a)  Odaberite željene numeriˇcke veliˇcine specificiranjem liste s nazivima stupaca. Podijelite 
# #podatke na skup za uˇcenje i skup za testiranje u omjeru 80%-20%.


data = pd.read_csv('data_C02_emission.csv')

x = data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
y = data[['CO2 Emissions (g/km)']]

x = x.to_numpy()
y = y.to_numpy()

# podijeli skup na podatkovni skup za ucenje i podatkovni skup za testiranje
X_train , X_test , y_train , y_test = train_test_split (x, y, test_size = 0.2, random_state = 1)

# b) Pomo´cu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
# o jednoj numeriˇckoj veliˇcini. Pri tome podatke koji pripadaju skupu za uˇcenje oznaˇcite
# plavom bojom, a podatke koji pripadaju skupu za testiranje oznaˇcite crvenom bojom.


plt.scatter(x = X_train[:,0], y = y_train[:,0], color = 'b', s = 9, label = 'train')
plt.scatter(x = X_test[:,0], y = y_test[:,0], color = 'r', s = 9, label = 'test')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()

plt.figure()
plt.scatter(x = X_train[:,1], y = y_train[:,0], color = 'b', s = 9,label = 'train')
plt.scatter(x = X_test[:,1], y = y_test[:,0], color = 'r', s = 9, label = 'test')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()

plt.figure()
plt.scatter(x = X_train[:,2], y = y_train[:,0], color = 'b', s = 9,label = 'train')
plt.scatter(x = X_test[:,2], y = y_test[:,0], color = 'r', s = 9,label = 'test')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()


# c) Izvršite standardizaciju ulaznih veliˇcina skupa za uˇcenje. Prikažite histogram vrijednosti
# jedne ulazne veliˇcine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
# transformirajte ulazne veliˇcine skupa podataka za testiranje.

# # min - max skaliranje
sc = MinMaxScaler()
X_train_n = sc. fit_transform ( X_train )
#print(X_train_n)


fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
ax[0].hist(X_train[:,0], bins = 20)
ax[0].set_title('Engine Size (L)')
ax[1].hist(X_train_n[:,0], bins = 20)
ax[1].set_title('Engine Size (L) - scaled')
plt.show()

X_test_n = sc. transform ( X_test )

# d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
# povežite ih s izrazom 4.6.


linearModel = lm.LinearRegression()
linearModel.fit( X_train_n , y_train)     

theta0 = linearModel.intercept_
theta = linearModel.coef_
print(theta0)
print(theta)
print(f'y(x) = {theta0[0]} +{theta[0,0]} x1 +{theta[0,1]} x2 {theta[0,2]} x3 {theta[0,3]} x4 +{theta[0,4]} x5 {theta[0,5]} x6')

# e) Izvršite procjenu izlazne veliˇcine na temelju ulaznih veliˇcina skupa za testiranje. Prikažite
# pomoc´u dijagrama raspršenja odnos izmed¯u stvarnih vrijednosti izlazne velicˇine i procjene
# dobivene modelom.

# predikcija izlazne velicine na skupu podataka za testiranje
y_test_p = linearModel.predict( X_test_n )
# print(y_test_p)
# print(X_test_n)
plt.figure()
plt.scatter( y_test, y_test_p, c = 'm')
plt.xlabel('CO2 Emission - real')
plt.ylabel('CO2 Emission - prediction')
# plt.show()


# f) Izvršite vrednovanje modela na naˇcin da izraˇcunate vrijednosti regresijskih metrika na
# skupu podataka za testiranje.


MSE = mean_squared_error(y_test,y_test_p)
print(f'MSE: {MSE}')

RMSE = math.sqrt(MSE)
print(f'RMSE: {RMSE}')

# evaluacija modela na skupu podataka za testiranje pomocu MAE
MAE = mean_absolute_error ( y_test , y_test_p )
print(f'MAE: {MAE}')

MAPE = mean_absolute_percentage_error(y_test, y_test_p)
print(f'MAPE: {MAPE}')

r2 = r2_score(y_test, y_test_p)
print(f'R^2: {r2}')


# g) Što se dogad¯a s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj
# ulaznih veliˇcina?
#['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)'] da ostanu ti stupci

X_train = np.delete(X_train, 5, axis = 1)
X_train = np.delete(X_train, 4, axis = 1)
X_train = np.delete(X_train, 3, axis = 1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

lModel = lm.LinearRegression()
lModel.fit( X_train_scaled, y_train)    

X_test = np.delete(X_test, 5, axis = 1)
X_test = np.delete(X_test, 4, axis = 1)
X_test = np.delete(X_test, 3, axis = 1)

X_test_n = scaler. transform ( X_test )

y_test_p = lModel.predict( X_test_n )

print("Reduced number of input values")

MSE = mean_squared_error(y_test,y_test_p)
print(f'MSE: {MSE}')

RMSE = math.sqrt(MSE)
print(f'RMSE: {RMSE}')

# evaluacija modela na skupu podataka za testiranje pomocu MAE
MAE = mean_absolute_error ( y_test , y_test_p )
print(f'MAE: {MAE}')

MAPE = mean_absolute_percentage_error(y_test, y_test_p)
print(f'MAPE: {MAPE}')

r2 = r2_score(y_test, y_test_p)
print(f'R^2: {r2}')

plt.show()