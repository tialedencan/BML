import pandas as pd
import matplotlib.pyplot as plt

data = pd. read_csv('data_C02_emission.csv')

# a)
data ['CO2 Emissions (g/km)'].plot( kind ='hist', bins = 20)
plt.show()


# b)
  
print(data['Fuel Type'].value_counts())

data['Fuel Type'] = data['Fuel Type'].map({'X':0, 'Z': 1, 'D': 2, 'E': 3, 'N':4})

data.plot.scatter (x='Fuel Consumption City (L/100km)' ,
                        y='CO2 Emissions (g/km)' ,
                        c='Fuel Type', cmap ="cool", s=50)
plt.show ()


# c) 

grouped = data.groupby ('Fuel Type')
grouped.boxplot ( column =['Fuel Consumption Hwy (L/100km)'])
data.boxplot ( column = ['CO2 Emissions (g/km)'], by='Fuel Type')
plt.show()


# d)

grouped = data.groupby ('Fuel Type')
result = grouped['Fuel Type']
fuel_group={}

for group, vehicle in result:
    print(f"{group}:{len(vehicle)}")
    fuel_group[f"{group}"] = len(vehicle)

plt.bar(list(fuel_group.keys()),list(fuel_group.values()),color="g", width = 0.4)
plt.xlabel('Fuel type')
plt.ylabel('Number of vehicles')
plt.show()

 # e) 

grouped = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print(grouped)
plt.xlabel('Number of cylinders')
plt.ylabel('CO2 emissions mean')
plt.bar(grouped.keys(), grouped, width=0.6)
plt.show()