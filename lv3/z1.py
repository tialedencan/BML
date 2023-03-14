import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv( 'data_C02_emission.csv')

# a)

print(data.count) # isto je i sa print(data.describe)
print(data.size)
print(data.info()) #vjv. su htjeli tu metodu

# provjera koliko je izostalih vrijednosti po svakom stupcu DataFramea
print(data.isnull().sum())

# brisanje dupliciranih redova
data . drop_duplicates ()
# kada se obrisu pojedini redovi potrebno je resetirati indekse retka
data = data.reset_index ( drop = True )

col_obj = data[['Make','Model','Vehicle Class','Transmission','Fuel Type']]
print(col_obj)

for col in ['Make','Model','Vehicle Class','Transmission','Fuel Type']:
    data[col] = data[col].astype('category')

print(data.info())

# b)

highest_3_fuel_consumption_city = data[data['Fuel Consumption City (L/100km)'] <= data['Fuel Consumption City (L/100km)'].max()].head(3)
print(highest_3_fuel_consumption_city[['Make','Model','Fuel Consumption City (L/100km)']])

# c) Koliko vozila ima veliˇcinu motora izme ¯ du 2.5 i 3.5 L? Kolika je prosjeˇcna C02 emisija plinova za ova vozila?

num_of_vehicles =  data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print(len(num_of_vehicles))
mean_CO2_emission = num_of_vehicles['CO2 Emissions (g/km)'].mean()
print(mean_CO2_emission)
#print(mean_CO2_emission.sum()/len(num_of_vehicles))

# d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je prosjeˇcna emisija C02
#plinova automobila proizvod¯acˇa Audi koji imaju 4 cilindara?

audi_vehicles = data[data['Make'] == 'Audi']
print(len(audi_vehicles))
mean_audi_4_cylinders = audi_vehicles[audi_vehicles['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean()
print(mean_audi_4_cylinders)

# e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjeˇcna emisija C02 plinova s obzirom na
# broj cilindara?

grouped = data.groupby('Cylinders')

for group,cylinder_group in grouped:
    print(f"Mean for {group} cylinders: {cylinder_group['CO2 Emissions (g/km)'].mean()}")
    



# f) Kolika je prosjeˇcna gradska potrošnja u sluˇcaju vozila koja koriste dizel, a kolika za vozila
# koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?

grouped_fuel_types = data.groupby('Fuel Type')
for group,group_data in grouped_fuel_types:
    print(group)

diesel_consumption_mean = (data[data['Fuel Type'] == 'D'])['Fuel Consumption City (L/100km)'].mean()
print(diesel_consumption_mean)
diesel_consumption_median = (data[data['Fuel Type'] == 'D'])['Fuel Consumption City (L/100km)'].median()
print(diesel_consumption_median)

petrol_consumption_mean = (data[data['Fuel Type'] == 'X'])['Fuel Consumption City (L/100km)'].mean()
petrol_consumption_median = (data[data['Fuel Type'] == 'X'])['Fuel Consumption City (L/100km)'].median()
print(petrol_consumption_mean)
print(petrol_consumption_median) 

# g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najve´cu gradsku potrošnju goriva?

vehicles_wih_diesel_motor_4_cylinders = data[(data['Fuel Type'] == 'D')  &  (data['Cylinders'] == 4)]
max_city_consumption_D4 = vehicles_wih_diesel_motor_4_cylinders['Fuel Consumption City (L/100km)'].max()
result = vehicles_wih_diesel_motor_4_cylinders[vehicles_wih_diesel_motor_4_cylinders['Fuel Consumption City (L/100km)'] == max_city_consumption_D4]
print(result)

# h) Koliko ima vozila ima ruˇcni tip mjenjaˇca (bez obzira na broj brzina)?
counter = 0
for vehicle in data['Transmission']:
    if (vehicle.__contains__('M')) &  ~(vehicle.__contains__('AM')):
        counter += 1

print(counter)

print(len(data[data['Transmission'].str.contains('M') & ~(data['Transmission'].str.contains('AM'))]))


# i) Izracˇunajte korelaciju izmed¯u numericˇkih velicˇina. Komentirajte dobiveni rezultat

print(data['Engine Size (L)'].corr(data['Fuel Consumption City (L/100km)']))
print(data['Engine Size (L)'].corr(data['Fuel Consumption Hwy (L/100km)']))
print(data['Engine Size (L)'].corr(data['Fuel Consumption Comb (L/100km)']))
print(data['Engine Size (L)'].corr(data['CO2 Emissions (g/km)']))

# linearni porast, nepotpuna korelacija

