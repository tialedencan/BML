import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

# Importing csv module
import csv


with open("data.csv", 'r') as x:
    sample_data = list(csv.reader(x, delimiter=","))

sample_data = np.array(sample_data)

# a)
print(len(sample_data)-1)


# b)
s,height,weight = sample_data.T.tolist()
s=np.delete(s,0)

height=np.delete(height,0) #height
weight=np.delete(weight,0) #weight

h = height.astype(float)
w = weight.astype(float)
plt.scatter(h, w, color = 'hotpink')


# c)

fheight = [h[i] for i in range(len(height)) if i%50==0]
fweight = [w[i] for i in range(len(weight)) if i%50==0]


plt.scatter(fheight, fweight, color = '#88c999')

plt.xlabel ('Height')
plt.ylabel ('Weight')
plt.title ( 'Scatter plot')

plt.show()

# d) 

h = height.astype(float)
print(f"Min height: {min(h)}")
print(f"Max height: {max(h)}")
print(f"Arithmetic value: {mean(h)}")

# e) 

s = s.astype(float)
ind = (s == 1.0) #male
mh = [h[i] for i in range(len(h)) if ind[i]]
fh = [h[i] for i in range(len(h)) if ind[i] == False]

print(f"Min men height: {min(mh)}")
print(f"Max men height: {max(mh)}")
print(f"Arithmetic value men: {mean(mh)}")

print(f"Min women height: {min(fh)}")
print(f"Max women height: {max(fh)}")
print(f"Arithmetic value women: {mean(fh)}")

