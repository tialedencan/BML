import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_2.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


unique_colors = np.unique(img_array_aprox,axis=0).shape[0]
print("Number of unique colors: ", unique_colors) 

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

def recreate_image_binary(codebook, labels, w, h):
    for ind,l in enumerate(labels):
        if l==1:
            labels[ind]=0
        else:
            labels[ind]=1
    return codebook[labels].reshape(w, h, -1)


# inicijalizacija algoritma K srednjih vrijednosti
km = KMeans ( n_clusters = 2, init ='k-means++', n_init = 5, random_state = 0)
# pokretanje grupiranja primjera
km. fit (img_array_aprox)
# dodijeljivanje grupe svakom primjeru
labels = km. predict (img_array)
print(labels)
print(km.cluster_centers_)

plt.figure()
plt.title("Rezultantna slika")
plt.imshow(recreate_image(km.cluster_centers_,labels,w,h))
plt.tight_layout()

plt.figure()
plt.title("Rezultantna binarna slika")
plt.imshow(recreate_image_binary(km.cluster_centers_,labels,w,h))
plt.tight_layout()

plt.show()


Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
 kmeans = KMeans(n_clusters=num_clusters)
 kmeans.fit(img_array_aprox)
 Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()
