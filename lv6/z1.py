import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty='none') 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#Z1
# Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
# Podaci o korisnicima su spol, dob i procijenjena pla´ca. Razmatra se binarni klasifikacijski
# problem gdje su dob i procijenjena pla´ca ulazne veliˇcine, dok je kupovina (0 ili 1) izlazna
# veliˇcina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija
# plot_decision_region [1]. Podaci su podijeljeni na skup za uˇcenje i skup za testiranje modela
# u omjeru 80%-20% te su standardizirani. Izgra ¯ den je model logistiˇcke regresije te je izraˇcunata
# njegova toˇcnost na skupu podataka za uˇcenje i skupu podataka za testiranje. Potrebno je:
# 1. Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Izraˇcunajte toˇcnost
# klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje. Usporedite
# dobivene rezultate s rezultatima logistiˇcke regresije. Što primje´cujete vezano uz dobivenu
# granicu odluke KNN modela?
# 2. Kako izgleda granica odluke kada je K =1 i kada je K = 100?

from sklearn . neighbors import KNeighborsClassifier


# inicijalizacija i ucenje KNN modela
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit( X_train_n, y_train )

# predikcija 
y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print('KNN:')
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

# granica odluke pomocu KNN-a
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()

#za K=1
KNN_model_1 = KNeighborsClassifier(n_neighbors = 1)
KNN_model_1.fit( X_train_n, y_train )

# predikcija 
y_train_p_KNN1 = KNN_model_1.predict(X_train_n)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN1))))
plt.tight_layout()

#za K=100
KNN_model_100 = KNeighborsClassifier(n_neighbors = 100)
KNN_model_100.fit( X_train_n, y_train )

# predikcija 
y_train_p_KNN100 = KNN_model_100.predict(X_train_n)

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN100))))
plt.tight_layout()

plt.show()

# za K = 1 overfit, za K = 100 je underfit

#Z2
# Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
# algoritma KNN za podatke iz Zadatka 1.

from sklearn.model_selection import cross_val_score

model_KNN = KNeighborsClassifier()
model_KNN.fit( X_train_n, y_train )
y_train_p_KNN2 = model_KNN.predict(X_train_n)
y_test_p_KNN2 = model_KNN.predict(X_test_n)
scores = cross_val_score (model_KNN, X_train_n , y_train , cv = 10)
print ( scores )
param_grid = {'n_neighbors':range(1,100)}
grid_KNN = GridSearchCV (model_KNN, param_grid , cv =5, scoring ='accuracy',n_jobs =-1)
grid_KNN . fit ( X_train_n , y_train )
print ( grid_KNN.best_params_ )
print ( grid_KNN.best_score_ )

#Z3
# Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
# te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
# ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje?
# Mijenjajte tip kernela koji se koristi. Što primje´cujete?

from sklearn import svm

# inicijalizacija i ucenje SVM modela
SVM_model = svm.SVC( kernel ='rbf', gamma = 1, C= 0.1)
SVM_model.fit( X_train_n, y_train )

# predikcija na skupu podataka za testiranje
y_test_p_SVM = SVM_model.predict( X_test_n )
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

y_train_p_SVM = SVM_model.predict(X_train_n)

# granica odluke pomocu SVM-a
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

#Z4
#Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
#algoritma SVM za problem iz Zadatka 1.

from sklearn . svm import SVC
from sklearn . preprocessing import StandardScaler
from sklearn . pipeline import Pipeline
from sklearn . pipeline import make_pipeline
from sklearn . model_selection import GridSearchCV

model_SVM = svm.SVC( kernel ='rbf')
model_SVM.fit( X_train_n, y_train )

scores_SVM = cross_val_score (model_SVM, X_train_n , y_train , cv = 10)
print ( scores_SVM )

param_grid = {'C': [5, 1, 10 , 20, 100 , 100 ],
'gamma': [10 , 1, 0.1, 0.01 ]}

svm_gscv = GridSearchCV (model_SVM, param_grid , cv =5, scoring ='accuracy',n_jobs =-1)

svm_gscv.fit ( X_train_n , y_train )
print ( svm_gscv.best_params_ )
print ( svm_gscv.best_score_ )
#print ( svm_gscv . cv_results_ )