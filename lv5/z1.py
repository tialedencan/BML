import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import matplotlib.colors as mcolors
from sklearn . linear_model import LogisticRegression

from sklearn . metrics import accuracy_score
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay, classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# a) Prikažite podatke za ucenje u x1−x2 ravnini matplotlib biblioteke pri ˇcemu podatke obojite
# s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
# marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
# cmap kojima je mogu´ce definirati boju svake klase.

#print(X_train)
#print(y_train)
#print(len(y_train))

fig, ax = plt.subplots()

scatter = ax.scatter(X_train[:,0], X_train[:,1], c=y_train, s=12,cmap = mcolors.ListedColormap(["purple", "green"]))

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc='upper right', title="Classes")
ax.add_artist(legend1)

#test data
scatter = ax.scatter(X_test[:,0], X_test[:,1], c=y_test, s=12, cmap = mcolors.ListedColormap(["black", "blue"]), marker="x")

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc='lower left', title="Classes-test")
ax.add_artist(legend1)

plt.show()

# b) Izgradite model logistiˇcke regresije pomo´cu scikit-learn biblioteke na temelju skupa podataka
# za uˇcenje.

# inicijalizacija i ucenje modela logisticke regresije
LogRegression_model = LogisticRegression()
LogRegression_model.fit( X_train , y_train )

# predikcija na skupu podataka za testiranje
y_test_p = LogRegression_model.predict( X_test )

# c) Pronadite u atributima izgradenog modela parametre modela. Prikažite granicu odluke
# nauˇcenog modela u ravnini x1 −x2 zajedno s podacima za uˇcenje. Napomena: granica
# odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.

print("Parametri logistice regresije")
print("theta_0: " + str(LogRegression_model.intercept_[0]))
print("theta_1: " + str(LogRegression_model.coef_[0][0]))
print("theta_2: " + str(LogRegression_model.coef_[0][1]))

xp = np.array([X_train[:,1].min(), X_train[:,1].max()])
yp1 = -LogRegression_model.coef_[0][0]/LogRegression_model.coef_[0][1] * xp[0] - LogRegression_model.intercept_[0]/LogRegression_model.coef_[0][1]
yp2 = -LogRegression_model.coef_[0][0]/LogRegression_model.coef_[0][1] * xp[1] - LogRegression_model.intercept_[0]/LogRegression_model.coef_[0][1]
yp = np.array([yp1,yp2])
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, s=12,cmap = mcolors.ListedColormap(["purple", "green"]))
plt.plot(xp,yp,'r')
plt.show()

#from scipy.special import expit

#x_test_d = np.linspace(4.0,7.0,100)
# predict dummy y_test data based on the logistic model
#y_test_dummy = x_test_d * LogRegression_model.coef_ + LogRegression_model.intercept_
 
#sigmoid = expit(y_test_dummy)

#plt.scatter(X_train[:,0], X_train[:,1], c=y_train, s=12,cmap = mcolors.ListedColormap(["purple", "orange"]))
#plt.plot(X_test,sigmoid.ravel(), c="green", label = "logistic fit")

#print(LogRegression_model.intercept_)
#print(LogRegression_model.coef_)
#print(LogRegression_model.get_params())

# d) Provedite klasifikaciju skupa podataka za testiranje pomoc´u izgrad¯enog modela logisticˇke
# regresije. Izraˇcunajte i prikažite matricu zabune na testnim podacima. Izraˇcunate toˇcnost,
# preciznost i odziv na skupu podataka za testiranje.

# tocnost
print (" Tocnost : " , accuracy_score (y_test , y_test_p ))
# matrica zabune
cm = confusion_matrix ( y_test , y_test_p )
print (" Matrica zabune : " , cm)
disp = ConfusionMatrixDisplay (cm)
disp . plot ()
plt . show ()
# report
print ( classification_report (y_test , y_test_p ))

# e) Prikažite skup za testiranje u ravnini x1−x2. Zelenom bojom oznaˇcite dobro klasificirane
# primjere dok pogrešno klasificirane primjere oznaˇcite crnom bojom.

#print(LogRegression_model.predict_proba(X_test))

# c=y_test_p, cmap = mcolors.ListedColormap(["green","black"])
color=['green' if y == y_test_p[idx] else 'black' for idx, y in enumerate(y_test)]
plt.scatter(X_test[:,0],X_test[:,1],c=color)

plt.show()


#from scipy.special import expit
#sigmoid function y_new = expit (x_new)


# • Točnije, granica odluke je hiperravnina ΘTx=0 (za jednu ulaznu veličinu je skalar,
# za dvije ulazne veličine pravac, za tri ulazne veličine ravnina, ...)