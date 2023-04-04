import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay, classification_report, accuracy_score

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])
    plt.show()

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()
y=y[:,0]
# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


# a) Pomo´cu stupˇcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu
# pingvina) u skupu podataka za uˇcenje i skupu podataka za testiranje. Koristite numpy
# funkciju unique.

fig = plt.subplots(figsize =(12, 8))
# specie = df['species'].unique()

# # set width of bar
barWidth = 0.25
# Set position of bar on X axis
br1 = np.arange(3)
br2 = [x + barWidth for x in br1]
#br3 = [x + barWidth for x in br2]
 
number_of_examples_train_0 = len(y_train[y_train == 0])
number_of_examples_train_1 = len(y_train[y_train == 1])
number_of_examples_train_2 = len(y_train[y_train == 2])
number_of_examples_train = [number_of_examples_train_0,number_of_examples_train_1,number_of_examples_train_2]
plt.bar(br1, number_of_examples_train,width=barWidth)
number_of_examples_test_0 = len(y_test[y_test == 0])
number_of_examples_test_1 = len(y_test[y_test == 1])
number_of_examples_test_2 = len(y_test[y_test == 2])
number_of_examples_test = [number_of_examples_test_0,number_of_examples_test_1,number_of_examples_test_2]
plt.bar(br2, number_of_examples_test,width=barWidth, color = 'green')

# Adding Xticks
plt.xlabel('Species', fontweight ='bold', fontsize = 15)
plt.ylabel('Number of examples', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(3)],['Adelie', 'Chinstrap', 'Gentoo'])

plt.show()


# b) Izgradite model logistiˇcke regresije pomo´cu scikit-learn biblioteke na temelju skupa podataka
# za uˇcenje.

# inicijalizacija i ucenje modela logisticke regresije
log_reg_model = LogisticRegression()
log_reg_model.fit( X_train , y_train )

# predikcija na skupu podataka za testiranje
y_test_p = log_reg_model.predict( X_test )

# c) Pronad¯ite u atributima izgrad¯enog modela parametre modela. Koja je razlika u odnosu na
# binarni klasifikacijski problem iz prvog zadatka?

print(log_reg_model.intercept_)
print(log_reg_model.coef_)
print("Parametri logistice regresije dobiveni pomocu ugradene funkcije u sklearn")
print("theta_0: " + str(log_reg_model.intercept_[0]))
print("theta_1: " + str(log_reg_model.coef_[0][0]))
print("theta_2: " + str(log_reg_model.coef_[0][1]))

# Višeklasna lasifikcija, time bi se moglo gledati kao više zasebnih binarnih klasifikacija

# d) Pozovite funkciju plot_decision_region pri ˇcemu joj predajte podatke za uˇcenje i
# izgrad¯eni model logisticˇke regresije. Kako komentirate dobivene rezultate?

plot_decision_regions(X_train, y_train, classifier = log_reg_model)

# e) Provedite klasifikaciju skupa podataka za testiranje pomoc´u izgrad¯enog modela logisticˇke
# regresije. Izraˇcunajte i prikažite matricu zabune na testnim podacima. Izraˇcunajte toˇcnost.
# Pomo´cu classification_report funkcije izraˇcunajte vrijednost ˇcetiri glavne metrike
# na skupu podataka za testiranje.

# matrica zabune
cm = confusion_matrix ( y_test , y_test_p )
print (" Matrica zabune : " , cm)
disp = ConfusionMatrixDisplay ( confusion_matrix (y_test , y_test_p ))
disp . plot ()
plt . show ()
# report
print ( classification_report (y_test , y_test_p ))
print (" Tocnost : " , accuracy_score (y_test , y_test_p ))

# f) Dodajte u model još ulaznih veliˇcina. Što se doga ¯ da s rezultatima klasifikacije na skupu
# podataka za testiranje?

input_variables = ['bill_length_mm',
                    'flipper_length_mm',
                    'bill_depth_mm',
                    'body_mass_g']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# inicijalizacija i ucenje modela logisticke regresije
log_reg = LogisticRegression()
log_reg.fit( X_train , y_train )

# predikcija na skupu podataka za testiranje
y_test_p = log_reg.predict( X_test )

# matrica zabune
cm = confusion_matrix ( y_test , y_test_p )
print (" Matrica zabune : " , cm)
disp = ConfusionMatrixDisplay ( confusion_matrix (y_test , y_test_p ))
disp . plot ()
plt . show ()
# report
print ( classification_report (y_test , y_test_p ))
print (" Tocnost : " , accuracy_score (y_test , y_test_p ))

# dodavanjem ulaznog parametra  body_mass_g točnost modela se smanji, a dodavanjem još jednog parametra se poveća