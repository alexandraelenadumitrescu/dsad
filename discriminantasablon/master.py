import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# citire date
date = pd.read_csv("dateIN/ProiectB.csv")
date_aplicare=pd.read_csv("dateIN/ProiectB_apply.csv")
x = date.drop(columns=["VULNERAB","avariere"])
y = date["VULNERAB"]
x_aplicare = date_aplicare.drop(columns=["VULNERAB"], errors='ignore')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
lda.fit(x_train, y_train)

# Calcul scoruri discriminante model liniar
scoruri = lda.transform(x_test)
print("Scoruri discriminante\n", scoruri)
y_colors = pd.Series(y_test).astype('category').cat.codes

# Trasare plot instante in axe discriminante
plt.figure(figsize=(10,8))
plt.scatter(scoruri[:, 0], [0]*len(scoruri), c=y_colors, cmap="viridis", edgecolors='k', s=50)
#plt.scatter(scoruri[:, 0], [0]*len(scoruri), c=y_test, cmap="viridis", edgecolors='k', s=50)
plt.title("Instante in axe discriminante")
plt.xlabel("Axa discriminanta")
plt.yticks([])
plt.colorbar(label="Clasa")
plt.show()

# Trasare plot distributii in axele discriminante
lda_df = pd.DataFrame({"Scoruri discriminante": scoruri[:,0], "Clasa":y_test})
plt.figure(figsize=(10,8))
sb.histplot(lda_df, x="Scoruri discriminante", hue="Clasa", kde=True, bins=30, palette="viridis", alpha=0.6)
plt.title("Distributia scorurilor discriminante in axa LDA")
plt.xlabel("Scoruri Discriminante (Axa 1)")
plt.ylabel("Frecventa")
plt.show()

# Predictia in setul de testare model liniar
y_pred = lda.predict(x_test)
df_pred = pd.DataFrame({"Real": y_test.values, "Predictie":y_pred})
print("Predictia in setul de testare model liniar:\n", df_pred)

# Evaluare model liniar pe setul de testare (matricea de confuzie + indicatori de acuretete)
#matrice confuzie
matrice_confuzie = confusion_matrix(y_test, y_pred)
print("Matricea de confuzie:\n", matrice_confuzie)

#acuratete globala
acuratete_globala = accuracy_score(y_test, y_pred)
print("Acuratetea globala: ", acuratete_globala)

#acuratete medie
acuratete_per_clasa = matrice_confuzie.diagonal() / matrice_confuzie.sum(axis=1)
acuratete_medie = np.mean(acuratete_per_clasa)
print("Acuratete medie: ", acuratete_medie)

# Predictia in setul de aplicare model liniar
y_pred_aplicare = lda.predict(x_train)
df_pred_aplicare = pd.DataFrame({"Real": y_train.values, "Predictie": y_pred_aplicare})
print("Predictia in setul de aplicare model liniar:\n", df_pred_aplicare)

# Predictia in setul de testare model bayesian
#model bayesian
model_b = GaussianNB()
model_b.fit(x_train, y_train)
#predictia
predictie_b_test = model_b.predict(x_test)
print("Predictia in setul de testare model bayesian",predictie_b_test)

# Evaluare model bayesian (matricea de confuzie + indicatori de acuretete)
# Matricea de confuzie
matrice_confuzie_b = confusion_matrix(y_test, predictie_b_test)
print("Matricea de confuzie pentru modelul bayesian:\n", matrice_confuzie_b)

# Acuratete globala
acuratete_globala_b = accuracy_score(y_test, predictie_b_test)
print("Acuratetea globala pentru modelul bayesian: ", acuratete_globala_b)

# Acuratete medie
acuratete_per_clasa_b = matrice_confuzie_b.diagonal() / matrice_confuzie_b.sum(axis=1)
acuratete_medie_b = np.mean(acuratete_per_clasa_b)
print("Acuratete medie pentru modelul bayesian: ", acuratete_medie_b)

# # Predictia in setul de aplicare model bayesian
# predictie_b_aplicare = model_b.predict(x_train)
# print("Predictie Bayes", predictie_b_aplicare)



#ALTERNATIVVVV

# Predictia in setul de aplicare model bayesian daca ni se da set de aplicare
# Predictia in setul de aplicare model liniar
# --- CORRECTED APPLICATION SECTION ---

# 1. Prediction for Linear Model (LDA) on the NEW application set
y_pred_aplicare_lda = lda.predict(x_aplicare)
df_lda_app = pd.DataFrame({
    "ID": date_aplicare.index, # Useful to keep track of which row is which
    "Predictie_LDA": y_pred_aplicare_lda
})
print("LDA Application Predictions:\n", df_lda_app.head())

# 2. Prediction for Bayesian Model (NB) on the NEW application set
y_pred_aplicare_nb = model_b.predict(x_aplicare)
df_nb_app = pd.DataFrame({
    "ID": date_aplicare.index,
    "Predictie_Bayes": y_pred_aplicare_nb
})
print("Bayes Application Predictions:\n", df_nb_app.head())

