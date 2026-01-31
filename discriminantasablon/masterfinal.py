import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Date
date = pd.read_csv("dateIN/ProiectB.csv")
date_aplicare = pd.read_csv("dateIN/ProiectB_apply.csv")
x = date.drop(columns=["VULNERAB", "avariere"])
y = date["VULNERAB"]
x_aplicare = date_aplicare.drop(columns=["VULNERAB"], errors='ignore')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# ANALIZA DISCRIMINANTĂ
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Calcul scoruri discriminante model liniar
scoruri = lda.transform(x_test)
print("Scoruri discriminante:\n", scoruri)

# Trasare plot instanțe în axe discriminante
plt.figure(figsize=(10, 6))
y_colors = pd.Series(y_test).astype('category').cat.codes
plt.scatter(scoruri[:, 0], np.zeros(len(scoruri)), c=y_colors, cmap="viridis", edgecolors='k', s=50)
plt.title("Instanțe în axe discriminante")
plt.xlabel("Axa discriminantă")
plt.yticks([])
plt.colorbar(label="Clasa")
plt.show()

# Trasare plot distribuții în axele discriminante
plt.figure(figsize=(10, 6))
lda_df = pd.DataFrame({"Scoruri": scoruri[:, 0], "Clasa": y_test.values})
sns.histplot(lda_df, x="Scoruri", hue="Clasa", kde=True, bins=30, palette="viridis", alpha=0.6)
plt.title("Distribuția scorurilor discriminante")
plt.xlabel("Scoruri discriminante")
plt.ylabel("Frecvență")
plt.show()

# Predicția în setul de testare model liniar
y_pred_lda = lda.predict(x_test)
print("\nPredicția în setul de testare model liniar:\n",
      pd.DataFrame({"Real": y_test.values, "Predicție": y_pred_lda}))

# Evaluare model liniar pe setul de testare
cm_lda = confusion_matrix(y_test, y_pred_lda)
acc_global_lda = accuracy_score(y_test, y_pred_lda)
acc_medie_lda = np.mean(cm_lda.diagonal() / cm_lda.sum(axis=1))

print("\nMatricea de confuzie LDA:\n", cm_lda)
print(f"Acuratețe globală LDA: {acc_global_lda:.4f}")
print(f"Acuratețe medie LDA: {acc_medie_lda:.4f}")

# Predicția în setul de aplicare model liniar
y_pred_aplicare_lda = lda.predict(x_aplicare)
print("\nPredicția în setul de aplicare model liniar:\n",
      pd.DataFrame({"Predicție LDA": y_pred_aplicare_lda}))

# Predicția în setul de testare model bayesian
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)
print("\nPredicția în setul de testare model bayesian:\n",
      pd.DataFrame({"Real": y_test.values, "Predicție": y_pred_nb}))

# Evaluare model bayesian
cm_nb = confusion_matrix(y_test, y_pred_nb)
acc_global_nb = accuracy_score(y_test, y_pred_nb)
acc_medie_nb = np.mean(cm_nb.diagonal() / cm_nb.sum(axis=1))

print("\nMatricea de confuzie Naive Bayes:\n", cm_nb)
print(f"Acuratețe globală Naive Bayes: {acc_global_nb:.4f}")
print(f"Acuratețe medie Naive Bayes: {acc_medie_nb:.4f}")

# Predicția în setul de aplicare model bayesian
y_pred_aplicare_nb = nb.predict(x_aplicare)
print("\nPredicția în setul de aplicare model bayesian:\n",
      pd.DataFrame({"Predicție Naive Bayes": y_pred_aplicare_nb}))