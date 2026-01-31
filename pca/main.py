import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb

date = pd.read_csv("data_in/MiseNatPopTari.csv")
date_numerice = date.select_dtypes(include=[np.number])

scaler = StandardScaler()
date_standard = scaler.fit_transform(date_numerice)

pca = PCA()
pca.fit(date_standard)

# Varianta componente
varianta = pca.explained_variance_
print("Varianta componentelor:\n", varianta)

# Plot varianta componente
varianta_explicata = pca.explained_variance_ratio_
varianta_cumulativa = np.cumsum(varianta_explicata)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(varianta)+1), varianta_explicata, alpha=0.7, label="Varianta explicata")
plt.plot(range(1, len(varianta)+1), varianta_cumulativa, marker="o", linestyle="--", label="Varianta cumulativa")
plt.axhline(y=0.8, color='green', linestyle='--', label='80%')
plt.xlabel("Componente")
plt.ylabel("Varianta")
plt.legend()
plt.grid()
plt.show()

# Corelatii factoriale
corelatie_factoriala = pca.components_.T * np.sqrt(varianta)
print("\nCorelatii factoriale:\n", corelatie_factoriala)

# Corelograma corelatii factoriale
plt.figure(figsize=(12, 8))
sb.heatmap(corelatie_factoriala, annot=True, fmt=".2f", cmap='coolwarm', center=0,
           xticklabels=[f"PC{i+1}" for i in range(len(varianta))],
           yticklabels=date_numerice.columns)
plt.xlabel("Componente")
plt.ylabel("Variabile")
plt.show()

# Cercul corelatiilor
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.add_patch(plt.Circle((0, 0), 1, color="gray", fill=False))
for i, (x, y) in enumerate(corelatie_factoriala[:, :2]):
    ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc="red", ec="red")
    ax.text(x*1.1, y*1.1, date_numerice.columns[i], fontsize=9, ha='center')
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
plt.xlabel(f"PC1 ({varianta_explicata[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({varianta_explicata[1]*100:.1f}%)")
plt.grid(alpha=0.3)
plt.show()

# Scoruri
scoruri = pca.transform(date_standard)
print("\nScoruri (primele 5):\n", scoruri[:5])

# Plot scoruri
plt.figure(figsize=(10, 8))
plt.scatter(scoruri[:, 0], scoruri[:, 1], alpha=0.6)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel(f"PC1 ({varianta_explicata[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({varianta_explicata[1]*100:.1f}%)")
plt.grid(alpha=0.3)
plt.show()

# Cosinusuri
cosinusuri = scoruri**2 / np.sum(scoruri**2, axis=1, keepdims=True)
print("\nCosinusuri (primele 5):\n", cosinusuri[:5])

# Contributii
contributii = varianta_explicata * 100
print("\nContributii componente (%):\n", contributii)

# Comunalitati
comunalitati = np.sum(corelatie_factoriala**2, axis=1)
print("\nComunalitati:\n", comunalitati)

# Corelograma comunalitati
plt.figure(figsize=(8, 10))
sb.heatmap(pd.DataFrame(comunalitati, columns=['Comunalități'], index=date_numerice.columns),
           annot=True, fmt=".2f", cmap='viridis')
plt.show()