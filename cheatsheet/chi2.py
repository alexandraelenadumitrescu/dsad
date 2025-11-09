import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, IntSlider

sns.set(style="whitegrid")


def plot_chi2(df=1, n_samples=10000):
    # Generează df variabile normale independente
    Z = np.random.normal(0, 1, size=(n_samples, df))

    # Suma pătratelor → chi-pătrat
    chi2 = np.sum(Z ** 2, axis=1)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(chi2, bins=50, kde=True, stat="density", color="skyblue")
    plt.title(f"Distribuția Chi-pătrat ({df} grade de libertate)")
    plt.xlabel("Valori")
    plt.ylabel("Densitate")
    plt.show()


# Slider interactiv pentru grade de libertate
interact(plot_chi2, df=IntSlider(min=1, max=10, step=1, value=1));


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generează 1000 de valori din distribuția normală standard
data = np.random.normal(loc=0, scale=1, size=1000)
data2=np.random.normal(loc=0, scale=1, size=1000)
data3 = np.random.normal(loc=0, scale=1, size=1000)
data4= np.random.normal(loc=0, scale=1, size=1000)
data5 = np.random.normal(loc=0, scale=1, size=1000)
data6 = np.random.normal(loc=0, scale=1, size=1000)
data7 = np.random.normal(loc=0, scale=1, size=1000)

chi2=data**2
chi22=data**2+data2**2
chi23=chi22+data3**2
chi24=chi23+data4**2
chi25=chi24+data5**2
chi26=chi25+data6**2
chi27=chi26+data7**2


# Setări estetice Seaborn
sns.set(style="whitegrid")

# Histogramă + densitate kernel
sns.histplot(data, kde=True, stat="density", bins=30, color="skyblue")
sns.histplot(chi2, bins=50, kde=True, stat="density", color="orange")
sns.histplot(chi22, bins=50, kde=True, stat="density", color="green")
sns.histplot(chi23, bins=50, kde=True, stat="density", color="yellow")
sns.histplot(chi24, bins=50, kde=True, stat="density", color="red")
sns.histplot(chi25, bins=50, kde=True, stat="density", color="black")
sns.histplot(chi26, bins=50, kde=True, stat="density", color="purple")
sns.histplot(chi27, bins=50, kde=True, stat="density", color="brown")

plt.title("Distribuția Chi-pătrat (1 grad de libertate)")
plt.xlabel("Z^2")
plt.ylabel("Densitate")
plt.show()


# Titlu și axe
plt.title("Distribuția Normală Standard")
plt.xlabel("Valori")
plt.ylabel("Densitate")

plt.show()

