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
