import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==================== PARAMETRI CONFIGURABILI ====================
FISIER_DATE = "alcohol.csv"
K_PREDEFINIT = 5  # Partiție oarecare
# ==================================================================

# Citire date
date_raw = pd.read_csv(FISIER_DATE)
etichete = date_raw['Entity'].values if 'Entity' in date_raw.columns else None
date = date_raw.select_dtypes(include=[np.number])

if date.shape[1] == 0:
    raise ValueError("Dataset-ul nu conține coloane numerice!")

# Curățare date prin IMPUTARE
print(f"Date originale: {date.shape}")
if date.isna().any().any():
    for col in date.columns:
        if date[col].isna().any():
            date[col] = date[col].fillna(date[col].mean())

date = date.replace([np.inf, -np.inf], np.nan)
if date.isna().any().any():
    for col in date.columns:
        if date[col].isna().any():
            date[col] = date[col].fillna(date[col].mean())

if etichete is not None:
    etichete = etichete[date.index]

print(f"Date curățate: {date.shape}")

# Standardizare
scaler = StandardScaler()
date_std = scaler.fit_transform(date)
df_std = pd.DataFrame(date_std, columns=date.columns, index=date.index)
print("Date standardizate")

# 1. CALCUL IERARHIE
z = linkage(df_std, method="ward")
print("\n=== 1. MATRICE IERARHIE ===")
print(z)

# 2. CALCUL PARTIȚIE OPTIMALĂ - Elbow CLASIC CORECT
distanta = z[:, 2]
diferente = np.diff(distanta, 2)

# PROBLEMA: np.argmax(diferente) găsește indexul în vectorul diferente
# SOLUȚIA: trebuie să traducem corect acest index în număr de clusteri

# diferente[i] corespunde tranziției de la (n-i-1) la (n-i) clusteri
# unde n = len(distanta) + 1 (numărul de observații)
# Deci k = n - i - 1

idx_max = np.argmax(diferente)
n_obs = len(distanta) + 1
k_optimal = n_obs - idx_max - 1

print(f"\n=== 2. PARTIȚIE OPTIMALĂ - ELBOW CLASIC ===")
print(f"Index maxim în diferențe: {idx_max}")
print(f"Număr observații: {n_obs}")
print(f"Număr optim clusteri: {k_optimal}")

# Verificare: dacă k e prea mare (>30), avertizare
if k_optimal > 30:
    print(f"\n⚠️ ATENȚIE: k={k_optimal} pare prea mare!")
    print(f"Verifică datele sau consideră metoda cu restricție.")
    print(f"Cele mai mari 10 diferențe:")
    top_10_idx = np.argsort(diferente)[-10:][::-1]
    for rank, idx in enumerate(top_10_idx, 1):
        k = n_obs - idx - 1
        print(f"  {rank}. k={k:3d}, diferență={diferente[idx]:.4f}")

# Grafic Elbow
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Graf diferențe - diferente are len(distanta)-2 elemente
# diferente[i] corespunde la k = n_obs - i - 1
nr_clusteri_dif = np.array([n_obs - i - 1 for i in range(len(diferente))])
ax1.plot(nr_clusteri_dif, diferente, 'b-o', linewidth=2)
ax1.axvline(k_optimal, color='r', linestyle='--', linewidth=2, label=f'k={k_optimal}')
ax1.scatter([k_optimal], [diferente[idx_max]], color='r', s=200, zorder=5, marker='*')
ax1.set_title("Diferențe ordinul 2 (Elbow)")
ax1.set_xlabel("Număr clusteri")
ax1.set_ylabel("Diferență ord 2")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graf distanțe - distanta are len(distanta) elemente
nr_clusteri_dist = np.array([n_obs - i for i in range(len(distanta))])
ax2.plot(nr_clusteri_dist, distanta, 'g-o', linewidth=2)
ax2.axvline(k_optimal, color='r', linestyle='--', linewidth=2, label=f'k={k_optimal}')
ax2.set_title("Distanțe agregare")
ax2.set_xlabel("Număr clusteri")
ax2.set_ylabel("Distanță")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. CALCUL PARTIȚII
clusteri_optimal = fcluster(z, t=k_optimal, criterion="maxclust")
clusteri_k = fcluster(z, t=K_PREDEFINIT, criterion="maxclust")

# 4. INDECȘI SILHOUETTE
sil_optimal = silhouette_score(df_std, clusteri_optimal)
sil_k = silhouette_score(df_std, clusteri_k)
sil_instante_optimal = silhouette_samples(df_std, clusteri_optimal)
sil_instante_k = silhouette_samples(df_std, clusteri_k)

print(f"\n=== 4. INDECȘI SILHOUETTE ===")
print(f"Silhouette partiție optimală (k={k_optimal}): {sil_optimal:.3f}")
print(f"Silhouette partiție-{K_PREDEFINIT}: {sil_k:.3f}")


# 5. DENDROGRAMĂ
def plot_dendrograma(z, distanta, k, etichete, title):
    plt.figure(figsize=(12, 7))
    prag = (distanta[-(k - 1)] + distanta[-k]) / 2 if k > 1 else distanta[-1] + 1
    dendrogram(z, color_threshold=prag, labels=etichete, leaf_font_size=6)
    plt.axhline(y=prag, color='r', linestyle='--', label=f'k={k}')
    plt.title(title)
    plt.xlabel("Instanțe")
    plt.ylabel("Distanță")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


print(f"\n=== 5. DENDROGRAMĂ ===")
plot_dendrograma(z, distanta, k_optimal, etichete,
                 f"Dendrogramă - Partiție optimală (k={k_optimal})")
plot_dendrograma(z, distanta, K_PREDEFINIT, etichete,
                 f"Dendrogramă - Partiție-{K_PREDEFINIT}")


# 6. PLOT SILHOUETTE
def plot_silhouette(data, labels, title):
    n_clusters = len(np.unique(labels))
    sil_vals = silhouette_samples(data, labels)
    sil_avg = silhouette_score(data, labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for i in range(1, n_clusters + 1):
        cluster_sil = sil_vals[labels == i]
        cluster_sil.sort()
        y_upper = y_lower + cluster_sil.shape[0]

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=plt.cm.viridis(float(i) / n_clusters),
                         edgecolor='none', alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * cluster_sil.shape[0], f'C{i}')
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--",
               label=f'Medie: {sil_avg:.3f}')
    ax.set_title(title)
    ax.set_xlabel("Coeficient Silhouette")
    ax.set_ylabel("Cluster")
    ax.set_xlim([-0.1, 1])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.show()


print(f"\n=== 6. PLOT SILHOUETTE ===")
plot_silhouette(df_std, clusteri_optimal,
                f"Plot Silhouette - k={k_optimal}")
plot_silhouette(df_std, clusteri_k,
                f"Plot Silhouette - k={K_PREDEFINIT}")


# 7. HISTOGRAME
def plot_histograme(data, labels, title, k):
    for var in data.columns:
        fig, axs = plt.subplots(1, k, figsize=(min(15, k * 3), 4), sharey=True)
        fig.suptitle(f"Histograme '{var}' - {title}", fontsize=12)

        if k == 1:
            axs = [axs]
        elif k > 1:
            axs = axs if hasattr(axs, '__iter__') else [axs]

        for i in range(1, min(k + 1, 21)):  # Max 20 subplots
            if i - 1 < len(axs):
                cluster_vals = data[var][labels == i]
                axs[i - 1].hist(cluster_vals, bins=10, color='steelblue',
                                alpha=0.7, edgecolor='black')
                axs[i - 1].set_xlabel(f"C{i}")
                axs[i - 1].set_title(f'n={cluster_vals.shape[0]}', fontsize=10)
                axs[i - 1].grid(True, alpha=0.3, axis='y')

        if len(axs) > 0:
            axs[0].set_ylabel("Frecvență")
        plt.tight_layout()
        plt.show()

        if k > 20:
            print(f"  ⚠️ Prea multe clusteri ({k}) pentru histograme - afișate doar primele 20")
            break


print(f"\n=== 7. HISTOGRAME ===")
if k_optimal <= 20:
    plot_histograme(date, clusteri_optimal, f"k={k_optimal}", k_optimal)
else:
    print(f"⚠️ Skip histograme pentru k={k_optimal} (prea multe clusteri)")
plot_histograme(date, clusteri_k, f"k={K_PREDEFINIT}", K_PREDEFINIT)

# 8. PLOT PCA
pca = PCA(n_components=2)
date_pca = pca.fit_transform(df_std)


def plot_pca(data_pca, labels, pca, etichete, title):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels,
                          cmap='viridis', s=100, alpha=0.6, edgecolors='black')
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.grid(True, alpha=0.3)

    if etichete is not None and len(np.unique(labels)) <= 20:
        for i, (x, y) in enumerate(data_pca):
            if i % 5 == 0:
                plt.annotate(etichete[i][:8], (x, y), fontsize=6, alpha=0.7)

    plt.tight_layout()
    plt.show()


print(f"\n=== 8. PLOT PCA ===")
plot_pca(date_pca, clusteri_optimal, pca, etichete,
         f"Partiție PCA - k={k_optimal}")
plot_pca(date_pca, clusteri_k, pca, etichete,
         f"Partiție PCA - k={K_PREDEFINIT}")

print(f"\n=== REZUMAT ===")
print(f"k optimal (Elbow clasic): {k_optimal} (Silhouette: {sil_optimal:.3f})")
print(f"k predefinit: {K_PREDEFINIT} (Silhouette: {sil_k:.3f})")
print(f"Varianță PCA: {sum(pca.explained_variance_ratio_):.1%}")