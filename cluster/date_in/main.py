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
matrice_ierarhie = linkage(df_std, method="ward")
print("\n=== 1. MATRICE IERARHIE ===")
print(matrice_ierarhie)

# 2. CALCUL PARTIȚIE OPTIMALĂ - Elbow cu diferență ORDINUL 1
distante = matrice_ierarhie[:, 2]
dif_distante = np.diff(distante)  # Diferență ordinul 1 (nu 2!)
idx_max = np.argmax(dif_distante)

nobs = matrice_ierarhie.shape[0]
k_optimal = nobs - idx_max

# Threshold pentru dendrogramă
threshold_optimal = (distante[idx_max] + distante[idx_max + 1]) / 2

print(f"\n=== 2. PARTIȚIE OPTIMALĂ - ELBOW (diferență ord 1) ===")
print(f"Index maxim în diferențe: {idx_max}")
print(f"Număr pași agregare: {nobs}")
print(f"Număr optim clusteri: {k_optimal}")
print(f"Threshold optim: {threshold_optimal:.4f}")

# Grafic Elbow
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Graf diferențe ordinul 1
nr_clusteri_dif = np.array([nobs - i for i in range(len(dif_distante))])
ax1.plot(nr_clusteri_dif, dif_distante, 'b-o', linewidth=2)
ax1.axvline(k_optimal, color='r', linestyle='--', linewidth=2, label=f'k={k_optimal}')
ax1.scatter([k_optimal], [dif_distante[idx_max]], color='r', s=200, zorder=5, marker='*')
ax1.set_title("Diferențe ordinul 1 (Elbow)")
ax1.set_xlabel("Număr clusteri")
ax1.set_ylabel("Diferență ord 1")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graf distanțe
nr_clusteri_dist = np.array([nobs + 1 - i for i in range(len(distante))])
ax2.plot(nr_clusteri_dist, distante, 'g-o', linewidth=2)
ax2.axvline(k_optimal, color='r', linestyle='--', linewidth=2, label=f'k={k_optimal}')
ax2.axhline(threshold_optimal, color='orange', linestyle=':', linewidth=2, label='Threshold')
ax2.set_title("Distanțe agregare")
ax2.set_xlabel("Număr clusteri")
ax2.set_ylabel("Distanță")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. CALCUL PARTIȚII
clusteri_optimal = fcluster(matrice_ierarhie, k_optimal, criterion="maxclust")
clusteri_k = fcluster(matrice_ierarhie, K_PREDEFINIT, criterion="maxclust")

# Threshold pentru partiția k
threshold_k = (distante[nobs - K_PREDEFINIT] + distante[nobs - K_PREDEFINIT + 1]) / 2

print(f"\n=== 3. PARTIȚII ===")
print(f"Partiție optimală: {np.bincount(clusteri_optimal)[1:]} instanțe/cluster")
print(f"Partiție k={K_PREDEFINIT}: {np.bincount(clusteri_k)[1:]} instanțe/cluster")
print(f"Threshold k={K_PREDEFINIT}: {threshold_k:.4f}")

# 4. INDECȘI SILHOUETTE
sil_optimal = silhouette_score(df_std, clusteri_optimal)
sil_k = silhouette_score(df_std, clusteri_k)
sil_instante_optimal = silhouette_samples(df_std, clusteri_optimal)
sil_instante_k = silhouette_samples(df_std, clusteri_k)

print(f"\n=== 4. INDECȘI SILHOUETTE ===")
print(f"Silhouette partiție optimală (k={k_optimal}): {sil_optimal:.3f}")
print(f"Silhouette partiție-{K_PREDEFINIT}: {sil_k:.3f}")

# 5. DENDROGRAMĂ CU EVIDENȚIERE CULOARE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Partiția optimă
dendrogram(matrice_ierarhie, color_threshold=threshold_optimal, labels=etichete,
           leaf_font_size=6, ax=ax1)
ax1.axhline(threshold_optimal, color='red', linestyle='--', linewidth=2, label='Prag optim')
ax1.set_title(f'Dendrogramă - Partiție optimală ({k_optimal} clustere)')
ax1.set_xlabel('Instanțe')
ax1.set_ylabel('Distanță')
ax1.legend()
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

# Partiția k
dendrogram(matrice_ierarhie, color_threshold=threshold_k, labels=etichete,
           leaf_font_size=6, ax=ax2)
ax2.axhline(threshold_k, color='magenta', linestyle='--', linewidth=2, label=f'Prag k={K_PREDEFINIT}')
ax2.set_title(f'Dendrogramă - Partiție k={K_PREDEFINIT}')
ax2.set_xlabel('Instanțe')
ax2.set_ylabel('Distanță')
ax2.legend()
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

plt.tight_layout()
plt.show()

print(f"\n=== 5. DENDROGRAMĂ ===")


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
                f"Plot Silhouette - Partiție optimală ({k_optimal} clustere)")
plot_silhouette(df_std, clusteri_k,
                f"Plot Silhouette - Partiție k={K_PREDEFINIT}")


# 7. HISTOGRAME
def plot_histograme(data, labels, title, k):
    df_grafic = pd.DataFrame(data, columns=date.columns)
    df_grafic['Cluster'] = labels

    for col in data.columns:
        fig, ax = plt.subplots(figsize=(10, 5))

        for eticheta_cluster in sorted(np.unique(labels)):
            sectiune = df_grafic[df_grafic['Cluster'] == eticheta_cluster]
            ax.hist(sectiune[col], bins=15, alpha=0.5, label=f'Cluster {eticheta_cluster}')

        ax.set_xlabel(col)
        ax.set_ylabel('Frecvență')
        ax.set_title(f'Histogramă {col} - {title}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()


print(f"\n=== 7. HISTOGRAME ===")
plot_histograme(date, clusteri_optimal, f"Partiție optimală ({k_optimal} clustere)", k_optimal)
plot_histograme(date, clusteri_k, f"Partiție k={K_PREDEFINIT}", K_PREDEFINIT)

# 8. PLOT PCA
pca = PCA()
scores = pca.fit_transform(df_std)

df_scatter = pd.DataFrame(scores[:, :2], columns=['PC1', 'PC2'])
df_scatter['Cluster_Optimal'] = clusteri_optimal
df_scatter['Cluster_K'] = clusteri_k

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot pentru partiția optimă
for eticheta_cluster in sorted(np.unique(clusteri_optimal)):
    sectiune = df_scatter[df_scatter['Cluster_Optimal'] == eticheta_cluster]
    ax1.scatter(sectiune['PC1'], sectiune['PC2'],
                label=f'Cluster {eticheta_cluster}', alpha=0.6, s=100)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.set_title(f'Scatter plot - Partiție optimală ({k_optimal} clustere)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot pentru partiția k
for eticheta_cluster in sorted(np.unique(clusteri_k)):
    sectiune = df_scatter[df_scatter['Cluster_K'] == eticheta_cluster]
    ax2.scatter(sectiune['PC1'], sectiune['PC2'],
                label=f'Cluster {eticheta_cluster}', alpha=0.6, s=100)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.set_title(f'Scatter plot - Partiție k={K_PREDEFINIT}')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== 8. PLOT PCA ===")

print(f"\n=== REZUMAT ===")
print(f"k optimal (Elbow ord 1): {k_optimal} (Silhouette: {sil_optimal:.3f})")
print(f"k predefinit: {K_PREDEFINIT} (Silhouette: {sil_k:.3f})")
print(f"Varianță PCA (PC1+PC2): {sum(pca.explained_variance_ratio_[:2]):.1%}")