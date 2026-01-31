import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.cross_decomposition import CCA
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

# Incarcare date
date = pd.read_csv('Vot.csv')

for col in date.columns:
    if date[col].isna().any():
        if is_numeric_dtype(date[col]):
            date[col].fillna(date[col].mean(), inplace=True)
        else:
            date[col].fillna(date[col].mode()[0], inplace=True)

x = date.iloc[:, 3:7].values
y = date.iloc[:, 7:11].values

n, p, q = len(date), x.shape[1], y.shape[1]
m = min(p, q)

x = StandardScaler().fit_transform(x)
y = StandardScaler().fit_transform(y)

# Analiza canonica / Calcul scoruri canonice
cca = CCA(n_components=m)
cca.fit(x, y)
x_c, y_c = cca.transform(x, y)

# Calcul corelatii canonice
r = np.array([np.corrcoef(x_c[:, i], y_c[:, i])[0, 1] for i in range(m)])
r2 = r ** 2

# Determinare relevanta radacini canonice (Test Bartlett)
r2_safe = np.clip(r2, 0, 0.9999999)
w_lambda = np.flip(np.cumprod(np.flip(1 - r2_safe)))
p_vals = []
for k in range(m):
    df = (p - k) * (q - k)
    chi2_stat = -(n - 1 - 0.5 * (p + q + 1)) * np.log(w_lambda[k])
    p_vals.append(1 - chi2.cdf(chi2_stat, df))
p_vals = np.array(p_vals)

# Calcul corelatii variabile observate - variabile canonice
corr_x_xc = np.corrcoef(x.T, x_c.T)[:p, p:]
corr_y_yc = np.corrcoef(y.T, y_c.T)[:q, q:]

# Trasare plot corelatii variabile observate - variabile canonice (cercul corelatiilor)
if m >= 2:
    plt.figure(figsize=(9, 9))
    theta = np.arange(0, 2 * np.pi, 0.01)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', lw=1.5)
    plt.axhline(0, c='gray', lw=0.5)
    plt.axvline(0, c='gray', lw=0.5)
    plt.scatter(corr_x_xc[:, 0], corr_x_xc[:, 1], c='blue', s=100, edgecolors='k', label='X')
    plt.scatter(corr_y_yc[:, 0], corr_y_yc[:, 1], c='red', s=100, edgecolors='k', label='Y')

    for i in range(p):
        plt.text(corr_x_xc[i, 0], corr_x_xc[i, 1], f'X{i+1}', c='blue', fontweight='bold')
    for i in range(q):
        plt.text(corr_y_yc[i, 0], corr_y_yc[i, 1], f'Y{i+1}', c='red', fontweight='bold')

    plt.title('Cercul corelatiilor')
    plt.legend()
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.savefig('cercul_corelatiilor.png')
    plt.close()

# Trasare corelograma corelatii variabile observate - variabile canonice
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr_x_xc, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[0])
axes[0].set_title('Corelograma X')
sns.heatmap(corr_y_yc, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1])
axes[1].set_title('Corelograma Y')
plt.savefig('corelograme.png')
plt.close()

# Trasare plot instante in spatiile celor doua variabile (Biplot)
if m >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(x_c[:, 0], x_c[:, 1], alpha=0.5)
    axes[0].set_title('Instante X')
    axes[1].scatter(y_c[:, 0], y_c[:, 1], alpha=0.5, c='coral')
    axes[1].set_title('Instante Y')
    plt.savefig('biplot.png')
    plt.close()

# Calcul varianta explicata si redundanta informationala
communality_x = corr_x_xc ** 2
communality_y = corr_y_yc ** 2
var_expl_x = communality_x.mean(axis=0)
var_expl_y = communality_y.mean(axis=0)
redundancy_y_given_x = (r2 * var_expl_y).sum()
redundancy_x_given_y = (r2 * var_expl_x).sum()

# Salvare rezultate
pd.DataFrame(x_c).to_csv('x.csv', index=False)
pd.DataFrame(y_c).to_csv('y.csv', index=False)
pd.DataFrame({'R': r, 'p_val': p_vals}).to_csv('r.csv', index=False)