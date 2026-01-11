import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import seaborn as sns

df_penguins = pd.read_csv("penguins_size.csv")
print(df_penguins)


def clean(df):
    isinstance(df, pd.DataFrame)
    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    return df


df_penguins_clean = clean(df_penguins)

# vb dependenta si vb independente
X = df_penguins_clean.drop(columns=["species"])
Y = df_penguins_clean["species"]

# Salvare nume originale înainte de encoding
original_columns = X.columns.tolist()

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# STANDARDIZARE
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# creare si antrenare model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)

# predictii
Y_pred = lda.predict(X_test)
df_predict = pd.DataFrame({
    "Real": Y_test,
    "Predicted": Y_pred
})
df_predict.to_csv("predicted.csv")

# evaluare
# matrice de confuzie
conf = confusion_matrix(Y_test, Y_pred)
print("Matricea de confuzie")
print(conf)

# acuratete globala
glob = accuracy_score(Y_test, Y_pred)
print("acuratete globala: ", glob)

# acuratete medie
acc_per_class = conf.diagonal() / conf.sum(axis=1)
acc_mean = np.mean(acc_per_class)
print("acuratete medie: ", acc_mean)

# grafic distributie
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)
num_axes = X_train_lda.shape[1]

if num_axes < 2:
    print("numar insuficient de axe")
else:
    plt.figure(figsize=(10, 6))
    for label, marker, color in zip(np.unique(Y_train), ('o', 'x', 's'), ('blue', 'red', 'green')):
        label_indices = np.where(Y_train == label)[0]
        plt.scatter(X_train_lda[label_indices, 0],
                    X_train_lda[label_indices, 1],
                    label=f"Clasa {label}",
                    marker=marker,
                    color=color
                    )
    plt.title("Distribuțiile pe axele discriminante")
    plt.xlabel("Axa discriminantă 1")
    plt.ylabel("Axa discriminantă 2")
    plt.legend(title="Clase")
    plt.grid()
    plt.tight_layout()
    plt.savefig('grafic_distributii.png', dpi=300)
    plt.show()



print("\n" + "=" * 80)
print("TABELE SUPLIMENTARE PENTRU TEMĂ")
print("=" * 80)

# 1. PUTEREA DE DISCRIMINARE A VARIABILELOR PREDICTOR (Test Fisher)
print("\n1. PUTEREA DE DISCRIMINARE A VARIABILELOR PREDICTOR")
f_stats = []
p_values = []
for col in X.columns:
    groups = [X_train[Y_train == species][col].values for species in Y_train.unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    f_stats.append(f_stat)
    p_values.append(p_val)

df_fisher_predictori = pd.DataFrame({
    'Variabila': original_columns,
    'F-Statistic': f_stats,
    'p-value': p_values,
    'Semnificativ': ['Da' if p < 0.05 else 'Nu' for p in p_values]
}).sort_values('F-Statistic', ascending=False)
print(df_fisher_predictori)
df_fisher_predictori.to_csv("1_putere_discriminare_predictori.csv", index=False)

# 2. SCORURI DISCRIMINANTE (Coeficienți variabile discriminante)
print("\n2. SCORURI DISCRIMINANTE")
df_scoruri = pd.DataFrame(
    lda.scalings_,
    index=original_columns,
    columns=[f'LD{i + 1}' for i in range(num_axes)]
)
print(df_scoruri)
df_scoruri.to_csv("2_scoruri_discriminante.csv")

# 3. PUTEREA DE DISCRIMINARE A VARIABILELOR DISCRIMINANTE (Test Fisher)
print("\n3. PUTEREA AXELOR DISCRIMINANTE")
f_stats_lda = []
p_values_lda = []
for i in range(num_axes):
    groups = [X_train_lda[Y_train == species, i] for species in Y_train.unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    f_stats_lda.append(f_stat)
    p_values_lda.append(p_val)

df_fisher_axe = pd.DataFrame({
    'Axa': [f'LD{i + 1}' for i in range(num_axes)],
    'Varianță explicată (%)': lda.explained_variance_ratio_ * 100,
    'F-Statistic': f_stats_lda,
    'p-value': p_values_lda,
    'Semnificativ': ['Da' if p < 0.05 else 'Nu' for p in p_values_lda]
})
print(df_fisher_axe)
df_fisher_axe.to_csv("3_putere_axe_discriminante.csv", index=False)

# 4. CLASIFICARE ÎN SETUL DE ANTRENAMENT
print("\n4. CLASIFICARE SET ANTRENAMENT")
Y_train_pred = lda.predict(X_train)

df_clasificare_train = pd.DataFrame({
    'Real': Y_train.values,
    'Prezis': Y_train_pred
})
print(df_clasificare_train.head(20))
df_clasificare_train.to_csv("4_clasificare_antrenament.csv", index=False)

# 5. ACURATEȚEA CLASIFICĂRII (Set antrenament)
print("\n5. ACURATEȚE CLASIFICARE (SET ANTRENAMENT)")
conf_train = confusion_matrix(Y_train, Y_train_pred)
acc_train = accuracy_score(Y_train, Y_train_pred)
kappa_train = cohen_kappa_score(Y_train, Y_train_pred)

print(f"Acuratețe globală: {acc_train:.4f} ({acc_train * 100:.2f}%)")
print(f"Index Cohen-Kappa: {kappa_train:.4f}")

# Acuratețe pe clasă
acc_per_class_train = conf_train.diagonal() / conf_train.sum(axis=1)
df_acuratete = pd.DataFrame({
    'Clasa': sorted(Y_train.unique()),
    'Acuratețe': acc_per_class_train,
    'Acuratețe (%)': acc_per_class_train * 100
})
print(df_acuratete)
df_acuratete.to_csv("5_acuratete_clasificare.csv", index=False)

# 6. MATRICEA DE CONFUZIE (Set antrenament)
print("\n6. MATRICE CONFUZIE (SET ANTRENAMENT)")
print(conf_train)
df_conf_train = pd.DataFrame(
    conf_train,
    index=[f'Real_{cls}' for cls in sorted(Y_train.unique())],
    columns=[f'Prezis_{cls}' for cls in sorted(Y_train.unique())]
)
df_conf_train.to_csv("6_matrice_confuzie_antrenament.csv")

# 7. GRAFIC: INSTANȚE ȘI CENTRII PE AXELE DISCRIMINANTE
print("\n7. GRAFIC INSTANȚE ȘI CENTRII")
plt.figure(figsize=(12, 8))
colors = {'Adelie': 'blue', 'Chinstrap': 'red', 'Gentoo': 'green'}
markers = {'Adelie': 'o', 'Chinstrap': 'x', 'Gentoo': 's'}

for species in sorted(Y_train.unique()):
    indices = Y_train == species
    plt.scatter(X_train_lda[indices, 0], X_train_lda[indices, 1],
                label=species,
                color=colors.get(species),
                marker=markers.get(species),
                alpha=0.6, s=50)

    # Calcul și plotare centrii
    centroid_x = X_train_lda[indices, 0].mean()
    centroid_y = X_train_lda[indices, 1].mean()
    plt.scatter(centroid_x, centroid_y, marker='*', s=500,
                color=colors.get(species), edgecolors='black', linewidths=2,
                label=f'Centru {species}')

plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0] * 100:.1f}%)')
plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1] * 100:.1f}%)')
plt.title('Instanțe și centrii pe axele discriminante')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('7_instante_centrii.png', dpi=300)
plt.show()

# 8. APLICARE PE SETUL DE TESTARE
print("\n" + "=" * 80)
print("APLICARE MODELULUI LDA PE SETUL DE TESTARE")
print("=" * 80)

# Evaluare pe testare
kappa_test = cohen_kappa_score(Y_test, Y_pred)
conf_test = confusion_matrix(Y_test, Y_pred)

print(f"\nPERFORMANȚĂ MODEL LDA PE SETUL DE TESTARE:")
print(f"Acuratețe globală: {glob:.4f} ({glob * 100:.2f}%)")
print(f"Index Cohen-Kappa: {kappa_test:.4f}")

# Tabel comparație
df_comparatie = pd.DataFrame({
    'Set de date': ['Antrenament', 'Testare'],
    'Acuratețe globală': [acc_train, glob],
    'Cohen-Kappa': [kappa_train, kappa_test]
})
print("\nComparație performanță:")
print(df_comparatie)
df_comparatie.to_csv("8_comparatie_antrenament_testare.csv", index=False)

# Salvare predicții finale
df_final = pd.DataFrame({
    'Real': Y_test.values,
    'Prezis': Y_pred,
    'Corect': Y_test.values == Y_pred
})
print("\nPrimele 20 predicții finale:")
print(df_final.head(20))
df_final.to_csv("9_predictii_finale_testare.csv", index=False)

# Matrice confuzie testare
print("\nMatrice confuzie testare:")
print(conf_test)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(Y_test.unique()),
            yticklabels=sorted(Y_test.unique()))
plt.title('Matrice de confuzie - Set de testare')
plt.ylabel('Valoare reală')
plt.xlabel('Valoare prezisă')
plt.tight_layout()
plt.savefig('10_matrice_confuzie_testare.png', dpi=300)
plt.show()

# ============================================================================
# DISCRIMINAREA BAYESIANĂ (NAIVE BAYES)
# ============================================================================
print("\n" + "=" * 80)
print("DISCRIMINAREA BAYESIANĂ (NAIVE BAYES)")
print("=" * 80)

# Creare și antrenare model Naive Bayes
model_bayes = GaussianNB()
model_bayes.fit(X_train, Y_train)

# Predicții pe set antrenament
Y_train_pred_bayes = model_bayes.predict(X_train)

# Clasificare în setul de antrenament
df_clasificare_train_bayes = pd.DataFrame({
    'Real': Y_train.values,
    'Prezis': Y_train_pred_bayes
})
print("\nPrimele 20 clasificări Naive Bayes (antrenament):")
print(df_clasificare_train_bayes.head(20))
df_clasificare_train_bayes.to_csv("11_clasificare_antrenament_bayes.csv", index=False)

# Acuratețea clasificării (antrenament)
conf_train_bayes = confusion_matrix(Y_train, Y_train_pred_bayes)
acc_train_bayes = accuracy_score(Y_train, Y_train_pred_bayes)
kappa_train_bayes = cohen_kappa_score(Y_train, Y_train_pred_bayes)

print(f"\nACURATEȚE NAIVE BAYES (SET ANTRENAMENT):")
print(f"Acuratețe globală: {acc_train_bayes:.4f} ({acc_train_bayes * 100:.2f}%)")
print(f"Index Cohen-Kappa: {kappa_train_bayes:.4f}")

# Acuratețe pe clasă
acc_per_class_train_bayes = conf_train_bayes.diagonal() / conf_train_bayes.sum(axis=1)
df_acuratete_bayes = pd.DataFrame({
    'Clasa': sorted(Y_train.unique()),
    'Acuratețe': acc_per_class_train_bayes,
    'Acuratețe (%)': acc_per_class_train_bayes * 100
})
print(df_acuratete_bayes)
df_acuratete_bayes.to_csv("12_acuratete_clasificare_bayes.csv", index=False)

# Matricea de confuzie (antrenament)
print("\nMATRICE CONFUZIE NAIVE BAYES (SET ANTRENAMENT):")
print(conf_train_bayes)
df_conf_train_bayes = pd.DataFrame(
    conf_train_bayes,
    index=[f'Real_{cls}' for cls in sorted(Y_train.unique())],
    columns=[f'Prezis_{cls}' for cls in sorted(Y_train.unique())]
)
df_conf_train_bayes.to_csv("13_matrice_confuzie_antrenament_bayes.csv")

# Predicții pe set testare
Y_pred_bayes = model_bayes.predict(X_test)
acc_test_bayes = accuracy_score(Y_test, Y_pred_bayes)
kappa_test_bayes = cohen_kappa_score(Y_test, Y_pred_bayes)

print(f"\nACURATEȚE NAIVE BAYES (SET TESTARE):")
print(f"Acuratețe globală: {acc_test_bayes:.4f} ({acc_test_bayes * 100:.2f}%)")
print(f"Index Cohen-Kappa: {kappa_test_bayes:.4f}")

# ============================================================================
# COMPARAȚIE MODELE ȘI ALEGEREA CELUI MAI BUN
# ============================================================================
print("\n" + "=" * 80)
print("COMPARAȚIE MODELE: LDA vs NAIVE BAYES")
print("=" * 80)

df_comparatie_modele = pd.DataFrame({
    'Model': ['LDA', 'Naive Bayes'],
    'Acuratețe Antrenament': [acc_train, acc_train_bayes],
    'Kappa Antrenament': [kappa_train, kappa_train_bayes],
    'Acuratețe Testare': [glob, acc_test_bayes],
    'Kappa Testare': [kappa_test, kappa_test_bayes]
})
print(df_comparatie_modele)
df_comparatie_modele.to_csv("14_comparatie_modele.csv", index=False)

# Alegerea celui mai bun model bazat pe acuratețe testare
best_idx = df_comparatie_modele['Acuratețe Testare'].idxmax()
best_model_name = df_comparatie_modele.loc[best_idx, 'Model']
best_acc = df_comparatie_modele.loc[best_idx, 'Acuratețe Testare']
best_kappa = df_comparatie_modele.loc[best_idx, 'Kappa Testare']

print(f"\n{'=' * 80}")
print(f"CEL MAI BUN MODEL: {best_model_name}")
print(f"Acuratețe testare: {best_acc:.4f} ({best_acc * 100:.2f}%)")
print(f"Cohen-Kappa testare: {best_kappa:.4f}")
print(f"{'=' * 80}")

# Salvare predicții finale cu ambele modele
df_predictii_finale = pd.DataFrame({
    'Real': Y_test.values,
    'Prezis_LDA': Y_pred,
    'Prezis_Bayes': Y_pred_bayes,
    'Corect_LDA': Y_test.values == Y_pred,
    'Corect_Bayes': Y_test.values == Y_pred_bayes
})
print("\nPrimele 20 predicții finale (ambele modele):")
print(df_predictii_finale.head(20))
df_predictii_finale.to_csv("15_predictii_finale_ambele_modele.csv", index=False)

# Analiză diferențe între modele
diferente = df_predictii_finale[df_predictii_finale['Prezis_LDA'] != df_predictii_finale['Prezis_Bayes']]
print(
    f"\nNumăr instanțe cu predicții diferite: {len(diferente)} din {len(Y_test)} ({len(diferente) / len(Y_test) * 100:.1f}%)")
if len(diferente) > 0:
    print("\nInstanțe cu predicții diferite:")
    print(diferente.head(10))
    diferente.to_csv("16_diferente_predictii.csv", index=False)

# Erori pentru fiecare model
erori_lda = df_predictii_finale[~df_predictii_finale['Corect_LDA']]
erori_bayes = df_predictii_finale[~df_predictii_finale['Corect_Bayes']]

print(f"\nErori LDA: {len(erori_lda)}")
print(f"Erori Naive Bayes: {len(erori_bayes)}")

if len(erori_lda) > 0:
    erori_lda.to_csv("17_erori_lda.csv", index=False)
if len(erori_bayes) > 0:
    erori_bayes.to_csv("18_erori_bayes.csv", index=False)

# Matrice confuzie comparativă
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# LDA
sns.heatmap(conf_test, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=sorted(Y_test.unique()),
            yticklabels=sorted(Y_test.unique()))
axes[0].set_title(f'LDA - Acuratețe: {glob:.3f}')
axes[0].set_ylabel('Valoare reală')
axes[0].set_xlabel('Valoare prezisă')

# Naive Bayes
conf_test_bayes = confusion_matrix(Y_test, Y_pred_bayes)
sns.heatmap(conf_test_bayes, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=sorted(Y_test.unique()),
            yticklabels=sorted(Y_test.unique()))
axes[1].set_title(f'Naive Bayes - Acuratețe: {acc_test_bayes:.3f}')
axes[1].set_ylabel('Valoare reală')
axes[1].set_xlabel('Valoare prezisă')

plt.tight_layout()
plt.savefig('19_comparatie_matrici_confuzie.png', dpi=300)
plt.show()

