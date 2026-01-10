import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df_penguins=pd.read_csv("penguins_size.csv")
print(df_penguins)

def clean(df):
    isinstance(df,pd.DataFrame)
    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col]=df[col].fillna(df[col].mean())
                else:
                    df[col]=df[col].fillna(df[col].mode()[0])
    return df
df_penguins_clean=clean(df_penguins)

#vb dependenta si vb independente

X=df_penguins_clean.drop(columns=["species"])
Y=df_penguins_clean["species"]


original_columns = X.columns.tolist()

label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le


#split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

#creare si antrenare model
lda= LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)

#predictii
Y_pred=lda.predict(X_test)
df_predict=pd.DataFrame({
    "Real":Y_test,
    "Predicted":Y_pred
})
df_predict.to_csv("predicted.csv")

#evaluare

#matrice de confuzie
conf=confusion_matrix(Y_test,Y_pred)
print("Matricea de confuzie")
print(conf)

#acuratete globala
glob=accuracy_score(Y_test,Y_pred)
print("acuratete globala: ",glob)


#acuratete medie
acc_per_class=conf.diagonal()/conf.sum(axis=1)
acc_mean=np.mean(acc_per_class)
print("acuratete medie: ",acc_mean)

#grafic distributie
X_train_lda=lda.transform(X_train)
X_test_lda=lda.transform(X_test)
num_axes=X_train_lda.shape[1]

if num_axes<2:
    print("numar insuficient de axe")
else:
    plt.figure(figsize=(10,6))
    for label,marker,color in zip(np.unique(Y_train),('o','x','s'),('blue','red','green')):
        label_indices=np.where(Y_train==label)[0]
        plt.scatter(X_train_lda[label_indices,0],
                    X_train_lda[label_indices,1],
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
    plt.show()

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

# 8. APLICARE PE SETUL DE TESTARE (cel mai bun model = LDA)
print("\n" + "=" * 80)
print("APLICARE PE SETUL DE TESTARE")
print("=" * 80)

# Evaluare pe testare
kappa_test = cohen_kappa_score(Y_test, Y_pred)
conf_test = confusion_matrix(Y_test, Y_pred)

print(f"\nACURATEȚE SET TESTARE:")
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
sb.heatmap(conf_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(Y_test.unique()),
            yticklabels=sorted(Y_test.unique()))
plt.title('Matrice de confuzie - Set de testare')
plt.ylabel('Valoare reală')
plt.xlabel('Valoare prezisă')
plt.tight_layout()
plt.savefig('10_matrice_confuzie_testare.png', dpi=300)
plt.show()



