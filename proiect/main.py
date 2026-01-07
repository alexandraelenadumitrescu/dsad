"""
PROIECT: Clasificarea speciilor de pinguini din arhipelagul Palmer
         după caracteristici morfologice
TEMA 2: Analiză Discriminantă (LDA + Discriminare Bayesiană)
DATASET: Palmer Penguins (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Setări grafice
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==============================================================================
# 1. ÎNCĂRCAREA ȘI PREGĂTIREA DATELOR
# ==============================================================================

def load_and_prepare_data(filepath):
    """
    Încarcă dataset-ul Palmer Penguins și face preprocessing.
    """
    # Încărcare date
    df = pd.read_csv(filepath)

    # Afișare informații generale
    print("=" * 80)
    print("INFORMAȚII DATASET")
    print("=" * 80)
    print(f"Dimensiuni: {df.shape[0]} observații, {df.shape[1]} variabile")
    print(f"\nPrimele 5 rânduri:")
    print(df.head())
    print(f"\nVariabile disponibile: {df.columns.tolist()}")
    print(f"\nValori lipsă:\n{df.isnull().sum()}")

    # Curățare date - eliminăm rândurile cu valori lipsă
    df_clean = df.dropna()

    # Selectăm doar variabilele numerice pentru analiză
    # Variabilele predictor (X): caracteristici morfologice
    predictor_columns = ['culmen_length_mm', 'culmen_depth_mm',
                         'flipper_length_mm', 'body_mass_g']

    # Verificăm dacă există coloana 'species' (sau 'Species')
    species_col = 'species' if 'species' in df_clean.columns else 'Species'

    X = df_clean[predictor_columns].values
    y = df_clean[species_col].values

    print(f"\nDupă curățare: {X.shape[0]} observații valide")
    print(f"Specii: {np.unique(y)}")
    print(f"Distribuția speciilor:\n{pd.Series(y).value_counts()}")

    return X, y, predictor_columns, df_clean


# ==============================================================================
# 2. TESTELE FISHER PENTRU PREDICTORI
# ==============================================================================

def fisher_tests_predictors(X, y, predictor_names):
    """
    Calculează puterea de discriminare și testele Fisher pentru fiecare predictor.

    Puterea de discriminare = Varianta între grupe / Varianta totală
    Test Fisher: F = (SSB / (k-1)) / (SSW / (n-k))
    unde k = număr de grupe, n = număr de observații
    """
    print("\n" + "=" * 80)
    print("PUTEREA DE DISCRIMINARE A VARIABILELOR PREDICTOR")
    print("=" * 80)

    classes = np.unique(y)
    n_classes = len(classes)
    n_samples = X.shape[0]

    results = []

    for i, pred_name in enumerate(predictor_names):
        x_var = X[:, i]

        # Calculăm media globală
        grand_mean = np.mean(x_var)

        # Varianta între grupe (Between-group variance) - SSB
        ssb = 0
        for c in classes:
            x_class = x_var[y == c]
            n_class = len(x_class)
            class_mean = np.mean(x_class)
            ssb += n_class * (class_mean - grand_mean) ** 2

        # Varianta în interiorul grupelor (Within-group variance) - SSW
        ssw = 0
        for c in classes:
            x_class = x_var[y == c]
            class_mean = np.mean(x_class)
            ssw += np.sum((x_class - class_mean) ** 2)

        # Varianta totală
        sst = np.sum((x_var - grand_mean) ** 2)

        # Puterea de discriminare (R²)
        discrimination_power = ssb / sst if sst > 0 else 0

        # Test Fisher (F-statistic)
        msb = ssb / (n_classes - 1)  # Mean Square Between
        msw = ssw / (n_samples - n_classes)  # Mean Square Within
        f_stat = msb / msw if msw > 0 else 0

        # P-value
        p_value = 1 - stats.f.cdf(f_stat, n_classes - 1, n_samples - n_classes)

        results.append({
            'Variabila': pred_name,
            'Putere_Discriminare': discrimination_power,
            'F_Statistic': f_stat,
            'P_Value': p_value,
            'Semnificativ': 'DA' if p_value < 0.05 else 'NU'
        })

    df_fisher = pd.DataFrame(results)
    df_fisher = df_fisher.sort_values('Putere_Discriminare', ascending=False)

    print(df_fisher.to_string(index=False))
    print("\nInterpretare:")
    print("- Putere de discriminare: [0, 1] - cât de bine separă variabila grupele")
    print("- F-Statistic mare + P-Value < 0.05 = variabila discriminează semnificativ")

    return df_fisher


# ==============================================================================
# 3. ANALIZA LINIARĂ DISCRIMINANTĂ (LDA)
# ==============================================================================

def apply_lda(X_train, y_train, X_test, y_test, predictor_names):
    """
    Aplică LDA și calculează toate metricile necesare.
    """
    print("\n" + "=" * 80)
    print("ANALIZA LINIARĂ DISCRIMINANTĂ (LDA)")
    print("=" * 80)

    # Standardizare date
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Aplicare LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)

    # Scoruri discriminante (proiecție în spațiul discriminant)
    X_train_lda = lda.transform(X_train_scaled)
    X_test_lda = lda.transform(X_test_scaled)

    print(f"\nNumăr de axe discriminante: {X_train_lda.shape[1]}")
    print(f"(Număr maxim de axe = min(n_predictori, n_clase - 1))")

    # Coeficienți LDA (direcțiile axelor discriminante)
    print("\n--- Coeficienții LDA (direcții discriminante) ---")
    df_coef = pd.DataFrame(
        lda.scalings_,
        index=predictor_names,
        columns=[f'LD{i + 1}' for i in range(lda.scalings_.shape[1])]
    )
    print(df_coef)

    # Explained variance ratio
    print(f"\n--- Proporția varianței explicate de fiecare axă ---")
    if hasattr(lda, 'explained_variance_ratio_'):
        for i, var in enumerate(lda.explained_variance_ratio_):
            print(f"LD{i + 1}: {var * 100:.2f}%")

    # Testele Fisher pentru axele discriminante
    print("\n--- Testele Fisher pentru axele discriminante ---")
    df_fisher_ld = fisher_tests_discriminant_axes(X_train_lda, y_train)

    # Predicții pe setul de antrenament
    y_train_pred = lda.predict(X_train_scaled)

    # Metrici de clasificare
    acc_train = accuracy_score(y_train, y_train_pred)
    kappa_train = cohen_kappa_score(y_train, y_train_pred)

    print("\n--- Clasificare pe setul de ANTRENAMENT ---")
    print(f"Acuratețe globală: {acc_train * 100:.2f}%")
    print(f"Cohen Kappa: {kappa_train:.4f}")

    # Matrice de confuzie
    cm_train = confusion_matrix(y_train, y_train_pred)
    print("\nMatricea de confuzie (antrenament):")
    print(pd.DataFrame(cm_train,
                       index=[f"Actual_{c}" for c in lda.classes_],
                       columns=[f"Pred_{c}" for c in lda.classes_]))

    # Predicții pe setul de test/aplicare
    y_test_pred = lda.predict(X_test_scaled)
    acc_test = accuracy_score(y_test, y_test_pred)
    kappa_test = cohen_kappa_score(y_test, y_test_pred)

    print("\n--- Clasificare pe setul de APLICARE (test) ---")
    print(f"Acuratețe globală: {acc_test * 100:.2f}%")
    print(f"Cohen Kappa: {kappa_test:.4f}")

    cm_test = confusion_matrix(y_test, y_test_pred)
    print("\nMatricea de confuzie (test):")
    print(pd.DataFrame(cm_test,
                       index=[f"Actual_{c}" for c in lda.classes_],
                       columns=[f"Pred_{c}" for c in lda.classes_]))

    return lda, scaler, X_train_lda, y_train, X_test_lda, y_test, y_test_pred, acc_test, kappa_test, df_fisher_ld


# ==============================================================================
# 4. TESTELE FISHER PENTRU AXELE DISCRIMINANTE
# ==============================================================================

def fisher_tests_discriminant_axes(X_lda, y):
    """
    Calculează testele Fisher pentru fiecare axă discriminantă.
    """
    n_axes = X_lda.shape[1]
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples = X_lda.shape[0]

    results = []

    for i in range(n_axes):
        x_axis = X_lda[:, i]

        # Media globală
        grand_mean = np.mean(x_axis)

        # SSB
        ssb = 0
        for c in classes:
            x_class = x_axis[y == c]
            n_class = len(x_class)
            class_mean = np.mean(x_class)
            ssb += n_class * (class_mean - grand_mean) ** 2

        # SSW
        ssw = 0
        for c in classes:
            x_class = x_axis[y == c]
            class_mean = np.mean(x_class)
            ssw += np.sum((x_class - class_mean) ** 2)

        # Test Fisher
        msb = ssb / (n_classes - 1)
        msw = ssw / (n_samples - n_classes)
        f_stat = msb / msw if msw > 0 else 0
        p_value = 1 - stats.f.cdf(f_stat, n_classes - 1, n_samples - n_classes)

        results.append({
            'Axa': f'LD{i + 1}',
            'F_Statistic': f_stat,
            'P_Value': p_value,
            'Semnificativ': 'DA' if p_value < 0.05 else 'NU'
        })

    df_fisher_ld = pd.DataFrame(results)
    print(df_fisher_ld.to_string(index=False))

    return df_fisher_ld


# ==============================================================================
# 5. DISCRIMINAREA BAYESIANĂ
# ==============================================================================

def apply_bayesian_discriminant(X_train, y_train, X_test, y_test):
    """
    Aplică Discriminarea Bayesiană (Quadratic Discriminant Analysis).
    """
    print("\n" + "=" * 80)
    print("DISCRIMINAREA BAYESIANĂ (QDA)")
    print("=" * 80)

    # Standardizare
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)

    # Predicții antrenament
    y_train_pred = qda.predict(X_train_scaled)
    acc_train = accuracy_score(y_train, y_train_pred)
    kappa_train = cohen_kappa_score(y_train, y_train_pred)

    print("\n--- Clasificare pe setul de ANTRENAMENT ---")
    print(f"Acuratețe globală: {acc_train * 100:.2f}%")
    print(f"Cohen Kappa: {kappa_train:.4f}")

    cm_train = confusion_matrix(y_train, y_train_pred)
    print("\nMatricea de confuzie (antrenament):")
    print(pd.DataFrame(cm_train,
                       index=[f"Actual_{c}" for c in qda.classes_],
                       columns=[f"Pred_{c}" for c in qda.classes_]))

    # Predicții test
    y_test_pred = qda.predict(X_test_scaled)
    acc_test = accuracy_score(y_test, y_test_pred)
    kappa_test = cohen_kappa_score(y_test, y_test_pred)

    print("\n--- Clasificare pe setul de APLICARE (test) ---")
    print(f"Acuratețe globală: {acc_test * 100:.2f}%")
    print(f"Cohen Kappa: {kappa_test:.4f}")

    cm_test = confusion_matrix(y_test, y_test_pred)
    print("\nMatricea de confuzie (test):")
    print(pd.DataFrame(cm_test,
                       index=[f"Actual_{c}" for c in qda.classes_],
                       columns=[f"Pred_{c}" for c in qda.classes_]))

    return qda, y_test_pred, acc_test, kappa_test


# ==============================================================================
# 6. VIZUALIZĂRI
# ==============================================================================

def plot_lda_results(X_lda, y, lda, title="LDA - Instanțe în spațiul discriminant"):
    """
    Plot instanțe pe axele discriminante cu centrele grupelor.
    """
    n_axes = X_lda.shape[1]
    classes = np.unique(y)

    # Calculăm centrele (mediile) pentru fiecare clasă
    centers = []
    for c in classes:
        center = np.mean(X_lda[y == c], axis=0)
        centers.append(center)
    centers = np.array(centers)

    if n_axes >= 2:
        # Plot 2D
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, c in enumerate(classes):
            mask = y == c
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                       label=c, alpha=0.6, s=50)

        # Adăugăm centrele
        ax.scatter(centers[:, 0], centers[:, 1],
                   c='red', marker='X', s=300,
                   edgecolors='black', linewidths=2,
                   label='Centre', zorder=5)

        ax.set_xlabel('LD1', fontsize=12)
        ax.set_ylabel('LD2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lda_scatter_plot.png', dpi=300)
        plt.show()

    elif n_axes == 1:
        # Plot 1D (histogram)
        fig, ax = plt.subplots(figsize=(10, 6))

        for c in classes:
            mask = y == c
            ax.hist(X_lda[mask, 0], bins=20, alpha=0.5, label=c)

        ax.set_xlabel('LD1', fontsize=12)
        ax.set_ylabel('Frecvență', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lda_histogram.png', dpi=300)
        plt.show()


def plot_discriminant_distributions(X_lda, y):
    """
    Plot distribuțiile pentru fiecare axă discriminantă.
    """
    n_axes = X_lda.shape[1]
    classes = np.unique(y)

    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]

    for i in range(n_axes):
        for c in classes:
            mask = y == c
            axes[i].hist(X_lda[mask, i], bins=20, alpha=0.5, label=c, density=True)

        axes[i].set_xlabel(f'LD{i + 1}', fontsize=12)
        axes[i].set_ylabel('Densitate', fontsize=12)
        axes[i].set_title(f'Distribuția axei LD{i + 1}', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lda_distributions.png', dpi=300)
    plt.show()


# ==============================================================================
# 7. SALVARE REZULTATE
# ==============================================================================

def save_predictions(y_test, y_pred_lda, y_pred_qda):
    """
    Salvează predicțiile într-un fișier CSV.
    """
    df_predictions = pd.DataFrame({
        'Adevarat': y_test,
        'Predictie_LDA': y_pred_lda,
        'Predictie_QDA': y_pred_qda
    })

    df_predictions.to_csv('predictii_clasificare.csv', index=False)
    print("\n✓ Predicțiile au fost salvate în 'predictii_clasificare.csv'")


# ==============================================================================
# 8. MAIN - EXECUȚIE PRINCIPALĂ
# ==============================================================================

def main():
    """
    Funcția principală care rulează întregul proiect.
    """
    print("=" * 80)
    print("PROIECT: Clasificarea speciilor de pinguini Palmer")
    print("TEMA 2: Analiză Discriminantă")
    print("=" * 80)

    # 1. ÎNCĂRCARE DATE
    # IMPORTANT: Înlocuiește cu calea către fișierul tău CSV
    filepath = 'penguins_size.csv'  # sau 'penguins_size.csv'

    try:
        X, y, predictor_names, df_clean = load_and_prepare_data(filepath)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Fișierul '{filepath}' nu a fost găsit!")
        print("\nDescarcă dataset-ul de la:")
        print("https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data")
        print("\nSau folosește altă cale către fișier.")
        return

    # 2. ÎMPĂRȚIRE DATE: 70% antrenament, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nÎmpărțire date:")
    print(f"  - Set antrenament: {X_train.shape[0]} observații")
    print(f"  - Set test/aplicare: {X_test.shape[0]} observații")

    # 3. TESTELE FISHER PENTRU PREDICTORI
    df_fisher_pred = fisher_tests_predictors(X_train, y_train, predictor_names)
    df_fisher_pred.to_csv('tabel_fisher_predictori.csv', index=False)

    # 4. APLICARE LDA
    (lda, scaler, X_train_lda, y_train, X_test_lda, y_test,
     y_pred_lda, acc_lda, kappa_lda, df_fisher_ld) = apply_lda(
        X_train, y_train, X_test, y_test, predictor_names
    )

    df_fisher_ld.to_csv('tabel_fisher_axe_discriminante.csv', index=False)

    # 5. APLICARE DISCRIMINARE BAYESIANĂ (QDA)
    qda, y_pred_qda, acc_qda, kappa_qda = apply_bayesian_discriminant(
        X_train, y_train, X_test, y_test
    )

    # 6. COMPARAȚIE MODELE
    print("\n" + "=" * 80)
    print("COMPARAȚIE MODELE - Setul de APLICARE (test)")
    print("=" * 80)
    print(f"\nLDA:")
    print(f"  - Acuratețe: {acc_lda * 100:.2f}%")
    print(f"  - Cohen Kappa: {kappa_lda:.4f}")
    print(f"\nQDA (Bayesian):")
    print(f"  - Acuratețe: {acc_qda * 100:.2f}%")
    print(f"  - Cohen Kappa: {kappa_qda:.4f}")

    if kappa_lda > kappa_qda:
        print(f"\n✓ CEL MAI BUN MODEL: LDA (Cohen Kappa = {kappa_lda:.4f})")
        best_model = "LDA"
    else:
        print(f"\n✓ CEL MAI BUN MODEL: QDA (Cohen Kappa = {kappa_qda:.4f})")
        best_model = "QDA"

    # Salvare comparație
    df_comparison = pd.DataFrame({
        'Model': ['LDA', 'QDA'],
        'Acuratete': [acc_lda, acc_qda],
        'Cohen_Kappa': [kappa_lda, kappa_qda],
        'Cel_mai_bun': [best_model == 'LDA', best_model == 'QDA']
    })
    df_comparison.to_csv('comparatie_modele.csv', index=False)

    # 7. VIZUALIZĂRI
    print("\n" + "=" * 80)
    print("GENERARE GRAFICE")
    print("=" * 80)

    plot_lda_results(X_train_lda, y_train, lda,
                     "LDA - Instanțe în spațiul discriminant (SET ANTRENAMENT)")

    plot_discriminant_distributions(X_train_lda, y_train)

    # 8. SALVARE PREDICȚII
    save_predictions(y_test, y_pred_lda, y_pred_qda)

    print("\n" + "=" * 80)
    print("✓ PROIECT FINALIZAT CU SUCCES!")
    print("=" * 80)
    print("\nFișiere generate:")
    print("  1. tabel_fisher_predictori.csv")
    print("  2. tabel_fisher_axe_discriminante.csv")
    print("  3. comparatie_modele.csv")
    print("  4. predictii_clasificare.csv")
    print("  5. lda_scatter_plot.png")
    print("  6. lda_distributions.png")
    print("\nAceste fișiere trebuie incluse în raportul Word/PDF!")


if __name__ == "__main__":
    main()