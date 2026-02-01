# """
# FUNCȚII ESENȚIALE PENTRU ACESTE EXERCIȚII:
# ==========================================
# 1. pd.read_csv() - citire fișier
# 2. df.groupby() - grupare date
# 3. df.merge() - îmbinare tabele
# 4. df[conditie] - filtrare
# 5. df['coloana'].max() - găsire maxim
# 6. df.to_csv() - salvare rezultat
#
# Asta e tot ce îți trebuie! Restul e logică.
# """
#
# import pandas as pd
# import numpy as np
#
# # ============================================================================
# # EXERCIȚIUL 1: Localități cu diversitate 0
# # ============================================================================
# # Citim fișierele
# diversitate = pd.read_csv('Diversitate.csv')
# coduri = pd.read_csv('Coduri_Localitati.csv')
#
# # Vedem structura datelor
# print("=== EXERCIȚIUL 1 ===")
# print("\nPrimele rânduri din Diversitate:")
# print(diversitate.head())
# print("\nPrimele rânduri din Coduri:")
# print(coduri.head())
#
# # Pas 1: Filtrăm localitățile unde TOATE valorile de diversitate sunt 0
# # Logica: grupăm pe Siruta+Localitate și verificăm dacă suma e 0
# # (dacă suma tuturor anilor = 0, înseamnă că toate valorile sunt 0)
#
# # Alegem doar coloanele cu ani (2008-2021)
# ani_coloane = [str(an) for an in range(2008, 2022)]
#
# # Calculăm suma diversității pe toți anii pentru fiecare localitate
# diversitate['suma_diversitate'] = diversitate[ani_coloane].sum(axis=1)
#
# # Filtrăm doar unde suma = 0
# diversitate_zero = diversitate[diversitate['suma_diversitate'] == 0]
#
# # Selectăm coloanele cerute
# cerinta1 = diversitate_zero[['Siruta', 'City'] + ani_coloane].copy()
#
# # Salvăm rezultatul
# cerinta1.to_csv('Cerinta1.csv', index=False)
# print(f"\n✓ Am găsit {len(cerinta1)} localități cu diversitate 0")
# print("\nExemplu rezultat:")
# print(cerinta1.head())
#
#
# # ============================================================================
# # EXERCIȚIUL 2: Județul cu diversitate medie maximă
# # ============================================================================
# print("\n\n=== EXERCIȚIUL 2 ===")
#
# # Pas 1: Îmbinăm tabelele pentru a avea județul
# # merge = îmbinare bazată pe o coloană comună (aici Siruta)
# date_complete = diversitate.merge(coduri, on='Siruta', how='left')
#
# print("\nDate după îmbinare:")
# print(date_complete[['Siruta', 'City', 'Judet', '2008', '2009']].head())
#
# # Pas 2: Calculăm diversitatea medie pentru fiecare localitate
# # (media pe toți anii)
# date_complete['diversitate_medie'] = date_complete[ani_coloane].mean(axis=1)
#
# # Pas 3: Grupăm pe județ și găsim diversitatea medie maximă în fiecare județ
# # groupby = grupare, apoi aplicăm funcția max()
# judete_max = date_complete.groupby('Judet')['diversitate_medie'].max().reset_index()
# judete_max.columns = ['Judet', 'Diversitate_Maxima']
#
# print("\nDiversitate maximă pe județe:")
# print(judete_max.head())
#
# # Pas 4: Găsim județul cu valoarea absolut maximă
# judet_castigator_idx = judete_max['Diversitate_Maxima'].idxmax()
# judet_castigator = judete_max.loc[judet_castigator_idx, 'Judet']
# valoare_maxima = judete_max.loc[judet_castigator_idx, 'Diversitate_Maxima']
#
# print(f"\n✓ Județul cu diversitate medie maximă: {judet_castigator}")
# print(f"  Valoarea diversității medii: {valoare_maxima}")
#
# # Pas 5: Găsim localitatea din acel județ care are diversitatea medie maximă
# localitatea_castigatoare = date_complete[
#     (date_complete['Judet'] == judet_castigator) &
#     (date_complete['diversitate_medie'] == valoare_maxima)
# ]
#
# # Salvăm rezultatul
# cerinta2 = localitatea_castigatoare[['Judet', 'City', 'Diversitate_Maxima']].copy()
# cerinta2.columns = ['Judet', 'Localitate', 'Diversitate_Maxima']
# cerinta2['Diversitate_Maxima'] = valoare_maxima
#
# cerinta2.to_csv('Cerinta2.csv', index=False)
# print("\nRezultat final:")
# print(cerinta2)
#
#
# # ============================================================================
# # BONUS: Explicație pas cu pas pentru înțelegere
# # ============================================================================
# print("\n\n=== EXPLICAȚIE DETALIATĂ ===")
# print("""
# EXERCIȚIUL 1 - Logica:
# ----------------------
# 1. Citești Diversitate.csv
# 2. Aduni toate valorile pe ani pentru fiecare localitate
# 3. Filtrezi unde suma = 0 (înseamnă că toate valorile sunt 0)
# 4. Selectezi coloanele cerute și salvezi
#
# EXERCIȚIUL 2 - Logica:
# ----------------------
# 1. Citești ambele fișiere
# 2. Le îmbini (merge) ca să ai județul pentru fiecare localitate
# 3. Calculezi media diversității pentru fiecare localitate
# 4. Grupezi pe județ și găsești maximul în fiecare grup
# 5. Găsești județul cu maximul absolut
# 6. Găsești localitatea din acel județ cu acea valoare
# 7. Salvezi rezultatul
#
# FUNCȚII CHEIE:
# --------------
# • merge() = îmbină două tabele bazat pe o coloană comună
# • groupby() = grupează datele și permite calcule pe grupuri
# • sum(axis=1) = sumează pe rânduri (toți anii)
# • mean(axis=1) = media pe rânduri
# • max() = găsește maximul
# • [conditie] = filtrare
#
# Cu aceste 6 funcții rezolvi 90% din problemele de date!
# """)




























import numpy as np
import pandas as pd

div = pd.read_csv("diversitateGENERAT.csv")
ani = [str(an) for an in range(2008, 2022)]

div['are_zero'] = (div[ani] == 0).any(axis=1)  # axis=1 !!!
div_zero = div[div['are_zero']]

cerinta1 = div_zero[['Siruta', 'City'] + ani].copy()
cerinta1.to_csv("Cerinta1.csv", index=False)  # index=False !!!