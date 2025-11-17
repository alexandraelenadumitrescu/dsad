import numpy as np
import pandas as pd
import os

# Asigurăm existența directorului data_out
os.makedirs("data_out", exist_ok=True)

# Citim datele
industrie = pd.read_csv("data_in/Industrie.csv", index_col=0)
populatie = pd.read_csv("data_in/PopulatieLocalitati.csv", index_col=0)

# Lista activităților industriale
activitati = ['Alimentara', 'Textila', 'Lemnului', 'ChimicaFarmaceutica',
              'Metalurgica', 'ConstructiiMasini', 'CalculatoareElectronica',
              'Mobila', 'Energetica']

# ============================================================================
# CERINȚA 1: Cifra de afaceri totală pe localitate
# ============================================================================
print("Procesare cerința 1...")
cifra_afaceri_totala = industrie[activitati].sum(axis=1)
output1 = pd.DataFrame({
    'Siruta': industrie.index,
    'Localitate': industrie['Localitate'],
    'Cifra de afaceri': cifra_afaceri_totala
})
output1.to_csv("data_out/Output1.csv", index=False)
print("✓ Output1.csv creat cu succes!")

# ============================================================================
# CERINȚA 2: Cifra de afaceri pe locuitor (sortare descrescătoare)
# ============================================================================
print("Procesare cerința 2...")
# Join între industrie și populație
industrie_pop = industrie.merge(populatie[['Localitate', 'Populatie']],
                                 left_index=True, right_index=True,
                                 suffixes=('', '_pop'))

# Calculăm CA/Loc
industrie_pop['CA_Total'] = industrie_pop[activitati].sum(axis=1)
industrie_pop['CF/Loc'] = industrie_pop['CA_Total'] / industrie_pop['Populatie']

# Sortăm descrescător și salvăm
output2 = industrie_pop[['Localitate', 'CF/Loc']].sort_values(
    by='CF/Loc', ascending=False
)
output2.insert(0, 'Siruta', output2.index)
output2.to_csv("data_out/Output2.csv", index=False)
print("✓ Output2.csv creat cu succes!")

# ============================================================================
# CERINȚA 3: Activitatea dominantă pe localitate (doar localități cu CA > 0)
# ============================================================================
print("Procesare cerința 3...")
# Filtrăm doar localitățile cu activitate (CA > 0)
industrie_activ = industrie[cifra_afaceri_totala > 0].copy()

# Găsim activitatea cu CA maximă pentru fiecare localitate
activitate_dominanta = industrie_activ[activitati].idxmax(axis=1)

output3 = pd.DataFrame({
    'Siruta': industrie_activ.index,
    'Localitate': industrie_activ['Localitate'],
    'Activitate': activitate_dominanta
})
output3.to_csv("data_out/Output3.csv", index=False)
print("✓ Output3.csv creat cu succes!")

# ============================================================================
# CERINȚA 4: Cifra de afaceri pe activitate la nivel de județ
# ============================================================================
print("Procesare cerința 4...")
# Join cu populație pentru a avea codul județului
industrie_judet = industrie.merge(populatie[['Judet']],
                                   left_index=True, right_index=True)

# Agregăm pe județ (groupby + sum)
ca_judet = industrie_judet.groupby('Judet')[activitati].sum()
ca_judet.index.name = 'Judet'
ca_judet.to_csv("data_out/Output4.csv")
print("✓ Output4.csv creat cu succes!")

# ============================================================================
# CERINȚA 5: CA totală pe locuitor la nivel de județ (sortare descrescătoare)
# ============================================================================
print("Procesare cerința 5...")
# Calculăm CA totală pe județ
ca_total_judet = ca_judet.sum(axis=1)

# Calculăm populația pe județ
populatie_judet = populatie.groupby('Judet')['Populatie'].sum()

# CA pe locuitor la nivel de județ
ca_loc_judet = ca_total_judet / populatie_judet
ca_loc_judet.name = 'CA/Loc'

# Sortăm descrescător
output5 = ca_loc_judet.sort_values(ascending=False)
output5.to_csv("data_out/Output5.csv")
print("✓ Output5.csv creat cu succes!")

print("\n" + "="*60)
print("✓✓✓ Toate cele 5 cerințe au fost procesate cu succes! ✓✓✓")
print("="*60)
print("Fișierele generate în directorul data_out/:")
print("  - Output1.csv (CA totală pe localitate)")
print("  - Output2.csv (CA/locuitor pe localitate, sortare desc)")
print("  - Output3.csv (Activitate dominantă pe localitate)")
print("  - Output4.csv (CA pe activitate la nivel județ)")
print("  - Output5.csv (CA/locuitor la nivel județ, sortare desc)")