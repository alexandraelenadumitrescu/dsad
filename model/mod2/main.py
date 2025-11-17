import numpy as np
import pandas as pd

# Citirea datelor
gdp = pd.read_csv("data_in/gdp.csv", index_col=0)
populatie = pd.read_csv("data_in/populatie.csv", index_col=0)

# Extragerea coloanelor cu ani și convertirea lor la numeric
ani = [col for col in gdp.columns if col not in ['COUNTRY']]

# Convertim coloanele cu ani la numeric
for an in ani:
    gdp[an] = pd.to_numeric(gdp[an], errors='coerce')

# Cerința 1: Valorile medii pentru fiecare țară
cerinta1 = gdp[ani].mean(axis=1)
cerinta1 = pd.DataFrame({
    'CODE': gdp.index,
    'Country': gdp['COUNTRY'],
    'PIB Mediu': cerinta1.values
})
cerinta1.to_csv("Output1.csv", index=False)
print("Cerința 1 completată!")

# Cerința 2: PIB pe locuitor pentru fiecare an
# Facem merge între gdp și populație (inner join pentru a păstra doar țările cu date complete)
gdp_pop = gdp.merge(populatie[['POPULATION']], left_index=True, right_index=True, how='inner')

# Calculăm PIB pe locuitor pentru fiecare an
pib_locuitor = pd.DataFrame()
pib_locuitor['CODE'] = gdp_pop.index
pib_locuitor['COUNTRY'] = gdp_pop['COUNTRY'].values

for an in ani:
    # PIB este în milioane de euro, îl împărțim la populație pentru a obține euro/locuitor
    pib_locuitor[an] = (gdp_pop[an].values * 1000000) / gdp_pop['POPULATION'].values

cerinta2 = pib_locuitor
cerinta2.to_csv("Output2.csv", index=False)
print("Cerința 2 completată!")

# Cerința 3: Anul cu cel mai mic PIB pentru fiecare țară
anul_minim = gdp[ani].idxmin(axis=1)
cerinta3 = pd.DataFrame({
    'CODE': gdp.index,
    'COUNTRY': gdp['COUNTRY'],
    'ANUL': anul_minim.values
})
cerinta3.to_csv("Output3.csv", index=False)
print("Cerința 3 completată!")

# Cerința 4: PIB la nivel de regiune pentru fiecare an
# Facem merge cu populatie pentru a obține regiunea
gdp_regiune = gdp.merge(populatie[['REGION']], left_index=True, right_index=True, how='inner')

# Grupăm pe regiune și sumăm
cerinta4 = gdp_regiune[ani + ['REGION']].groupby(by='REGION').sum()
cerinta4.index.name = 'REGION'
cerinta4.to_csv("Output4.csv")
print("Cerința 4 completată!")

# Cerința 5: Țările cu cel mai mare PIB pe locuitor pentru fiecare regiune și an
# Folosim gdp_pop care are deja doar țările cu date complete
gdp_pop_regiune = gdp_pop.merge(populatie[['REGION']], left_index=True, right_index=True, how='inner')

# Calculăm PIB pe locuitor pentru toate țările
pib_locuitor_calc = pd.DataFrame(index=gdp_pop_regiune.index)

for an in ani:
    pib_locuitor_calc[an] = (gdp_pop_regiune[an].values * 1000000) / gdp_pop_regiune['POPULATION'].values

# Adăugăm regiunea
pib_locuitor_calc['REGION'] = gdp_pop_regiune['REGION']

# Pentru fiecare regiune și an, găsim țara cu PIB maxim pe locuitor
cerinta5_data = {}
regiuni = pib_locuitor_calc['REGION'].unique()

for regiune in regiuni:
    cerinta5_data[regiune] = {}
    date_regiune = pib_locuitor_calc[pib_locuitor_calc['REGION'] == regiune]

    for an in ani:
        # Găsim indexul (codul țării) cu valoarea maximă pentru anul respectiv
        if not date_regiune[an].isna().all() and len(date_regiune) > 0:
            tara_max = date_regiune[an].idxmax()
            cerinta5_data[regiune][an] = tara_max
        else:
            cerinta5_data[regiune][an] = ''

# Convertim la DataFrame
cerinta5 = pd.DataFrame(cerinta5_data).T
cerinta5.index.name = 'REGION'
cerinta5.to_csv("Output5.csv")
print("Cerința 5 completată!")

print("\nToate cerințele au fost procesate și salvate!")