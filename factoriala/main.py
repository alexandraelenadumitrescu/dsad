import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df_indici_diversitate=pd.read_csv("data_in/Diversitate.csv")
df_coduri_loc=pd.read_csv("data_in/Coduri_Localitati.csv")

# prelucram setul de date pentru a avea variabile numerice
indici_diversitate=df_indici_diversitate.iloc[:,2:]

#verificam daca avem valori lipsa
if indici_diversitate.isnull().values.any():
    indici_diversitate=indici_diversitate.fillna(indici_diversitate.mean())

#B-1 varianta factorilor,procentul de varianta, procentul cumulat de varianta
#ATENTIE!! NU AVEM NEVOIE DE BARLETT SI KMO PENTRU VARIANTE, DOAR DACA NI SE CER

#PAS 1- testul Barlett pentru a vedea daca datele sunt potrivite pentru analiza factoriala
#-se aplica mereu pe date brute, nestandardizate
#chi,p_value=calculate_bartlett_sphericity(indici_diversitate)
#print(f'p_value: {p_value:.3f}')
#daca p_value<0.05 atunci datele noastre sunt corelate intre ele si modelul este valid

#PAS 2- aplicam testul KMO- masoara cat de puternic sunt corelate variabilele
#-verifica daca corelatiile sunt suficient de puternice pentru factori comuni
#kmo_all,kmo_model=calculate_kmo(indici_diversitate)
#print(f"kmo_all: {kmo_all}")
#print(f'kmo_model: {kmo_model:.3f}')

#------APLICARE FACTORABILITATE-------

#fa_temp=FactorAnalyzer(rotation="varimax")
#e ca si cum determini componentele la pca, doar ca aici determini numarul de factori
#fa_temp.fit(indici_diversitate)
# ev,v=fa_temp.get_eigenvalues()
# nr_fa=sum(ev>1)
#print(f'nr_fa: {nr_fa}')
#facem analiza pe numarul corecti de factori
fa_final=FactorAnalyzer(n_factors=2,rotation="varimax")#fara rotatie doar se pune None
fa_final.fit(indici_diversitate)
#-------VARIANTA CU ROTATIE-------
varianta_cu_rotatie=fa_final.get_factor_variance()
print(varianta_cu_rotatie)
#get_factor_variance imi calculeaza automat varianta, procentul variantelor, procentul cumulat
df_varianta=pd.DataFrame({

  'Varianta factorilor':varianta_cu_rotatie[0],
   'Procent varianta': varianta_cu_rotatie[1],
    'procent cumulat': varianta_cu_rotatie[2]
})
df_varianta.to_csv("data_out/Varianta.csv")


#B-2 corelatii
corelatii_rotatie=fa_final.loadings_
df_corelatii=pd.DataFrame(
    corelatii_rotatie,
    columns=[f'F{i+1}' for i in range(corelatii_rotatie.shape[1])],

)
df_corelatii.to_csv("data_out/Corelatii.csv")

#B-3 cercul corelatiilor
loadings=fa_final.loadings_[:,0:2]
variabile=indici_diversitate.columns
plt.figure(figsize=(10,10))
#cerc
cerc=plt.Circle((0,0),1,color="black",fill=False)
plt.gca().add_patch(cerc)

#axe
plt.axhline(0,color="black",linestyle="--")
plt.axvline(0,color="black",linestyle="--")

#sageti si etichete
for i, var in enumerate(variabile):
    plt.arrow(0,0,loadings[i,0],loadings[i,1],head_width=0.03,color='red')
    plt.text(loadings[i,0]*1.1,loadings[i,1]*1.1,var)

plt.xlim(-1.2,1.2)
plt.ylim(-1.2,1.2)
plt.title("Cercul corelatiilor")
plt.xlabel("F1")
plt.ylabel("F2")
plt.savefig("data_out/Cercul_corelatiilor.png")

#CORELOGRAMA CORELATII
plt.figure(figsize=(10,10))
sns.heatmap(loadings,annot=True,cmap="Blues",center=0,
            vmin=-1,vmax=1,xticklabels=['F1','F2'],yticklabels=indici_diversitate.columns)
plt.savefig("data_out/Corelograma.png")

#CALCUL COMUNALITATI SI VARIANTA SPECIFICA
comunalitati=fa_final.get_communalities()# cate de multa varianta este ecplicata de factori
varianta_specifica=1-comunalitati#cata varianta ramane neexplicata
print(f'Comunalitati:{comunalitati}')
print(f'Varianta specifica:{varianta_specifica}')
df_comunalitati = pd.DataFrame({
    'An': indici_diversitate.columns,
    'Comunalitati': comunalitati,
    'Varianta_specifica': varianta_specifica
})
df_comunalitati.to_csv("data_out/Comunalitati.csv")

#PLOT COMUNALITATI SI VARIANTA SPECIFICA
ani=indici_diversitate.columns
x=np.arange(len(ani))
plt.figure(figsize=(10,10))
plt.bar(x-0.2,comunalitati,width=0.5,label='Comunalitati',color='blue')
plt.bar(x+0.2,varianta_specifica,width=0.5,label='Varianta specifica',color='red')
plt.xlabel("Ani")
plt.ylabel("Comunalitati")
plt.legend()
plt.savefig("data_out/ComunalitatiVarianta.png")

#CALCUL SCORURI
scoruri_cu_rotatie=fa_final.transform(indici_diversitate)
df_scoruri=pd.DataFrame(
    scoruri_cu_rotatie,
    index=df_indici_diversitate['Siruta'],
    columns=[f'S{i+1}' for i in range(scoruri_cu_rotatie.shape[1])]
)
df_scoruri.to_csv("data_out/Scoruri.csv")


