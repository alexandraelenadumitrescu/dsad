from operator import countOf

from func import exporta,standardizare_centrala,tabelare_matrice,calcul_corelatii_covariante,nan_replace
import pandas as pd
import  numpy as np
tabel_date=pd.read_csv("in/Teritorial_2022.csv",index_col=0)
tabel_date.to_csv("out/tabel.csv")
variabile_numerice=list(tabel_date.columns[3:])#intoarce o lista de nume de coloane de la 3 incolo,vreau sa iau doar datele numerice, de la coloana 3 pana la sfarsit
x=tabel_date[variabile_numerice].values
exporta(x,variabile_numerice,"viz.csv")
print(pd.DataFrame(x,columns=variabile_numerice).describe())
x_std=standardizare_centrala(x)
print(pd.DataFrame(x_std,columns=variabile_numerice).describe())
#centrare
x_c=standardizare_centrala(x,scal=False)
print(pd.DataFrame(x_c,columns=variabile_numerice).describe())
tabelare_matrice(x,tabel_date.index,variabile_numerice,"out/x_std.csv")




#print(x,type(x))
#print(countOf)
#print(np.isnan(x).sum())
#nan_replace(x)
#print(x)
#print(np.isnan(x).sum())
#print(variabile_numerice)
#print(variabile_numerice,type(variabile_numerice))
#print(tabel_date)