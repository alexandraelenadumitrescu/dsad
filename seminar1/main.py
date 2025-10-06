import sys

import numpy as np
import pandas as pd

from functii import nan_replace, standardizare_centrala, tabelare_matrice, calcul_corelatii_covariante

pd.set_option("display.max_columns", None)
np.set_printoptions(5,threshold=sys.maxsize,suppress=True)

tabel_date=pd.read_csv("data_in/Teritorial_2022.csv",index_col=0)# indexul va fi coloana 0 in de generarea altui index


# print(type(tabel_date))
# print(tabel_date)
variabile_numerice=list(tabel_date.columns[3:])#vreau sa iau doar datele numerice, de la coloana 3 pana la sfarsit

#print(variabile_numerice,type(variabile_numerice))

x=tabel_date[variabile_numerice].values
nan_replace(x)
#print(x,type(x))
#print(np.where(np.isnan(x)))


#standardizare
x_std=standardizare_centrala(x)
#centrare
x_c=standardizare_centrala(x,scal=False)
tabelare_matrice(x,tabel_date.index,variabile_numerice,"x_std.csv")


#corelatii/covariante
r=np.corrcoef(x,rowvar=False)#rowvar pentru a anunta ca vb nu sunt pe linii ci pe coloane
tabelare_matrice(r,variabile_numerice,variabile_numerice,"date.out")
v=np.cov(x,rowvar=False)
tabelare_matrice(v,variabile_numerice,variabile_numerice,"corelatii.csv")#1 pe diag principala

r_v=calcul_corelatii_covariante(x,tabel_date["Regiunea"].values)
for v in r_v:
    print("Regiunea ",v)
    print("Corelatii ")
    print(r_v[v][0])
    tabelare_matrice(r_v[v][0],variabile_numerice,variabile_numerice,"V"+v+".csv")
    print("covarianta " )
    print(r_v[v][1])
    tabelare_matrice(r_v[v][1], variabile_numerice, variabile_numerice, "R" + v + ".csv")