import numpy as np
import pandas as pd
from numpy.ma.extras import average


def nan_replace(x:np.ndarray):
    is_nan=np.isnan(x)#intoarce un boolean, un tensor de forma lui x in care se marcheaza pozitiile cu nan
    #print(is_nan)
    k=np.where(is_nan)#tuplu de array uri, k[0] array cu indicele liniilor unde conditia e adevarata, k[1] indicele coloanelor
    #print(k)
    #acum ca stim unde sunt nan urile vrem sa le inlocuim cu media pe coloane, adica pe directia primei axe pentru un calcul pe coloane, 1 pentru calcul pe linii
    x[k]=np.nanmean(x[:,k[1]],axis=0)#pentru fiecare nan
    #print(is_nan)

def standardizare_centrala(x:np.ndarray,scal=True,nlib=0):#degree of freedom-se scade valoarea din n
    #din fiecare coloana scadem media
    x_=x-np.mean(x,axis=0)
    if scal:
        x_=x_/np.std(x,axis=0,ddof=nlib)#in cazl nr gradelor de libertate nu este n, adaugam ca parametru aici ddof=ddof

    return x_

def tabelare_matrice(x:np.ndarray,nume_linii=None,nume_coloane=None,nume_fisier="out/data_out.csv"):
    tmp=pd.DataFrame(np.round(x),nume_linii,nume_coloane)
    tmp.to_csv(nume_fisier)

def calcul_corelatii_covariante(x:np.ndarray,y:np.ndarray):
    #y va fi vectorul de regiuni
    g=np.unique(y)
    #dictionar+tuplu
    r_v={}
    for v in g:
        x_=x[y==v,:]#il compar pe v cu y, produce un vetcor de booli
        r_v[v]=(np.corrcoef(x_,rowvar=False),np.cov(x_,rowvar=False))
    return r_v

import pandas as pd

def exporta(x, coloane=None, nume_fisier="valori.csv"):
    """
    Salvează un array (ex: x = tabel_date[variabile_numerice].values) într-un fișier CSV.

    Parametri:
      x           - matricea de valori (ex: tabel_date[variabile_numerice].values)
      coloane     - lista de nume de coloane (opțional)
      nume_fisier - numele fișierului CSV rezultat
    """
    # transformă array-ul într-un DataFrame
    df_export = pd.DataFrame(x, columns=coloane)

    # exportă ca fișier CSV
    df_export.to_csv(nume_fisier, index=False)

    print(f"✅ Fișierul '{nume_fisier}' a fost salvat cu succes.")
