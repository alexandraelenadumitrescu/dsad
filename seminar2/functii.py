import pandas as pd
from pandas.api.types import is_numeric_dtype#ne spune daca seria este sau nu numerica
import numpy as np

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        #verificam daca exista valori lipsa pe coloana
        if any(t[c].isna()):#pentru o serie produce o serie de tip boolean
            if is_numeric_dtype(t[c]):
                #daca e numerica inlocuim valorile lipsa cu media
                #t[c].fillna(t[c].mean(),inplace=True)
                t[c].fillna({c:t[c].mean()}, inplace=True)
            else:
                t.fillna({c:t[c].mode()[0]},inplace=True)#mode returneaza un vector de module, il alegem pe primul cel mai frecvent



# def calcul_ponderi(t:pd.Series):
#     print(t)
#     exit(0)

def calcul_ponderi(t:pd.Series):
    return t/t.sum()#operanti in serie rezultat in serie

def diversitate(t:pd.Series):
    #
    #p=t.iloc[1:].values #iloc pentru adresare prin index, implicit ar fi prin numar dar noi avem cheie literata
    p=t.values

    print(p)
    p[p==0]=1 #produce un vector de booleeni
    print(type(p[0]))
    #p=p[p>0]
    shannon=-np.sum(p*np.log2(p))
    simpson=1-np.sum(p*p)
    inv_simpson=1/np.sum(p*p)

    return pd.Series([shannon,simpson,inv_simpson],["Shannon","Simpson","InvSimpson"])