import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

def acp(x:np.ndarray, scal=True, ddof=0):#pentru esantion n-1, pentru populatie n grade de libertate
    n,m=x.shape
    x_=x-np.mean(x,axis=0)#media variantelor calcul pe coloana
    if scal:
        x_=x_/np.std(x,axis=0,ddof=ddof)#trimit param cu valoarea ddof
    r_v=(1/(n-ddof))*x_.T@x_#@ pentru inmultire matriceala
    valp,vecp=np.linalg.eig(r_v)
    #print(valp)
    #print(vecp)
    k=np.flip(np.argsort(valp))
    print(k)
    alpha=valp[k]
    a=vecp[:,k]
    return x_,r_v,alpha,a

def tabelare_varianta(alpha:np.ndarray):#pe prima col am prima varianta, pe a doua 1+2 comulat
    m=len(alpha)
    procent=alpha*100/sum(alpha)
    t=pd.DataFrame(data={
        "Varianta":alpha,
        "varianta cumulata":np.cumsum(alpha),
        "procent varianta":procent,
        "procent cumular":np.cumsum(procent)
    },index=["C"+str(i) for i in range(m)])
    return t


def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,nume_index=None,nume_fisier_output=None):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.index.name=nume_index
    if(nume_fisier_output is not None):
        temp.to_csv(nume_fisier_output)
    return temp# sa imi si intoarca un dataframe si sa si salveze un csv daca ii dau cale pentru salvare altfel doar imi returneaza un dataframe