import pandas as pd
from pandas.api.types import is_numeric_dtype#ne spune daca seria este sau nu numerica

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
