import pandas as pd

pd.set_option("display.max_columns",None)

from functii import nan_replace_df

t=pd.read_csv("data_in/Religious.csv",index_col=0)
#inlocuim valorile care lipsesc

nan_replace_df(t)
#print(t)
confesiuni=list(t)[2:]

coduri_localitati=pd.read_csv("data_in/Coduri_Localitati.csv",index_col=0)

t_localitate=t[confesiuni].merge()