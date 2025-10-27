import pandas as pd

from functii import nan_replace_df

set_date=pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv",index_col=0)
nan_replace_df(set_date)

variabile_observate=list(set_date)[3:]
x=set_date[variabile_observate].values