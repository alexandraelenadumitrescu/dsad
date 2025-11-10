import sys

import numpy as np
import pandas as pd

from functii import nan_replace_df, acp, tabelare_varianta, salvare_ndarray
from grafice import plot_varianta, show, corelograma, scatterplot

np.set_printoptions(3,sys.maxsize,suppress=True)
pd.set_option("display.max_columns",None)


set_date=pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv",index_col=0)
nan_replace_df(set_date)

variabile_observate=list(set_date)[3:]
x=set_date[variabile_observate].values

#print(x)
x_,r_v,alpha,a=acp(x)

t_r=salvare_ndarray(r_v,variabile_observate,variabile_observate,"Indicatori","data_out/R.csv")#matrice de co

#analiza variantei componentelor
t_varianta=tabelare_varianta(alpha)
t_varianta.round(3).to_csv("data_out/Varianta.csv")
k1,k2,k3=plot_varianta(alpha)
#kmin=min(k1,k2,k3)#cel mai restrictiv criteriu


corelograma(t_r,annot=len(variabile_observate)<10)
show()

#show()

#Analiza corelatiilor dintre vb obs si componente
c=x_@a#a-matr vect prop
s=c/np.sqrt(alpha)
#etichete componente sub forma de lista
etichete_componente=list(t_varianta.index)
t_c=salvare_ndarray(c,set_date.index,etichete_componente,set_date.index.name,"data_out/C.csv")#tabelul componentelor
t_s=salvare_ndarray(s,set_date.index,etichete_componente,set_date.index.name,"data_out/S.csv")#tabelul componentelor
#calculam corelatiile
#Rk=ak*alphal^1/2 doar pe date standardizate
n,m=x.shape
r_XC=np.corrcoef(x_,c,rowvar=False)[:m,m:]#corelatie intre x si ce
t_r_XC=salvare_ndarray(
    r_XC,variabile_observate,etichete_componente,"Indicatori","data_out/R_XC.csv"
)
corelograma(t_r_XC,"Corelatii variabile- componente",annot=m<10)
#aNALAIZA SCORUIRLOR
scatterplot(t_c,titlu="Plot componente")
scatterplot(t_s)
show()