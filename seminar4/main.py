import numpy as np
import pandas as pd
from numpy.ma.core import argmax
def medie_ponderata(t:pd.DataFrame):
    x=t.values
    m=np.average(x[:,:-1],axis=0,weights=x[:,-1])#toate mai putin ultima,doar ultima
    return pd.Series(m,t.columns[:-1])

industria_alimentara=pd.read_csv("data_in/IndustriaAlimentara.csv",index_col=0)

industrii=list(industria_alimentara)[1:]

numar_angajati=industria_alimentara[industrii].apply(func=lambda x:x.sum(),axis=1)

#print(numar_angajati)

cerinta1=industria_alimentara[numar_angajati>0]#produce o serie de tip boolean folosita ca masca in selectie, selectam doar liniile cu valorile coresp
cerinta1.to_csv("data_out/Cerinta1.csv")

cerinta2=industria_alimentara[industrii].apply(func=lambda x:x/x.sum(),axis=1)
assert isinstance(cerinta2,pd.DataFrame)
cerinta2.insert(0,"Localitate",industria_alimentara["Localitate"])
cerinta2.fillna(0,inplace=True)
cerinta2.to_csv("data_out/Cerinta2.csv")#cerinta 2 e de tip dataframe

populatia=pd.read_csv("data_in/PopulatieLocalitati.csv",index_col=0)
#facem un tabel in care vom avea populatia alaturi de celalalte - jonctiune pe baza de index nu pe baza de coloana
industria_alimentara_loc=industria_alimentara.merge(populatia,left_index=True,right_index=True)
#print(industria_alimentara_loc)
pondere_angajati=pd.DataFrame(industria_alimentara_loc[industrii+["Populatie"]]
                              .apply(func=lambda x:x[industrii]/x["Populatie"],axis=1),columns=["PondereAngajati"])#x este serie
pondere_angajati.fillna(0,inplace=True)
#print(pondere_angajati)
#industria_alimentara_loc["Pondere_"]=pondere_angajati
#cerinta3=industria_alimentara_loc.sort_values(by="Pondere",ascending=False)
#cerinta3.to_csv("data_out/Cerinta3.csv")

#cerinta 3

#print(industria_alimentara_loc)
industria_alimentara_judet=industria_alimentara_loc[industrii+["Populatie","Judet"]].groupby(by="Judet").sum()
#print(industria_alimentara_judet)
assert isinstance(industria_alimentara_judet,pd.DataFrame)
cerinta3=industria_alimentara_judet.apply(func=lambda x:x[industrii].sum()/ x["Populatie"]  ,axis=1)
cerinta3.name="Pondere"
#print(type(cerinta3))

#print(cerinta3)
assert isinstance(cerinta3,pd.Series)

cerinta3.sort_values(inplace=True,ascending=False)
cerinta3.to_csv("data_out/Cerinta3.csv")


#cerinta 4
#print(industria_alimentara_judet)
cerinta4=industria_alimentara_judet[industrii].apply(func=lambda x:x.index[x.argmax()],axis=1)#argmax intoarce indicele valorii
cerinta4.name="Activitate"
cerinta4.to_csv("data_out/Cerinta4.csv")

#cerinta 5
#print(industria_alimentara_loc)
coduri_judete=pd.read_csv("data_in/Coduri_judete.csv",index_col=0)
#facem o jonctiune coduri judete si industria_alimentara_loc , pe indicativul de judet
industria_alimentara_loc_reg=industria_alimentara_loc.merge(coduri_judete,left_on="Judet",right_index=True)
#print(industria_alimentara_loc_reg)
cerinta5=industria_alimentara_loc_reg[industrii+["Populatie","Regiune"]].groupby(by="Regiune").apply(func=medie_ponderata,include_groups=False)
print(cerinta5)
cerinta5.to_csv("data_out/Cerinta5.csv")

#cerinta 6
