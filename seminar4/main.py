import pandas as pd

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
pondere_angajati=pd.DataFrame(industria_alimentara_loc[industrii+["Populatie"]].apply(func=lambda x:x[industrii]/x["Populatie"],axis=1),columns=["PondereAngajati"])#x este serie
pondere_angajati.fillna(0,inplace=True)
print(pondere_angajati)
#industria_alimentara_loc["Pondere_"]=pondere_angajati
#cerinta3=industria_alimentara_loc.sort_values(by="Pondere",ascending=False)
#cerinta3.to_csv("data_out/Cerinta3.csv")