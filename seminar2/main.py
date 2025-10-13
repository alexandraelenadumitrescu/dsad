import pandas as pd

pd.set_option("display.max_columns",None)

from functii import nan_replace_df, calcul_ponderi

t=pd.read_csv("data_in/Religious.csv",index_col=0)
#inlocuim valorile care lipsesc

nan_replace_df(t)
#print(t)
confesiuni=list(t)[2:]

coduri_localitati=pd.read_csv("data_in/Coduri_Localitati.csv",index_col=0)

t_localitate=t[confesiuni].merge(coduri_localitati["County"],left_index=True,right_index=True)
#print(t_localitate)

t_judet=t_localitate.groupby(by="County").sum()#pot sa am si mai multe criterii, o lista de coloane
t_judet.to_csv("data_out/Religious_County.csv")

#afilierea cod judet cu denumirea regiunii
coduri_judete=pd.read_csv("data_in/Coduri_Judete.csv",index_col=0)

#aducem in tabelul in care avem datele criteriul de grupare-regiunea
assert isinstance(t_judet,pd.DataFrame)
t_judet=t_judet.merge(coduri_judete["Regiune"],left_index=True,right_index=True)

#sumarizarea
t_regiune=t_judet.groupby(by="Regiune").sum()
t_regiune.to_csv("data_out/Religious_Region.csv")




#calculam ponderile la niv localitatilor si judetelor
t_p_loc=t[confesiuni].apply(func=calcul_ponderi, axis=1)# aplicam pe linii
#t_p_loc.to_csv("data_out/Religious_p_loc.csv")
#inseram in tploc si numele localitatii
assert isinstance(t_p_loc,pd.DataFrame)
t_p_loc.insert(0,"City",t["City"])
t_p_loc.to_csv("data_out/Religious_p_loc.csv")


#o sa trimitem functia prin expresia lambda
#lista de param : si codul
t_judet = t_judet.apply(pd.to_numeric, errors='coerce')

t_p_jud=t_judet.apply(func=lambda x: x/x.sum(), axis=1)

t_p_jud.to_csv("data_out/Religious_p_county.csv")