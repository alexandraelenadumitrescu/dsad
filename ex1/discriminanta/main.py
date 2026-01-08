import pandas as pd
import io

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("perf.csv")
print(df)
print(df.isna().sum().sum())

def clean(df):
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col]=df[col].fillna(df[col].mean())
            else:
                df[col]=df[col].fillna(df[col].mode())
    return df

df=clean(df)
print(df.isna().sum().sum())
print(df)

df_perf=clean(pd.read_csv("perf.csv"))
df_new_emp=clean(pd.read_csv("newemp.csv"))


#dupa ce am curatat datele trebuie sa separam variabilele independente de variabila dependenta

vb_ind=df_perf.columns[2:]# luam coloanele de la indicele 1 pana la final
print("variabile independente: ", vb_ind)

X=df_perf[vb_ind]
Y=df_perf["Department"]
#standardizam cu standard scaler
X_apply=df_perf[vb_ind]
print("------------===================")
print(X)
print("---------------medie inainte de std si centrare" ,X.mean())
print("------------=================")
print(Y)
print(X.dtypes)

print("medie1:",X.mean())
print("medie2:",X_apply.mean())
print("->>>>>std dev", X.std())
#standardizare
scaler=StandardScaler()
X=scaler.fit_transform(X)
X_apply=scaler.transform(X_apply)
print("medie:",X.mean())
print("medie:",X_apply.mean())
print("std dev", X.std())

print("========================================")
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)#42 e ales aleator dar garanteaza reproductibilitatea e ca un seed
modelADL=LinearDiscriminantAnalysis()
modelADL.fit(x_train,y_train)


#scoruri
X_train_LDA=modelADL.transform(x_train)
X_test_LDA=modelADL.transform(x_test)



