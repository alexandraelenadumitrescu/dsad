import numpy as np
import pandas as pd
import io
import seaborn as sb
from matplotlib import pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
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
X_train_lda=modelADL.transform(x_train)
X_test_lda=modelADL.transform(x_test)

df_XTtrainLDA = pd.DataFrame(
    X_train_lda,
    columns=["LD" + str(i+1) for i in range(X_train_lda.shape[1])]
)
df_XTtrainLDA.to_csv("XTrainLDA.csv", index=False)

df_XTestLDA = pd.DataFrame(
    X_test_lda,
    columns=["LD" + str(i+1) for i in range(X_test_lda.shape[1])]
)
df_XTestLDA.to_csv("XTestLDA.csv", index=False)

# Evaluarea model
Y_pred = modelADL.predict(x_test)

variance = modelADL.explained_variance_ratio_
print("Variatia: ", variance)

matrice_confuzie = confusion_matrix(y_test, Y_pred)
df_matrice = pd.DataFrame(
    matrice_confuzie,
    index=np.unique(y_test),
    columns=np.unique(y_test)
)
df_matrice.to_csv("MatriceConfuzie.csv")

acuratetea = accuracy_score(y_test, Y_pred)
print("Acuratetea: ", acuratetea)

acuratetea_per_class = matrice_confuzie.diagonal() / matrice_confuzie.sum(axis=1)
acurateteaMedie = np.mean(acuratetea_per_class)
print("Acuratetea medie: ", acurateteaMedie)

# Predictii pe setul apply
df_apply_clean=pd.read_csv("newemp.csv")
Y_apply = modelADL.predict(X_apply)
df_apply_clean["Departament"] = Y_apply
df_apply_clean.to_csv("ApplyPredicted.csv")

silhouetteScore = silhouette_score(X_test_lda, y_test)
print("Scor Silhouette: ", silhouetteScore)

for label in np.unique(y_train):
    sb.kdeplot(X_train_lda[y_train == label, 0], label = label)
plt.title("Distributia pe axele discriminante")
plt.show()
