import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from seaborn import scatterplot

industrie = pd.read_csv('dateIN/Industrie.csv')
populatie = pd.read_csv('dateIN/PopulatieLocalitati.csv')
data_merge = pd.merge(industrie,populatie,on='Siruta')
print(data_merge.columns)

t1 = data_merge[['Siruta', 'Localitate_x','Alimentara', 'Textila', 'Lemnului',
       'ChimicaFarmaceutica', 'Metalurgica', 'ConstructiiMasini',
       'CalculatoareElectronica', 'Mobila', 'Energetica', 'Populatie']]

industrii = ['Alimentara', 'Textila', 'Lemnului','ChimicaFarmaceutica', 'Metalurgica', 'ConstructiiMasini',
       'CalculatoareElectronica', 'Mobila', 'Energetica']

for ind in industrii:
    t1[ind] = t1[ind]/t1['Populatie']

del t1['Populatie']
cerinta1 = t1.reset_index(drop=True)
cerinta1.to_csv('dataOUT/Cerinta1.csv',index=False)

date_jud = data_merge[industrii+['Judet']].groupby(by='Judet').agg(sum)

def maxCA(t):
    x = t.values
    max_linie = np.argmax(x)
    return pd.Series(data=[t.index[max_linie],x[max_linie]],index=['Activitate','CifraAfaceri'])

cerinta2 = date_jud[industrii].apply(func=maxCA,axis=1)
cerinta2.to_csv('dataOUT/Cerinta2.csv')

#ADL:
tabel_invatare_testare = pd.read_csv('dateIN/ProiectB.csv',index_col=0)
variabile = list(tabel_invatare_testare)
predictori = variabile[:10]
tinta = variabile[11]

x_train,x_test,y_train,y_test=train_test_split(tabel_invatare_testare[predictori],
                                               tabel_invatare_testare[tinta],
                                               test_size=0.4)
obiect_ADL = LinearDiscriminantAnalysis()
obiect_ADL.fit(x_train,y_train)

#scorurile discriminante->fisier
clase = obiect_ADL.classes_
q = len(clase)
m = q - 1

scoruri_test = obiect_ADL.transform(x_test)
etichete_scoruri = ['z' + str(i+1) for i in range(m)]
scoruri_test_df = pd.DataFrame(scoruri_test,x_test.index,etichete_scoruri)
scoruri_test_df.to_csv("dataOUT/z.csv")

#Graficul scorurilor discriminante in primele 2 axe
def plot_instanta(z,y,clase,k1=0,k2=1):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1,aspect=1)
    assert isinstance(ax,plt.Axes)
    ax.set_xlabel("z" + str(k1+1))
    ax.set_ylabel("z" + str(k2+1))
    ax.set_title("Plot instanta in axele discriminante",fontsize=14,
                 color = 'b')
    scatterplot(x=z[:,k1],y=z[:,k2],hue = y,hue_order=clase,ax=ax)
    plt.show()


for i in range(2): #m-1
    for j in range(i+1,2): #m
        plot_instanta(scoruri_test,y_test,clase,i,j)

#3.Predictii in setul de test si setul de invatare->fisiere
predictie_adl_test = obiect_ADL.predict(x_test)
predictie_adl_test_df = pd.DataFrame(data={"Predictii":predictie_adl_test},
                                      index=x_test.index)
predictie_adl_test_df.to_csv("dataOUT/predict_test.csv")

x_apply = pd.read_csv('dateIN/ProiectB_apply.csv',index_col=0)
predictie_adl_apply = obiect_ADL.predict(x_apply[predictori])
predictie_adl_apply_df = pd.DataFrame(data={"Predictii apply":predictie_adl_apply},
                                      index=x_apply.index)
predictie_adl_apply_df.to_csv('dataOUT/predict_apply.csv')






