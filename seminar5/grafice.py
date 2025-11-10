import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import heatmap


def plot_varianta(alpha:np.ndarray,procent_minimal=80,scal=True):
    m=len(alpha)
    x=np.arange(1,m+1)
    f=plt.figure(figsize=(8,5))
    ax=f.add_subplot(1,1,1)
    ax.set_title("PLot varianta",color="b",fontsize=18)
    ax.set_xlabel("componenta",fontsize=14)
    ax.set_ylabel("componenta", fontsize=14)
    ax.set_xticks(x)
    ax.plot(x,alpha)
    ax.scatter(x,alpha,c="r",alpha=0.5,label="punct")
    #vrem sa marcam criteriile,trasam niste linii in ultima componenta care indeplineste un criteriu -kaiser - c9-cele care sunt in stanga sunt semnificative, cele care sunt in dreapta nu sunt semnificative - se aplica doar daca modelul este standardizt
    if scal:
        #cum numaram cate valori indeplinesc o anumita conditie -where
        k1=len(np.where(alpha>1)[0])#produce un tuplu cu un vector-// matrice -produce un tuplu de 2 vectori, numaram elementele din primul vector
        ax.axhline(1,c="g",label="Criteriul Kaiser")
    procent_cumulat=np.cumsum(alpha*100/sum(alpha))
    k2=np.where(procent_cumulat>procent_minimal)[0][0]+1#intoarce un tuplu cu un singur vector,care contine indicii, primul vector,primul indice
    ax.axvline(k2,c="c",label="procent minimal("+str(procent_minimal)+")")

    k3=None
    eps=alpha[:m-1]-alpha[1:]
    sigma=eps[:m-2]-eps[1:]
    print(sigma)
    negative=sigma<0#vector de booli
    if any(negative):
        #aplic criteriul
        k3=np.where(negative)[0][0]+2#indexul primei comp neg, prod un tuplu pe un vector
        ax.axvline(k3,c="m",label="Criteriul Cattel-Elbow")
    ax.legend()
    plt.savefig("graphics/PlotVarianta.png")
    return k1,k2,k3


    return k1,k2

def corelograma(t:pd.DataFrame,titlu="corelograma",vmin=-1,cmap="RdYlBu",annot=True):
    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, color="b", fontsize=18)
    heatmap(t,vmin=vmin,vmax=1,cmap=cmap,annot=annot,ax=ax)





def show():
    plt.show()