import numpy as np
import matplotlib.pyplot as plt

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
    ax.legend()
    return k1




def show():
    plt.show()