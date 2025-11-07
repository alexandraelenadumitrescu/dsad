import os
import numpy as np
from func import tabelare_matrice

# Verifică directorul curent
print("Director curent:", os.getcwd())

# Verifică dacă directorul out/ există
print("Directorul 'out' există?", os.path.exists('out'))

# Verifică dacă este folder
print("Este director?", os.path.isdir('out'))

# Încearcă să salvezi fișierul
matrice = np.array([[1, 2, 3], [4, 5, 6]])
tabelare_matrice(matrice, nume_fisier="out/data_out.csv")

# Verifică dacă fișierul a fost creat
print("Fișier creat în out/?", os.path.exists('out/data_out.csv'))

# Listează tot ce este în directorul out/
print("\nConținutul directorului 'out':")
if os.path.exists('out'):
    print(os.listdir('out'))

# Caută fișierul în directorul principal
print("\nFișierul e în directorul principal?", os.path.exists('data_out.csv'))