import pandas as pd
import numpy as np






# 📚 SET DE EXERCIȚII PANDAS - PENTRU EXAMEN
# 📊 DATE DE LUCRU
# Dataset 1: Vanzari.csv
# Produs,Categorie,Preturi,Cantitati,Data,Judet
# Laptop,Electronice,"1200,1150,1300","5,3,7",2024-01-15,CJ
# Telefon,Electronice,"800,750,900","10,15,12",2024-01-16,BV
# Masa,Mobila,"500,450,550","2,4,3",2024-01-17,CJ
# Scaun,Mobila,"150,140,160","8,10,6",2024-01-18,IS
# Tableta,Electronice,"600,650,700","5,6,4",2024-01-19,BV
# Dataset 2: Angajati.csv
# ID,Nume,Departament,Salariul,Varsta,Oras,Bonusuri
# 1,Popescu Ion,IT,5000,28,Cluj,"500,300,200"
# 2,Ionescu Maria,HR,4000,35,Brasov,"400,350,300"
# 3,Georgescu Ana,IT,5500,30,Cluj,"600,400,500"
# 4,Marinescu Paul,Sales,4500,42,Iasi,"700,650,600"
# 5,Dumitrescu Laura,IT,5200,26,Brasov,"550,450,400"
# Dataset 3: Studenti.csv
# Nume,Note_Mate,Note_Info,Note_Engleza,Absente,An
# Andrei,"8,9,7,10","9,8,10","7,8,9","2,1,0,3",2
# Maria,"10,9,10,9","10,10,9","9,10,8","0,0,1,0",1
# Ion,"6,7,5,8","7,6,8","8,7,9","5,4,3,2",3
# Elena,"9,10,8,9","8,9,10","10,9,9","1,0,0,1",2

# 🎯 NIVEL 1 - ÎNCEPĂTOR (Citire, Explorare, Selecție)
# Exercițiul 1: Explorare Date
# Fișier: Vanzari.csv
#
# Citește fișierul
angajati_pd=pd.read_csv("data_in/angajati.csv")
studenti_pd=pd.read_csv("data_in/studenti.csv")
vanzari_pd=pd.read_csv("data_in/vanzari.csv")
# Afișează primele 3 rânduri
# Afișează dimensiunea dataset-ului (shape)
# Afișează tipurile de date pentru fiecare coloană
# Câte produse unice există?

print(vanzari_pd.head(3))
print(vanzari_pd[:][:3])
print(vanzari_pd.shape)
print(vanzari_pd.dtypes)
for col in vanzari_pd:
    print(type(vanzari_pd[col][1]))

print(vanzari_pd[vanzari_pd.columns[0]].unique().__len__())













#print(angajati_pd)

#print(angajati_pd.head(3))
print(angajati_pd[angajati_pd.columns[:3]][:3])
print(angajati_pd[:][:3])
print(angajati_pd.shape)
print(angajati_pd.dtypes)
for col in angajati_pd:
        print(type(angajati_pd[col][1]))




#
# Exercițiul 2: Selecție Simplă
# Fișier: Vanzari.csv
#
# Selectează doar coloanele Produs și Categorie
# Afișează toate produsele din categoria Electronice
# Afișează produsele din județul CJ
# Salvează rezultatul în selectie.csv

print(vanzari_pd)
print(vanzari_pd.columns)
print(vanzari_pd[['Produs','Categorie']])
print(vanzari_pd[vanzari_pd['Categorie']=='Electronice'])
print(vanzari_pd['Produs'][vanzari_pd['Categorie']=='Electronice'])
print(vanzari_pd['Produs'][vanzari_pd['Judet']=='CJ'])
a=vanzari_pd['Produs'][vanzari_pd['Judet']=='CJ']
a.to_csv("data_out/selectie.csv")


#
# Exercițiul 3: Filtrare Condiționată
# Fișier: Angajati.csv
#
# Afișează angajații cu salariul > 4500
# Afișează angajații din departamentul IT
# Afișează angajații din Cluj cu vârsta < 30
# Câți angajați sunt din Brașov?
print(angajati_pd)
print(angajati_pd[['Nume','Salariul']][angajati_pd['Salariul']>4500])
print(angajati_pd[['Nume','Departament']][angajati_pd['Departament']=='IT'])
print(angajati_pd[['Nume','Oras','Varsta']][(angajati_pd['Oras']=='Cluj')&(angajati_pd['Varsta']<30)])
#print(angajati_pd[['Nume','Oras','Varsta']][(angajati_pd['Oras']=='Cluj')])
print(angajati_pd[['Nume','Oras']][angajati_pd['Oras']=='Brasov'])
print(angajati_pd['Nume'][angajati_pd['Oras']=='Brasov'].__len__())
#
#
# 🔥 NIVEL 2 - MEDIU (Split, Apply, Calcule)
# Exercițiul 4: Split și Sumă
# Fișier: Vanzari.csv
# Cerință A (1 punct):
#
# Split coloana Preturi în listă de valori
# Calculează prețul mediu pentru fiecare produs
# Salvează: Produs, Pret_Mediu în preturi_medii.csv

print(vanzari_pd)
print(vanzari_pd['Preturi'].str.split())
#print(vanzari_pd['Preturi'].astype(int).groupby('Produs')['Preturi'].mean())
vanzari_pd['Preturi']=vanzari_pd['Preturi'].astype(int)
df['Preturi']
print(vanzari_pd.dtypes)


#
# Cerință B (2 puncte):
#
# Split coloana Cantitati în listă
# Calculează cantitatea totală vândută pentru fiecare produs
# Calculează venitul total: Pret_Mediu * Cantitate_Totala
# Salvează rezultatul în venituri.csv
#
# Exercițiul 5: Calcule pe Rânduri
# Fișier: Angajati.csv
# Cerință A (1 punct):
#
# Split coloana Bonusuri
# Calculează suma totală a bonusurilor pentru fiecare angajat
# Adaugă coloana Total_Bonusuri
#
# Cerință B (2 puncte):
#
# Calculează Salariu_Anual = Salariul * 12 + Total_Bonusuri
# Găsește angajatul cu cel mai mare salariu anual
# Salvează top 3 angajați cu cele mai mari salarii anuale
#
# Exercițiul 6: Medii pe Studenti
# Fișier: Studenti.csv
# Cerință (2 puncte):
#
# Pentru fiecare student, calculează media la fiecare materie
# Calculează media generală (media celor 3 medii)
# Calculează totalul absențelor
# Salvează: Nume, Medie_Generala, Total_Absente în rezultate.csv
# Sortează descrescător după medie
#
#
# 🚀 NIVEL 3 - AVANSAT (GroupBy, Agregare, Merge)
# Exercițiul 7: Analiză pe Județ
# Fișier: Vanzari.csv
# Cerință A (2 puncte):
#
# Calculează venitul total pentru fiecare produs (sum(Preturi) * sum(Cantitati))
# Grupează după Judet și calculează:
#
# Venitul total pe județ
# Numărul de produse vândute pe județ
# Venitul mediu pe produs în fiecare județ
#
#
#
# Cerință B (2 puncte):
#
# Identifică produsul cu cel mai mare venit în fiecare județ
# Salvează: Judet, Produs_Top, Venit_Max în top_judete.csv
#
# Exercițiul 8: Analiză Departament
# Fișier: Angajati.csv
# Cerință A (2 puncte):
#
# Grupează după Departament
# Calculează pentru fiecare departament:
#
# Salariul mediu
# Vârsta medie
# Numărul de angajați
# Suma totală bonusuri (după split)
#
#
#
# Cerință B (2 puncte):
#
# Grupează după Oras
# Găsește orașul cu cel mai mare salariu mediu
# Găsește orașul cu cei mai mulți angajați IT
# Salvează statistici pe oraș în statistici_orase.csv
#
# Exercițiul 9: Categoria Dominantă
# Fișier: Vanzari.csv
# Cerință (3 puncte):
#
# Calculează venitul pentru fiecare produs
# Grupează după Categorie și Judet
# Calculează venitul total pentru fiecare combinație Categorie-Județ
# Identifică categoria dominantă (cu cel mai mare venit) în fiecare județ
# Salvează: Judet, Categorie_Dominanta, Venit în dominante.csv
#
#
# 💪 NIVEL 4 - EXPERT (Probleme Complexe)
# Exercițiul 10: Analiză Temporală
# Fișier: Vanzari.csv
# Cerință (3 puncte):
#
# Convertește coloana Data la tip datetime
# Extrage luna și ziua săptămânii
# Calculează venitul total pe lună
# Calculează venitul mediu pe zi a săptămânii
# Identifică luna cu cele mai mari vânzări
# Salvează graficul vânzărilor lunare în vanzari_luna.csv
#
# Exercițiul 11: Clasament Studenti
# Fișier: Studenti.csv
# Cerință A (2 puncte):
#
# Calculează media la fiecare materie
# Identifică materia la care fiecare student are cea mai mare medie
# Calculează câți studenți au media generală > 8
# Grupează pe An și calculează media generală pe an
#
# Cerință B (2 puncte):
#
# Calculează un scor: Scor = Medie_Generala * 10 - Total_Absente * 0.5
# Clasează studenții după scor
# Identifică top 3 studenți
# Salvează clasamentul complet în clasament.csv
#
# Exercițiul 12: Merge și Analiză Complexă
# Creează 2 fișiere noi:
# Produse.csv:
# Produs,Producator,Cost_Productie
# Laptop,Dell,800
# Telefon,Samsung,500
# Masa,IKEA,300
# Scaun,IKEA,100
# Tableta,Apple,400
# Clienti.csv:
# ID_Vanzare,Produs,Client,Rating
# 1,Laptop,Popescu,5
# 2,Telefon,Ionescu,4
# 3,Masa,Georgescu,5
# 4,Scaun,Marinescu,3
# 5,Tableta,Dumitrescu,4
# Cerință (3 puncte):
#
# Merge Vanzari.csv cu Produse.csv pe Produs
# Calculează profitul: (Pret_Mediu - Cost_Productie) * Cantitate_Totala
# Merge rezultatul cu Clienti.csv
# Calculează profitul mediu per producător
# Calculează rating-ul mediu per categorie
# Identifică produsul cel mai profitabil
# Salvează analiza completă în analiza_finala.csv
#
#
# 🎓 NIVEL 5 - PROBLEME TIP EXAMEN
# Exercițiul 13: PROBLEMA COMPLEXĂ - Industrie
# Creează: Industrie2.csv
# Siruta,Localitate,Alimentara,Textila,Chimica,Metalurgica,Judet
# 1001,Cluj-Napoca,"100,150,200","50,60,70","80,90,100","120,130,140",CJ
# 1002,Brasov,"200,250,300","70,80,90","100,110,120","150,160,170",BV
# 1003,Iasi,"150,180,220","60,70,80","90,100,110","130,140,150",IS
# 1004,Cluj,"80,100,120","40,50,60","60,70,80","90,100,110",CJ
# Cerință A (2 puncte):
#
# Split toate coloanele de activități industriale
# Calculează cifra de afaceri totală pentru fiecare activitate și localitate
# Salvează: Siruta, Localitate, Total_Alimentara, Total_Textila, etc.
#
# Cerință B (3 puncte):
#
# Identifică activitatea industrială dominantă în fiecare localitate (cea cu cifra cea mai mare)
# Grupează pe Judet și calculează:
#
# Cifra totală de afaceri pe județ
# Activitatea dominantă la nivel de județ
# Numărul de localități din județ
#
#
# Salvează rezultatul în industrie_judete.csv
#
# Cerință C (2 puncte):
#
# Calculează procentul din cifra totală pentru fiecare activitate în fiecare localitate
# Sortează localitățile după cifra totală descrescător
# Salvează top 10 localități în top_localitati.csv