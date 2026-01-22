from itertools import count

import pandas as pd
import numpy as np
students=pd.read_csv("data_in/students.csv")
print(students.head(3))
print(students.__len__())
print(students.columns)

students_filtered=students[['Name','Class','Math']][students['Math']>8]
print(students[['Name','Class','Math']][students['Math']>8])
students_filtered.to_csv("data_out/elevi_buni_mate.csv")
print(len(students[['Name', 'Class', 'Math']][students['Math'] > 8]))

students['Average']=students[['Math','Physics','Chemistry']].mean(axis=1)
print(students)
students=students.sort_values('Average',ascending=False)
print(students)
print(students.head(3))

stud_filtered=students[['Name','Class','Math','Physics']][(students['Math']>=8)&(students['Physics']>=7)]
print(stud_filtered)

classes=pd.read_csv("data_in/classes.csv")

comb=pd.merge(students,classes,on='Class',how="inner")
print(comb)

students_complete=comb.to_csv("data_out/students_complete.csv")
# avg_group=comb.groupby('Class')
# avg_group['Math_avg']=comb.groupby('Class').mean('Math')
# avg_group=avg_group.sort_values('Math',ascending=False)
# print(avg_group)
avg_group=comb.groupby('Class').mean(True)
avg=avg_group[['Math']]
avg['medie']=avg_group[['Math']]
print(avg_group)
print(avg)
avg=avg.sort_values('medie',ascending=False)
print(avg)


avg_mate=comb.groupby('Class')['Math'].mean()
print(avg_mate)
avg_mate.columns=['Class','Math_medie']
print(avg_mate)

print("\n===========================")

clase=comb.groupby('Class').agg({
    'ID':'count',
    'Math':'mean',
    'Physics':'max',
    'Chemistry':'min'
})
clase.columns=['1','2','3','4']

print(clase)


print(comb)

top=comb.groupby('Class')['Average'].idxmax()
top=comb.loc[top]
print(top)


# # TEMPLATE:
# idx_max = df.groupby('Grupare')['Coloana_de_maxim'].idxmax()
# rezultat = df.loc[idx_max, ['coloane', 'dorite']]