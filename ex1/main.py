import pandas as pd
import io
df=pd.read_csv("date_in/Teritorial_2022.csv")

print(df)
print(df.isnull())
print(df.isnull().sum())
print(df.isnull().sum().sum())
df=df.fillna(df.select_dtypes(include=['number']).mean())
print(df.isnull().sum().sum())
print(df.mean(numeric_only=True))
print(df.describe().loc['std'])
