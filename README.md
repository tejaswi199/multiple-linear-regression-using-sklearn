import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv('/content/minihomeprices (2).csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
df['bedrooms']=df['bedrooms'].fillna(df['bedrooms'].median())
plt.figure(figsize=(7,7))
plt.title("Bedroom wise price increases")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
sns.barplot(x='bedrooms',y='price',data=df)
plt.show()
plt.figure(figsize=(7,7))
sns.lmplot(x='bedrooms',y='price',data=df)
plt.title("price and bedroom wise lineplot")
plt.xlabel("House Bedrooms")
plt.ylabel("House Price")
plt.show()
x=df.drop(['price'],axis=1)
y=df['price']
x['bedrooms']=x['bedrooms'].astype('int64')
mdl=LinearRegression()
mdl.fit(x,y)
