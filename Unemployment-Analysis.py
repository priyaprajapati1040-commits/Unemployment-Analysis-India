import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment in India.csv")

# First 5 rows
print(df.head())
print(df.info())

print(df.describe())

print(df.isnull().sum())
df.columns = df.columns.str.strip()

print(df.columns)
df.columns = df.columns.str.strip()

print(df.columns)
df.dropna(inplace=True)

print(df.isnull().sum())
df['Date'] = pd.to_datetime(df['Date'])

print(df.head())
plt.figure(figsize=(8,5))
sns.histplot(df['Estimated Unemployment Rate (%)'], bins=20, kde=True)

plt.title("Distribution of Unemployment Rate")
plt.show()
state_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

state_unemployment = state_unemployment.sort_values(ascending=False)

print(state_unemployment)
plt.figure(figsize=(12,6))

state_unemployment.plot(kind='bar')

plt.title("Average Unemployment Rate by State")
plt.ylabel("Unemployment Rate (%)")

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(
    x='Area',
    y='Estimated Unemployment Rate (%)',
    data=df
)

plt.title("Urban vs Rural Unemployment")
plt.show()
monthly = df.groupby(df['Date'].dt.month)[
    'Estimated Unemployment Rate (%)'
].mean()

plt.figure(figsize=(10,5))

monthly.plot(marker='o')

plt.title("Monthly Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")

plt.show()
covid_period = df[df['Date'] >= '2020-03-01']

plt.figure(figsize=(12,6))

sns.lineplot(
    x='Date',
    y='Estimated Unemployment Rate (%)',
    data=covid_period
)

plt.title("COVID-19 Impact on Unemployment")
plt.show()
plt.figure(figsize=(8,5))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()