import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"/Users/deepanshuyadav/Downloads/deliveries.csv")

#Basic Informations about dataaset
print(df.head())
print(df.tail())

print("Type of dataset: ",type(df))
print("Shape:", df.shape)
df.info()
print("Datatype: ",df.dtypes)
print("Statistical summary:",df.describe())
print("Most frequent Batsman: ",df['striker'].mode())

# Data Cleaning
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
print("Missing Values: ",df.isnull().sum())
print("Remove missing values: ",df.dropna())
print("Total number of missing values: ",df.isnull().sum().sum())
df['dismissed_player'] = df['dismissed_player'].fillna("Not Out")#Filling missing values

# Groupby Operations
batsman_runs = df.groupby('striker')['total_runs'].sum()
print(batsman_runs.head())

mean_runs = df.groupby('over')['total_runs'].mean().astype(int)
print("Mean runs per over: ",mean_runs.head())

median_over = df.groupby('over')['total_runs'].median()
print("Median runs per over: ",median_over)

std_runs = df['total_runs'].std()
print("STD:", std_runs)
print("Value counts: ",df['striker'].value_counts().head())

print("Batsman: ",df['striker'].unique())
print("Total Players:", df['striker'].nunique())

#Sorting
print(batsman_runs.sort_values(ascending=True).head())
print(batsman_runs.sort_values(ascending=False).head())

#MAximum and Minimum Runsiuytre
print("Maximum Runs:", df['total_runs'].max())
print("Minimum Runs:", df['total_runs'].min())

#Normalization
df['runs_normalized'] = (df['total_runs'] - df['total_runs'].min()) / (df['total_runs'].max() - df['total_runs'].min())
print(df[['total_runs', 'runs_normalized']].head())

# Correlation
corr = df.corr(numeric_only=True)
print(corr)

high_runs = df[df['total_runs'] > 4]
print("High Scoring balls: ",high_runs.head())

# Skewness
print("Skewness: ", df['total_runs'].skew())

# Outliers using IQR method
Q1 = df['total_runs'].quantile(0.25)
Q3 = df['total_runs'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['total_runs'] < lower) | (df['total_runs'] > upper)]
print("Outliers:\n", outliers.head())


# Visualisations

# Histogram
plt.figure()
plt.hist(df['total_runs'])
plt.title("Distribution of Runs")
plt.xlabel("Runs")
plt.ylabel("Frequency")
plt.show()

# Countplot
plt.figure()
sns.countplot(x='innings', data=df)
plt.title("Innings Distribution")
plt.show()

#Bar graph of top 10 players
plt.figure()
top_batsman = batsman_runs.sort_values(ascending=False).head(10)
top_batsman.plot(kind='bar')
plt.title("Top 10 Batsmen")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.title("Correlation Matrix")
plt.show()

#Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='batting_team',y='total_runs',hue='batting_team',data=df,palette='Set2')
plt.xticks(rotation=90)
plt.title("Boxplot Runs Distribution by Team")
plt.show()


# scatter
plt.figure(figsize=(8,5))
sns.scatterplot(
    x='over',
    y='total_runs',
    hue='over',
    size='total_runs',
    data=df,
    palette='coolwarm',
    alpha=0.6
)
plt.title("Runs vs Over")
plt.show()

#Pie Chart
team_runs = df.groupby('batting_team')['total_runs'].sum()
plt.figure(figsize=(7,7))
plt.pie(team_runs,labels=team_runs.index,autopct='%1.1f%%')
plt.title("Pie Chart Team-wise Run Contribution")
plt.show()

#Line Chart
runs_per_over = df.groupby('over')['total_runs'].sum()
plt.figure()
plt.plot(runs_per_over.index, runs_per_over.values,marker='o',linestyle='--',color='red')
plt.xlabel("Over")
plt.ylabel("Total Runs")
plt.title("Runs Scored per Over")
plt.show()
