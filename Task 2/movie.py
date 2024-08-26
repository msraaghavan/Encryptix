import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter("ignore")
sns.set_style("darkgrid")

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    df['Year'] = df['Year'].apply(lambda y: int(str(y)[1:5]) if isinstance(y, str) else np.nan)
    
    df['Duration'] = df['Duration'].apply(lambda d: int(str(d).split(" ")[0]) if isinstance(d, str) else np.nan)
    
    df['Votes'] = df['Votes'].replace("$5.16M", 516)
    df['Votes'] = pd.to_numeric(df['Votes'].str.replace(',', ''))
    
    df = df.dropna()
    
    return df

def prepare_data_for_analysis(df):
    X = df[['Year', 'Duration', 'Genre', 'Director', 'Actor 1']]
    y = df['Votes']
    
    label_enc = LabelEncoder()
    for column in ['Genre', 'Director', 'Actor 1']:
        X[column] = label_enc.fit_transform(X[column])
    
    return X, y

def train_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def visualize_top_worst_movies(df):
    top = df.sort_values("Votes", ascending=False).head(10)[['Name', 'Year', 'Duration', 'Votes', 'Director', 'Actor 1']].reset_index(drop=True)
    worst = df.sort_values("Votes", ascending=False).dropna().tail(10)[['Name', 'Year', 'Duration', 'Votes', 'Director', 'Actor 1']].reset_index(drop=True)
    
    plt.figure(figsize=(12, 6), dpi=100)
    bars = plt.barh(top['Name'], top['Votes'], color='skyblue', edgecolor='black')
    plt.bar_label(bars, fmt='%d', padding=3)
    plt.title('Top 10 Movies Based on Votes', fontsize=18, fontweight='bold')
    plt.xlabel('Number of Votes', fontsize=14)
    plt.ylabel('Movie Names', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.show()

    plt.figure(figsize=(12, 6), dpi=100)
    bars = plt.barh(worst['Name'], worst['Votes'], color='salmon', edgecolor='black')
    plt.bar_label(bars, fmt='%d', padding=3)
    plt.title('Worst 10 Movies Based on Votes', fontsize=18, fontweight='bold')
    plt.xlabel('Number of Votes', fontsize=14)
    plt.ylabel('Movie Names', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.show()


filepath = 'IMDb Movies India.csv'

df = load_and_clean_data(filepath)

X, y = prepare_data_for_analysis(df)

model, X_test, y_test = train_linear_regression(X, y)

visualize_top_worst_movies(df)