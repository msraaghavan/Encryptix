import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore")

sns.set(style='darkgrid')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1']
background_color = '#F7F7F7'

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    le = LabelEncoder()
    data['species'] = le.fit_transform(data['species'])
    
    species_names = {0: 'Iris setosa', 1: 'Iris versicolor', 2: 'Iris virginica'}
    data['species_name'] = data['species'].map(species_names)
    
    X = data.drop(['species', 'species_name'], axis=1)
    y = data['species']
    
    return X, y, data

def plot_pie_chart(data):
    plt.figure(figsize=(12, 8))
    
    species_counts = data['species_name'].value_counts()
    labels = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    sizes = species_counts.values
    
    plt.pie(sizes, labels=labels, colors=color_palette, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.6, edgecolor='white'))
    plt.title('Distribution of Iris Species', fontsize=20, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.gcf().set_facecolor(background_color)
    plt.gca().add_artist(plt.Circle((0, 0), 0.70, fc=background_color))
    plt.gcf().canvas.manager.set_window_title('')
    plt.tight_layout()
    plt.show()

def plot_3d_scatter(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(data['sepal_length'], data['petal_width'], data['petal_length'], c=data['species'], cmap=plt.cm.get_cmap('viridis', 3), s=50, alpha=0.8)
    
    ax.set_xlabel('Sepal Length', fontweight='bold')
    ax.set_ylabel('Petal Width', fontweight='bold')
    ax.set_zlabel('Petal Length', fontweight='bold')
    ax.set_title("3D Scatter Plot of Iris Species", fontsize=20, fontweight='bold', pad=20)
    
    labels = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    handles, _ = scatter.legend_elements()
    legend1 = ax.legend(handles=handles, title="Species", labels=labels, loc="upper right")
    ax.add_artist(legend1)
    
    plt.gcf().set_facecolor(background_color)
    plt.gcf().canvas.manager.set_window_title('')
    plt.tight_layout()
    plt.show()

def plot_pairplot(data):
    sns.set_style("whitegrid")
    sns.set_palette(color_palette)
    
    data['species_name'] = data['species'].map({0: 'Iris setosa', 1: 'Iris versicolor', 2: 'Iris virginica'})
    
    g = sns.pairplot(data, hue="species_name", 
                     kind='scatter', 
                     diag_kind='kde', 
                     plot_kws={'alpha': 0.7, 's': 50, 'edgecolor': 'white'},
                     diag_kws={'shade': True, 'linewidth': 2.5, 'alpha': 0.5})
    
    fig = plt.gcf()
    fig.set_size_inches(16, 13)
    
    plt.suptitle("Pairplot of Iris Species", fontsize=24, fontweight='bold', y=0.98)
    
    for ax in g.axes.flatten():
        ax.tick_params(labelsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='bold')
        ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='bold')
    
    g._legend.remove()
    labels = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
    legend = g.fig.legend(labels=labels, title="Species", loc='lower right', bbox_to_anchor=(0.95, 0.08), ncol=1, fontsize=11, title_fontsize=13, frameon=True, fancybox=True, shadow=True)
    
    for text in legend.get_texts():
        text.set_fontweight('bold')
    g.fig.set_facecolor(background_color)
  
    plt.gcf().canvas.manager.set_window_title('')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_decision_tree(X, y):
    dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt.fit(X, y)
    
    plt.figure(figsize=(20, 12))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=["Setosa", "Versicolor", "Virginica"], rounded=True, fontsize=10, precision=2, impurity=False, proportion=True, node_ids=True, label='root')
    plt.title('Simplified Decision Tree Structure', fontsize=24, fontweight='bold', pad=20)
    plt.gcf().set_facecolor(background_color)
    plt.gcf().canvas.manager.set_window_title('')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def train_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name} Accuracy: {accuracy:.2f}")

def main(filepath):
    X, y, data = load_and_preprocess_data(filepath)
    plot_pie_chart(data)
    plot_3d_scatter(data)
    plot_pairplot(data)
    plot_decision_tree(X, y)
    train_models(X, y)

filepath = r"C:\Users\msraa\Task 3\IRIS.csv"
main(filepath)