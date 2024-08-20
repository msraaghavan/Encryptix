import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.animation as animation

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    return data

def prepare_data_for_prediction(data):
    data_for_prediction = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    label_enc = LabelEncoder()
    data_for_prediction['Gender'] = label_enc.fit_transform(data_for_prediction['Sex'])
    data_for_prediction['Embarked'] = label_enc.fit_transform(data_for_prediction['Embarked'])
    return data_for_prediction

def scale_features(data_for_prediction):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_for_prediction[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
    X = pd.DataFrame(scaled_features, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
    X['Gender'] = data_for_prediction['Gender']
    X['Embarked'] = data_for_prediction['Embarked']
    return X

def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    return log_model

def make_predictions(log_model, X):
    y_proba = log_model.predict_proba(X)[:, 1]
    predictions = {
        'Predicted_Probability': y_proba,
        'Predicted_Survival_Logistic': log_model.predict(X)
    }
    return predictions

def combine_and_display_results(data, predictions):
    data_for_prediction = prepare_data_for_prediction(data)  
    for key, value in predictions.items():
        data_for_prediction[key] = value

    data_for_prediction_with_details = pd.concat(
        [data[['PassengerId', 'Name', 'Ticket', 'Sex', 'Age', 'SibSp', 'Parch']],
         
         data_for_prediction[['Predicted_Survival_Logistic', 'Predicted_Probability']]],
        
        axis=1
    )
    
    data_for_prediction_with_details.rename(columns={'Sex': 'Gender'}, inplace=True)

    survived_log = data_for_prediction_with_details[data_for_prediction_with_details['Predicted_Survival_Logistic'] == 1]
   
    not_survived_log = data_for_prediction_with_details[data_for_prediction_with_details['Predicted_Survival_Logistic'] == 0]

    print("\nPassengers predicted to survive = ")
    print(survived_log[['Ticket', 'Name', 'Gender', 'Age', 'SibSp', 'Parch']])

    print("\nPassengers predicted not to survive = ")
    print(not_survived_log[['Ticket', 'Name', 'Gender', 'Age', 'SibSp', 'Parch']])

    return data_for_prediction_with_details  

def plot_histogram(data_for_prediction_with_details):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_facecolor('lightgray')
    ax.patch.set_alpha(0.2)

    plt.hist(data_for_prediction_with_details['Predicted_Probability'], bins=30, color='teal', edgecolor='orange', alpha=0.8)
    plt.title('Histogram of Predicted Probabilities from Logistic Regression', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    def animate(i):
        edge_colors = ['orange', 'red', 'yellow']
        plt.gca().patch.set_edgecolor(edge_colors[i % len(edge_colors)])
        return plt.gca().patch,

    histogram = animation.FuncAnimation(plt.gcf(), animate, frames=30, interval=100)
    plt.tight_layout()
    plt.show()

filepath = 'Titanic-Dataset.csv'
data = load_and_clean_data(filepath)
data_for_prediction = prepare_data_for_prediction(data)
X = scale_features(data_for_prediction)
y = data['Survived']
log_model = train_logistic_regression(X, y)
predictions = make_predictions(log_model, X)
data_for_prediction_with_details = combine_and_display_results(data, predictions)
plot_histogram(data_for_prediction_with_details)