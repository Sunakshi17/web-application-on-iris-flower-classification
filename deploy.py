from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
dataset = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal length", "sepal width", "petal length", "petal width", "Class_labels"]
iris_data = pd.read_csv(dataset, names=column_names)

# Seperate features and target
X = iris_data.iloc[:, :-1].values
Y = iris_data.iloc[:, -1].values

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input values from the request
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

     # Create the user input array
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Get the user input for test size
    test_size = float(request.form['test_size'])

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    # Load the trained models
    model_LR = LogisticRegression()
    model_LR.fit(X_train, y_train)

    model_DT = DecisionTreeClassifier()
    model_DT.fit(X_train, y_train)

    model_SVM = SVC()
    model_SVM.fit(X_train, y_train)

    model_RF = RandomForestClassifier()
    model_RF.fit(X_train, y_train)

    model_NB = GaussianNB()
    model_NB.fit(X_train, y_train)

    # Use the trained models to make predictions on the user input
    lr_prediction = model_LR.predict(user_input)
    dt_prediction = model_DT.predict(user_input)
    svm_prediction = model_SVM.predict(user_input)
    rf_prediction = model_RF.predict(user_input)
    nb_prediction = model_NB.predict(user_input)

    # Calculate the accuracy of each model
    lr_accuracy = accuracy_score(y_test, model_LR.predict(X_test)) * 100
    dt_accuracy = accuracy_score(y_test, model_DT.predict(X_test)) * 100
    svm_accuracy = accuracy_score(y_test, model_SVM.predict(X_test)) * 100
    rf_accuracy = accuracy_score(y_test, model_RF.predict(X_test)) * 100
    nb_accuracy = accuracy_score(y_test, model_NB.predict(X_test)) * 100

    # Return the results as the response
    return render_template('result.html',
                           lr_prediction=lr_prediction,
                           dt_prediction=dt_prediction,
                           svm_prediction=svm_prediction,
                           rf_prediction=rf_prediction,
                           nb_prediction=nb_prediction,
                           lr_accuracy=lr_accuracy,
                           dt_accuracy=dt_accuracy,
                           svm_accuracy=svm_accuracy,
                           rf_accuracy=rf_accuracy,
                           nb_accuracy=nb_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
