from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Load in the data and separate the data and their labels
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the data to train on 70% and reserve 30% to test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Define the model with the hyperparameters desired
abc = AdaBoostClassifier()


# Run the model on the data
model = abc.fit(X_train, y_train)


# Make predictions on our testing data
y_pred = model.predict(X_test)


# Compare predictions to the actual labels to get an estimate for the accuracy of the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))