import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
pd.set_option('display.max_colwidth', None)

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# Data Analysis

print(train.info())
print(train.info())
print(train.describe())
print(train['Survived'].value_counts())

#sns.set()
#print(train['Survived'].value_counts())
#sns.countplot('Survived', data=train)
#plt.show()
#print(train['Sex'].value_counts())
#sns.countplot('Sex', data=train)
#plt.show()
#sns.countplot('Sex', hue="Survived", data=train)
#plt.show()
#sns.countplot('Pclass', data=train)
#plt.show()

# DATA PROCESSING

print("Train Shape:", train.shape)
print("Train Shape:", test.shape)
'''
print(train.head())

print(train.isnull().sum())
print(test.isnull().sum())
'''

# throwing out the cabin column because it's incomplete

train = train.drop(columns='Cabin', axis=1)
test = test.drop(columns='Cabin', axis=1)

# replacing the missing values in Age column with mean value

train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

# finding the mode value of Embarked column and replacing in train.csv

#print(train['Embarked'].mode()[0])
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# replacing the missing values in Fare column with mean value in test.csv

test['Fare'].fillna(test['Fare'].mean(), inplace=True)

# check for missing values again
'''
print(train.isnull().sum())
print(test.isnull().sum())
'''


# Formatting Data

#print(train['Sex'].value_counts())
#print(train['Embarked'].value_counts())
train.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
#print(train.head())
test.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

# Separating features and Target

X_train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y_train = train['Survived']

X_test = test.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1)
#Y_test = test['Survived']
#print(X)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

# Logistic Regression
'''
logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, Y_train)

# accuracy
X_train_prediction = logistic_regression.predict(X_train)
print(X_train_prediction)
training_data_accuracy = round(logistic_regression.score(X_train, Y_train)*100, 2)
print("Accuracy score of training data:", training_data_accuracy, '%')
#X_test_prediction = model.predict(X_test)
#test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
#print("Accuracy score of training data:", test_data_accuracy)

ids = pd.read_csv('./test.csv')[['PassengerId']].values
Y_pred = logistic_regression.predict(X_test)
print(Y_pred)

df = {'PassengerId': ids.ravel(), 'Survived': Y_pred}
submission = pd.DataFrame(df)
submission.to_csv("my_submission.csv", index=False)

# Accuracy is around 80% which is not so good
# That is why I will opt for a faster algorithm
'''
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

#random_forest.score(X_train, Y_train)

training_data_accuracy = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Accuracy score of training data:", training_data_accuracy, '%')

ids = pd.read_csv('./test.csv')[['PassengerId']].values
df = {'PassengerId': ids.ravel(), 'Survived': Y_prediction}
submission = pd.DataFrame(df)
submission.to_csv("Submission.csv", index=False)


# Since Random Forest has accuracy around 98% that is the algorithm I am going with

